"""Lazy model wrapper for the fine-tuned agriculture SLM.

:class:`ModelWrapper` never loads weights in ``__init__``. Heavy imports
(``torch``, ``transformers``, ``peft``, ``bitsandbytes``) are guarded inside
methods so that importing this module — and running the whole pipeline in
retrieval-only mode — works on machines without any ML stack installed.

Expected layout (from ``config.yaml``):

* ``model_name`` / ``tokenizer_name`` — HF hub id of the base model.
* ``quantization: 4bit`` — load in 4-bit via bitsandbytes when available.
* ``adapter_dir`` — optional LoRA adapter attached on top of the base model.
* ``merged_dir`` — optional fully-merged checkpoint (preferred if present).
* ``max_new_tokens``, ``temperature`` — generation hyper-parameters.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config, repo_root  # noqa: E402


def _try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


class ModelWrapper:
    """Thin wrapper around a causal LM with lazy loading.

    Parameters
    ----------
    cfg: dict
        Config dict from :func:`src.data.config_loader.load_config`.
    adapter_path: str | None
        Path to a LoRA adapter directory. Defaults to ``adapter_dir`` from
        the config; ``None`` means "base model only".
    """

    def __init__(self, cfg=None, adapter_path=None):
        self.cfg = cfg or load_config()
        self.adapter_path = adapter_path or self.cfg.get("adapter_dir")
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Availability probing (no heavy imports leak out of these methods)
    # ------------------------------------------------------------------
    def _weights_reachable(self) -> tuple[bool, str]:
        """Check whether some usable weights exist locally.

        We deliberately do NOT attempt a hub download here: ``available()``
        must be a cheap, offline-safe probe. Returns (ok, reason).
        """
        root = repo_root()
        candidates = []
        merged = self.cfg.get("merged_dir")
        if merged:
            candidates.append(("merged_dir", root / merged))
        if self.adapter_path:
            candidates.append(("adapter_dir", root / self.adapter_path))
        base = self.cfg.get("model_name")
        # A local snapshot of the base model (e.g. HF cache dir configured
        # by the deployment) also counts; the hub id itself is not enough
        # because downloading is not attempted in this probe.
        local_base = root / "models" / "base"
        candidates.append(("models/base", local_base))

        for label, path in candidates:
            if path.exists() and any(path.iterdir()):
                return True, f"{label} -> {path}"
        return False, (
            "no local weights found (checked merged_dir, adapter_dir, "
            "models/base); hub download is not attempted by available()"
        )

    def available(self) -> bool:
        """True iff torch+transformers import AND local weights are reachable."""
        if _try_import("torch") is None:
            return False
        if _try_import("transformers") is None:
            return False
        ok, _ = self._weights_reachable()
        return ok

    def availability_reason(self) -> str:
        """Human-readable explanation of why the model is (un)available."""
        missing = []
        if _try_import("torch") is None:
            missing.append("torch not installed")
        if _try_import("transformers") is None:
            missing.append("transformers not installed")
        ok, reason = self._weights_reachable()
        if not ok:
            missing.append(reason)
        if missing:
            return "; ".join(missing)
        return f"ready ({reason})"

    # ------------------------------------------------------------------
    # Lazy loading + generation
    # ------------------------------------------------------------------
    def _load(self):
        torch = _try_import("torch")
        transformers = _try_import("transformers")
        if torch is None or transformers is None:
            raise RuntimeError(
                "Cannot load model: torch and transformers are required. "
                f"Probe result: {self.availability_reason()}"
            )
        ok, reason = self._weights_reachable()
        if not ok:
            raise RuntimeError(f"Cannot load model: {reason}")

        model_name = self.cfg.get("model_name")
        tokenizer_name = self.cfg.get("tokenizer_name", model_name)
        quantization = str(self.cfg.get("quantization", "")).lower()

        load_kwargs = {"device_map": "auto"}
        if quantization == "4bit":
            bnb = _try_import("bitsandbytes")
            if bnb is None:
                raise RuntimeError(
                    "Config requests 4-bit quantization but bitsandbytes "
                    "is not installed. Install bitsandbytes or change "
                    "`quantization` in config.yaml."
                )
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization in ("8bit",):
            load_kwargs["load_in_8bit"] = True
        elif quantization in ("bf16", "fp16"):
            load_kwargs["torch_dtype"] = (
                torch.bfloat16 if quantization == "bf16" else torch.float16
            )

        # Prefer a merged checkpoint when present, else base + adapter.
        root = repo_root()
        merged = self.cfg.get("merged_dir")
        base_id_or_path = model_name
        if merged and (root / merged).exists() and any((root / merged).iterdir()):
            base_id_or_path = str(root / merged)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            base_id_or_path, **load_kwargs
        )

        adapter_dir = root / self.adapter_path if self.adapter_path else None
        if (adapter_dir and adapter_dir.exists() and any(adapter_dir.iterdir())
                and not (merged and (root / merged).exists())):
            peft = _try_import("peft")
            if peft is None:
                raise RuntimeError(
                    f"LoRA adapter found at {adapter_dir} but peft is not "
                    "installed; cannot attach it."
                )
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, str(adapter_dir))

        self._model.eval()

    def generate(self, prompt: str) -> str:
        """Generate a completion for *prompt*, loading weights on first call.

        Raises
        ------
        RuntimeError
            With an informative message when the model is unavailable
            (missing libraries or weights) instead of a bare ImportError.
        """
        if not self.available():
            raise RuntimeError(
                "Model unavailable in this environment: "
                f"{self.availability_reason()}. "
                "Run the pipeline in retrieval-only mode instead."
            )
        if self._model is None:
            self._load()

        torch = _try_import("torch")
        max_new_tokens = int(self.cfg.get("max_new_tokens", 256))
        temperature = float(self.cfg.get("temperature", 0.7))

        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        # Strip the prompt tokens; decode only the new continuation.
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    w = ModelWrapper()
    print("available:", w.available())
    print("reason:", w.availability_reason())
