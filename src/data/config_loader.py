"""Stdlib-only YAML-subset parser + config loader.

The whole project reads ``config.yaml`` through :func:`load_config` so that
hyper-parameters live in exactly one place. ``pyyaml`` is deliberately NOT
required: the data pipeline must run on stock Python, so this module parses
the restricted YAML subset used by ``config.yaml``:

* ``#`` comments and blank lines
* ``key: scalar``  (int, float, bool, null, quoted/plain strings)
* nested maps via 2-space indentation
* lists of scalars via ``- item``

Paths in the config are resolved relative to the repository root
(the directory containing ``config.yaml``).
"""

from pathlib import Path


def _strip_comment(line: str) -> str:
    in_single = in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or line[idx - 1] in (" ", "\t"):
                return line[:idx].rstrip()
    return line.rstrip()


def _parse_scalar(text: str):
    s = text.strip()
    if s in ("", "~", "null", "Null", "NULL", "None"):
        return None
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_yaml_subset(text: str) -> dict:
    """Parse the restricted YAML subset used by config.yaml (stdlib only)."""
    lines = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw)
        if stripped.strip():
            lines.append(stripped)

    root: dict = {}
    # stack of (indent_level, container); containers are dict or list
    stack = [(-1, root)]
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content == "-" or content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"List item under non-list container: {line!r}")
            item = content[1:].strip()
            parent.append(_parse_scalar(item))
            i += 1
            continue

        if ":" not in content:
            raise ValueError(f"Cannot parse line: {line!r}")
        key, _, value = content.partition(":")
        key = key.strip().strip("'\"")
        value = value.strip()

        # Look ahead: is there a more-indented block below?
        child_indent = None
        if i + 1 < n:
            nxt = lines[i + 1]
            ni = len(nxt) - len(nxt.lstrip(" "))
            if ni > indent:
                child_indent = ni

        if value == "" and child_indent is not None:
            first_child = lines[i + 1].strip()
            container = (
                [] if (first_child == "-" or first_child.startswith("- ")) else {}
            )
            if not isinstance(parent, dict):
                raise ValueError(f"Nested block under list item: {line!r}")
            parent[key] = container
            stack.append((indent, container))
        else:
            if isinstance(parent, list):
                raise ValueError(f"Mapping entry inside list: {line!r}")
            parent[key] = _parse_scalar(value)
        i += 1
    return root


def repo_root() -> Path:
    """Repository root = directory containing config.yaml."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str = "config.yaml") -> dict:
    """Load config.yaml (path may be relative to repo root)."""
    p = Path(path)
    if not p.is_absolute():
        candidate = repo_root() / path
        if candidate.exists():
            p = candidate
    return parse_yaml_subset(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    cfg = load_config()
    print(f"loaded {len(cfg)} top-level keys from config.yaml")
    for key in sorted(cfg):
        print(f"  {key}: {cfg[key]!r}")
