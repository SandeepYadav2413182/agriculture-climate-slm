"""Weather providers for the climate module.

* :class:`WeatherProvider` — abstract interface; ``get_current(location)``
  returns a dict with keys ``temperature_c``, ``humidity_pct``,
  ``rainfall_mm_24h``, ``wind_kph``, ``forecast_summary``, ``source``.
* :class:`MockWeatherProvider` — deterministic **synthetic** weather derived
  from ``hash(location)``. Clearly labeled as mock data; useful for tests,
  demos and offline development. NOT real observations.
* :class:`OpenMeteoProvider` — real data from the Open-Meteo APIs
  (geocoding + forecast) over plain ``urllib``; only active when
  ``open_meteo_enabled: true`` in the config.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Dict


class WeatherProvider(ABC):
    @abstractmethod
    def get_current(self, location: str) -> Dict:
        """Current weather for ``location``.

        Returns a dict with keys: ``temperature_c``, ``humidity_pct``,
        ``rainfall_mm_24h``, ``wind_kph``, ``forecast_summary``, ``source``.
        """


class MockWeatherProvider(WeatherProvider):
    """Deterministic synthetic weather from ``hash(location)``.

    The values are pseudo-random but stable for a given location string, so
    tests and demos are reproducible. **This is mock data — not a real
    observation or forecast.** The ``source`` field says ``"mock (synthetic)"``
    to make that explicit downstream.
    """

    def get_current(self, location: str) -> Dict:
        digest = hashlib.sha256(location.strip().lower().encode("utf-8")).digest()
        pick = lambda lo, hi, i: lo + (digest[i] / 255.0) * (hi - lo)

        temperature_c = round(pick(18.0, 38.0, 0), 1)
        humidity_pct = round(pick(35.0, 95.0, 1), 0)
        rainfall_mm_24h = round(pick(0.0, 25.0, 2) ** 1.6 / 25.0, 1)
        wind_kph = round(pick(4.0, 32.0, 3), 1)
        if rainfall_mm_24h > 8:
            summary = "Heavy rain likely in the next 24 hours."
        elif rainfall_mm_24h > 2:
            summary = "Light showers possible in the next 24 hours."
        elif temperature_c > 35:
            summary = "Hot and dry conditions expected."
        else:
            summary = "Partly cloudy, no significant rain expected."
        return {
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "rainfall_mm_24h": rainfall_mm_24h,
            "wind_kph": wind_kph,
            "forecast_summary": summary,
            "source": "mock (synthetic)",
        }


class OpenMeteoProvider(WeatherProvider):
    """Real weather via the Open-Meteo geocoding + forecast APIs (urllib)."""

    GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout_s: int = 10) -> None:
        self.timeout_s = timeout_s

    def _get_json(self, url: str) -> Dict:
        import json
        import urllib.parse
        import urllib.request

        with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_current(self, location: str) -> Dict:
        import urllib.parse

        geo_qs = urllib.parse.urlencode({"name": location, "count": 1,
                                         "language": "en", "format": "json"})
        geo = self._get_json(f"{self.GEOCODE_URL}?{geo_qs}")
        results = geo.get("results") or []
        if not results:
            raise ValueError(f"Open-Meteo could not geocode location: {location!r}")
        lat, lon = results[0]["latitude"], results[0]["longitude"]

        fc_qs = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon, "current": ",".join(
                ["temperature_2m", "relative_humidity_2m", "precipitation",
                 "wind_speed_10m", "weather_code"]),
            "daily": "precipitation_sum",
            "timezone": "auto",
        })
        fc = self._get_json(f"{self.FORECAST_URL}?{fc_qs}")
        cur = fc.get("current", {})
        daily = fc.get("daily", {})
        rain_24h = (daily.get("precipitation_sum") or [0.0])[0]

        return {
            "temperature_c": round(float(cur.get("temperature_2m", 0.0)), 1),
            "humidity_pct": round(float(cur.get("relative_humidity_2m", 0.0)), 0),
            "rainfall_mm_24h": round(float(rain_24h), 1),
            "wind_kph": round(float(cur.get("wind_speed_10m", 0.0)), 1),
            "forecast_summary": f"Weather code {cur.get('weather_code')} at observation time.",
            "source": "open-meteo",
        }


def get_weather_provider(cfg: dict) -> WeatherProvider:
    """Return the configured weather provider.

    ``weather_provider: openmeteo`` requires ``open_meteo_enabled: true``,
    otherwise a ``RuntimeError`` is raised. Any other value (default
    ``mock``) yields the synthetic :class:`MockWeatherProvider`.
    """
    name = str(cfg.get("weather_provider", "mock")).lower()
    if name in ("openmeteo", "open-meteo", "open_meteo"):
        if not bool(cfg.get("open_meteo_enabled", False)):
            raise RuntimeError(
                "OpenMeteoProvider requested but disabled: set "
                "'open_meteo_enabled: true' in config.yaml to enable live weather."
            )
        return OpenMeteoProvider()
    return MockWeatherProvider()
