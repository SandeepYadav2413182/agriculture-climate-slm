"""Turn a weather dict into a short agronomy advisory paragraph.

Flags considered:
* spraying suitability — wind speed and recent/upcoming rain
* irrigation need — rainfall vs. temperature
* heat / frost stress — temperature extremes
"""

from __future__ import annotations

from typing import Dict, Optional


def build_climate_context(weather: Dict, crop: Optional[str] = None,
                          crop_stage: Optional[str] = None) -> str:
    """Render a short advisory paragraph from a weather dict.

    ``weather`` must carry the keys produced by
    :class:`src.climate.weather.WeatherProvider` (temperature_c, humidity_pct,
    rainfall_mm_24h, wind_kph, forecast_summary, source).
    """
    temp = float(weather.get("temperature_c", 0.0))
    humidity = float(weather.get("humidity_pct", 0.0))
    rain = float(weather.get("rainfall_mm_24h", 0.0))
    wind = float(weather.get("wind_kph", 0.0))
    summary = str(weather.get("forecast_summary", ""))
    source = str(weather.get("source", "unknown"))

    subject = f"{crop} ({crop_stage} stage)" if crop else "the crop"
    if crop_stage and not crop:
        subject = f"the crop ({crop_stage} stage)"

    lines = [f"Current conditions ({source}): {temp:.1f}\u00b0C, "
             f"{humidity:.0f}% humidity, {rain:.1f} mm rain in last 24h, "
             f"wind {wind:.1f} kph. {summary}"]

    # --- spraying suitability -----------------------------------------
    if wind > 20 or rain > 5:
        spray = (f"Spraying is NOT advisable for {subject} right now: "
                 f"{'wind is too strong for even coverage' if wind > 20 else ''}"
                 f"{' and ' if wind > 20 and rain > 5 else ''}"
                 f"{'recent rain will wash off the spray' if rain > 5 else ''}. "
                 "Wait for calmer, drier conditions.")
    elif wind > 12:
        spray = (f"Spraying {subject} is marginal: moderate wind may cause drift. "
                 "Spray early morning or late evening if needed.")
    else:
        spray = (f"Conditions are suitable for spraying {subject}: light wind "
                 "and little recent rain. Prefer early morning application.")

    # --- irrigation need ------------------------------------------------
    if rain >= 8:
        irrig = (f"No irrigation needed for {subject}: recent rainfall "
                 "is sufficient.")
    elif temp >= 35 and rain < 2:
        irrig = (f"Irrigate {subject} promptly: high temperature with almost "
                 "no rain means high evapotranspiration.")
    elif rain < 2:
        irrig = (f"Consider a light irrigation for {subject}: dry spell with "
                 "minimal rainfall.")
    else:
        irrig = (f"Soil moisture for {subject} looks adequate; monitor before "
                 "irrigating.")

    # --- stress flags ----------------------------------------------------
    stress = []
    if temp >= 38:
        stress.append("HEAT STRESS risk: provide shade/mulch where feasible "
                      "and avoid midday field operations")
    if temp <= 4:
        stress.append("FROST RISK: protect sensitive crops overnight "
                      "(smoke/mulch/irrigation)")
    if humidity >= 90 and temp >= 25:
        stress.append("high humidity + warmth favors fungal disease: scout "
                      "for leaf spots and blight")

    paragraph = " ".join([lines[0], spray, irrig] + stress)
    return paragraph
