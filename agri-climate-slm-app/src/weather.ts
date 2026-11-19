export interface WeatherCtx {
  location: string;
  summary: string;
  note: string;
  mock: boolean;
}

const CONDITIONS: Array<{ label: string; temp: [number, number] }> = [
  { label: "clear skies", temp: [28, 36] },
  { label: "partly cloudy", temp: [26, 33] },
  { label: "light rain", temp: [24, 30] },
  { label: "overcast", temp: [25, 31] },
  { label: "hot and dry", temp: [33, 41] },
  { label: "humid", temp: [27, 34] },
];

function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h;
}

/** Mock provider: deterministic, clearly labeled, no live data claimed. */
function mockWeather(location: string): WeatherCtx {
  const h = hashStr(location.toLowerCase());
  const c = CONDITIONS[h % CONDITIONS.length];
  const temp = c.temp[0] + (h % (c.temp[1] - c.temp[0] + 1));
  return {
    location,
    summary: `${c.label}, around ${temp}°C near ${location}`,
    note: "Mock weather context for demonstration. Enable the Open-Meteo provider for live data.",
    mock: true,
  };
}

/**
 * Optional live provider (Open-Meteo, no API key). Disabled by default.
 * Enable with WEATHER_PROVIDER=open-meteo.
 */
export async function openMeteoWeather(location: string): Promise<WeatherCtx | null> {
  try {
    const geo = await fetch(
      `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(location)}&count=1`
    ).then((r) => r.json() as Promise<{ results?: Array<{ latitude: number; longitude: number; name: string }> }>);
    const place = geo.results?.[0];
    if (!place) return null;
    const wx = await fetch(
      `https://api.open-meteo.com/v1/forecast?latitude=${place.latitude}&longitude=${place.longitude}&current=temperature_2m,weathercode&timezone=auto`
    ).then((r) => r.json() as Promise<{ current?: { temperature_2m: number; weathercode: number } }>);
    const t = wx.current?.temperature_2m;
    return {
      location: place.name,
      summary: t !== undefined ? `currently around ${Math.round(t)}°C near ${place.name}` : `live conditions near ${place.name}`,
      note: "Live weather via Open-Meteo.",
      mock: false,
    };
  } catch {
    return null;
  }
}

export async function resolveWeather(location?: string): Promise<WeatherCtx | null> {
  const loc = location?.trim();
  if (!loc) return null;
  if (process.env.WEATHER_PROVIDER === "open-meteo") {
    const live = await openMeteoWeather(loc);
    if (live) return live;
  }
  return mockWeather(loc);
}
