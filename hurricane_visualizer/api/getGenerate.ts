import { SimulationConfig, ScenarioData } from "../types";

export async function generateHurricaneScenario(
  config: SimulationConfig
): Promise<ScenarioData> {
  const res = await fetch("http://localhost:8000/api/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(config),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Backend error");
  }

  return res.json();
}
