import React, { useState, useEffect } from 'react';
import TrackVisualizer from './components/TrackVisualizer';
import MetricsCard from './components/MetricsCard';
import TimeSeriesChart from './components/TimeSeriesChart';
import { generateHurricaneScenario } from './api/getGenerate';
import { ScenarioData, GenerateScenarioRequest, SFDiffDimension, SFDiffVariant, AVAILABLE_MODELS } from './types';

const DATASETS = [
  { id: "AL122005", name: "Hurricane Katrina (2005)", desc: "Cat 5, Gulf of Mexico Landfall" },
  { id: "AL182012", name: "Hurricane Sandy (2012)", desc: "Cat 3, East Coast Anomalous Turn" },
  { id: "AL152017", name: "Hurricane Maria (2017)", desc: "Cat 5, Caribbean Devastation" },
  { id: "AL092022", name: "Hurricane Ian (2022)", desc: "Cat 5, Rapid Intensification" },
  { id: "AL132023", name: "Hurricane Lee (2023)", desc: "Cat 5, Open Atlantic Recurvature" },
];

const VARIANTS: SFDiffVariant[] = ['S4', 'S5'];
const DIMENSIONS: SFDiffDimension[] = ['2D', '3D', '4D'];

const App: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [scenarioData, setScenarioData] = useState<ScenarioData | null>(null);
  
  // Simulation Configuration State
  const [selectedDataset, setSelectedDataset] = useState<string>(DATASETS[0].id);
  const [selectedVariant, setSelectedVariant] = useState<SFDiffVariant>('S4');
  const [selectedDimension, setSelectedDimension] = useState<SFDiffDimension>('3D');
  const [activeModels, setActiveModels] = useState<string[]>(['ar', 'kf']); // Start with basic baselines
  
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    
    const request: GenerateScenarioRequest = {
    sim_config: {
        sfdiffVariant: selectedVariant,
        sfdiffDimension: selectedDimension,
        activeModels: activeModels,
    },
    model_id: selectedVariant+"_"+selectedDimension, // e.g. "s4_2D"
    dataset_id: selectedDataset, // e.g. "AL092017_t=36"
    };

    try {
      const data = await generateHurricaneScenario(request);
      setScenarioData(data);
    } catch (err: any) {
        setError(err.message || "Failed to generate simulation.");
    } finally {
      setLoading(false);
    }
  };

  const addModel = (modelId: string) => {
    if (!activeModels.includes(modelId)) {
        setActiveModels([...activeModels, modelId]);
        // Auto regenerate when model is added? Or just set state and let user click run?
        // For better UX in a "simulation" app, usually you click Run. 
        // But to make the (+) button feel instant, we might need to re-run or just update state.
        // Let's update state and trigger a run implicitly to fetch data for the new model.
        // Ideally we would just fetch the missing model data, but simplifying to re-run.
        // We'll set a flag or just let the user click run to see new results.
        // Actually, if I don't run, the MetricsCard won't show the new data because 'scenarioData' doesn't have it.
        // Let's trigger generate immediately in next effect or manually.
        // State update is async, so we'll use a useEffect or just call a wrapper.
        // Let's force a re-run but we need the new state. 
    }
  };
  
  // Effect to trigger run when models change? No, that might be too aggressive if user is configuring.
  // But the (+) button in MetricsCard implies "Show me this now". 
  // We will simply update the active list. The MetricsCard will only show models that exist in scenarioData.
  // So we MUST re-fetch if we want to show it.
  useEffect(() => {
      if (activeModels.length > 2) { // crude check to skip initial mount double fetch if strict mode
          handleGenerate();
      }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeModels]);


  // Initial load
  useEffect(() => {
    handleGenerate();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run once on mount

  return (
    <div className="min-h-screen bg-background text-gray-100 p-4 md:p-8 font-sans">
      {/* Header */}
      <header className="max-w-7xl mx-auto mb-8 flex flex-col md:flex-row justify-between items-center border-b border-white/10 pb-6">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-accent bg-clip-text text-transparent">
            SFDiff Explorer
          </h1>
          <p className="text-gray-400 mt-1">
            Score-Based Diffusion for Hurricane Forecasting
          </p>
        </div>
        <div className="flex gap-4 mt-4 md:mt-0">
          <div className="text-right hidden md:block">
            <p className="text-xs text-gray-500 uppercase">Backend Status</p>
            <p className="font-mono text-green-400 flex items-center gap-1 justify-end">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
              Connected (Python)
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto">
        {/* Control Bar - Enhanced */}
        <div className="bg-surface rounded-xl p-6 mb-8 shadow-lg border border-white/5 space-y-4">
          <h2 className="text-sm font-bold text-gray-300 uppercase tracking-wide mb-2 border-b border-white/5 pb-2">
            Simulation Configuration
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {/* Dataset Select */}
            <div className="md:col-span-2">
              <label className="block text-xs text-gray-400 mb-1 ml-1">
                Historical Hurricane Scenario
              </label>
              <div className="relative">
                <select
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  className="w-full bg-background border border-gray-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-primary transition-colors appearance-none"
                >
                  {DATASETS.map((ds) => (
                    <option key={ds.id} value={ds.id}>
                      {ds.name}
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
                  <svg
                    className="fill-current h-4 w-4"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                  >
                    <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                  </svg>
                </div>
              </div>
            </div>

            {/* Variant Select */}
            <div>
              <label className="block text-xs text-gray-400 mb-1 ml-1">
                SFDiff Architecture
              </label>
              <div className="relative">
                <select
                  value={selectedVariant}
                  onChange={(e) =>
                    setSelectedVariant(e.target.value as SFDiffVariant)
                  }
                  className="w-full bg-background border border-gray-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-accent transition-colors appearance-none"
                >
                  {VARIANTS.map((v) => (
                    <option key={v} value={v}>
                      {v} Model
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
                  <svg
                    className="fill-current h-4 w-4"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                  >
                    <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                  </svg>
                </div>
              </div>
            </div>

            {/* Dimension Select */}
            <div>
              <label className="block text-xs text-gray-400 mb-1 ml-1">
                Output Dimension
              </label>
              <div className="flex bg-background rounded-lg border border-gray-700 p-1">
                {DIMENSIONS.map((d) => (
                  <button
                    key={d}
                    onClick={() => setSelectedDimension(d)}
                    className={`flex-1 rounded py-1.5 text-xs font-bold transition-all ${
                      selectedDimension === d
                        ? "bg-accent text-white shadow"
                        : "text-gray-400 hover:text-white"
                    }`}
                  >
                    {d}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="pt-2 flex justify-end">
            <button
              onClick={handleGenerate}
              disabled={loading}
              className={`px-8 py-3 rounded-lg font-bold text-white transition-all w-full md:w-auto shadow-lg ${
                loading
                  ? "bg-gray-600 cursor-not-allowed"
                  : "bg-primary hover:bg-blue-600 shadow-blue-500/20 active:scale-95"
              }`}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg
                    className="animate-spin h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Running SFDiff...
                </span>
              ) : (
                "Run Forecast Simulation"
              )}
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-500/20 border border-red-500/50 text-red-200 p-4 rounded-lg mb-8 text-center animate-pulse">
            {error}
          </div>
        )}

        {scenarioData && !loading && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in">
            {/* Left Column: Visualization */}
            <div className="lg:col-span-2 space-y-6">
              <TrackVisualizer trackData={scenarioData.track} />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Only show chart if relevant for dimension */}
                <div className="transition-all">
                  {selectedDimension !== "2D" ? (
                    <TimeSeriesChart
                      trackData={scenarioData.track}
                      dataKey="windSpeed"
                      title="Wind Speed Intensity"
                      unit="kt"
                      color="#f43f5e"
                    />
                  ) : (
                    <div className="opacity-50 grayscale">
                      <p className="text-xs text-center text-yellow-500 mt-1">
                        Not available in 2D Mode
                      </p>
                    </div>
                  )}
                </div>

                <div className="transition-all">
                  {selectedDimension === "4D" ? (
                    <TimeSeriesChart
                      trackData={scenarioData.track}
                      dataKey="pressure"
                      title="Central Pressure"
                      unit="mb"
                      color="#3b82f6"
                    />
                  ) : (
                    <div className="opacity-50 grayscale">
                      <p className="text-xs text-center text-yellow-500 mt-1">
                        Available in 4D Mode only
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right Column: Details & Metrics */}
            <div className="space-y-6">
              <div className="bg-surface/50 backdrop-blur-md border border-white/10 rounded-xl p-6 shadow-xl">
                <div className="flex justify-between items-start">
                  <div>
                    <h2 className="text-xl font-bold mb-1">
                      {scenarioData.name}
                    </h2>
                    <p className="text-gray-400 text-sm">
                      {DATASETS.find((d) => d.name === scenarioData.name)
                        ?.desc || scenarioData.description}
                    </p>
                  </div>
                  <span className="px-3 py-1 rounded bg-accent/20 text-accent text-xs font-mono border border-accent/20">
                    {selectedVariant}
                  </span>
                </div>

                <div className="mt-4 pt-4 border-t border-white/5 grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-xs text-gray-500 block">
                      Output Mode
                    </span>
                    <span className="text-sm font-mono text-gray-200">
                      {selectedDimension}
                    </span>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500 block">
                      Resolution
                    </span>
                    <span className="text-sm font-mono text-gray-200">
                      6 Hr / 0.5°
                    </span>
                  </div>
                </div>
              </div>

              <MetricsCard
                metrics={scenarioData.metrics}
                activeModels={activeModels}
                onAddModel={addModel}
              />

              <div className="bg-surface/50 backdrop-blur-md border border-white/10 rounded-xl p-6 shadow-xl">
                <h3 className="text-md font-bold mb-4 text-gray-200">
                  Execution Log
                </h3>
                <div className="space-y-2 text-xs font-mono text-gray-400">
                  <div className="flex justify-between">
                    <span>Core Model:</span>
                    <span className="text-gray-200">
                      SFDiff_{selectedVariant}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Dimension:</span>
                    <span className="text-gray-200">{selectedDimension}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Active Baselines:</span>
                    <span className="text-gray-200">
                      {activeModels.map((m) => m.toUpperCase()).join(", ")}
                    </span>
                  </div>
                  <div className="flex justify-between border-t border-white/5 pt-2 mt-2">
                    <span>Latency:</span>
                    <span className="text-green-400">142ms</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;