import React, { useState } from 'react';
import { ModelMetrics, AVAILABLE_MODELS, AVAILABLE_METRICS } from '../types';

interface MetricsCardProps {
  metrics: ModelMetrics;
  activeModels: string[];
  onAddModel: (modelId: string) => void;
}

const MetricsCard: React.FC<MetricsCardProps> = ({ metrics, activeModels, onAddModel }) => {
  const [displayedMetrics, setDisplayedMetrics] = useState<string[]>(['mse', 'crps', 'ade', 'fde']);
  const [isAddingMetric, setIsAddingMetric] = useState(false);
  const [isAddingModel, setIsAddingModel] = useState(false);

  // Get list of models that have data
  const availableDataModels = Object.keys(metrics);
  
  // Helper to check if a model is "active" (displayed)
  // We display all models passed in metrics props generally, but let's assume `activeModels` controls simulation,
  // while `metrics` contains results. We show what is in `metrics`.
  
  const handleAddMetric = (metricId: string) => {
    if (!displayedMetrics.includes(metricId)) {
        setDisplayedMetrics([...displayedMetrics, metricId]);
    }
    setIsAddingMetric(false);
  };

  const unusedMetrics = AVAILABLE_METRICS.filter(m => !displayedMetrics.includes(m.id));
  const unusedModels = AVAILABLE_MODELS.filter(m => !activeModels.includes(m.id));

  return (
    <div className="bg-surface/50 backdrop-blur-md border border-white/10 rounded-xl p-6 shadow-xl relative">
      <h3 className="text-xl font-bold text-gray-100 mb-6 flex items-center gap-2">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        Model Comparison
      </h3>

      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left text-gray-400">
            <thead className="text-xs text-gray-200 uppercase bg-white/5">
                <tr>
                    <th className="px-4 py-3 rounded-tl-lg">Model</th>
                    {displayedMetrics.map(mId => {
                        const m = AVAILABLE_METRICS.find(am => am.id === mId);
                        return <th key={mId} className="px-4 py-3">{m?.name || mId}</th>
                    })}
                    <th className="px-4 py-3 rounded-tr-lg w-10 text-center">
                        <button 
                            onClick={() => setIsAddingMetric(!isAddingMetric)}
                            className="text-gray-400 hover:text-white transition-colors"
                            title="Add Metric"
                        >
                            +
                        </button>
                        {isAddingMetric && (
                            <div className="absolute right-0 mt-2 w-40 bg-surface border border-white/10 rounded shadow-xl z-50">
                                {unusedMetrics.length === 0 ? (
                                    <div className="px-4 py-2 text-xs">No more metrics</div>
                                ) : (
                                    unusedMetrics.map(m => (
                                        <button 
                                            key={m.id}
                                            onClick={() => handleAddMetric(m.id)}
                                            className="block w-full text-left px-4 py-2 hover:bg-white/5 text-xs text-gray-300"
                                        >
                                            {m.name}
                                        </button>
                                    ))
                                )}
                            </div>
                        )}
                    </th>
                </tr>
            </thead>
            <tbody>
                {/* Always show SFDiff first or last? Let's sort alphabetically or specific order */}
                {Object.keys(metrics).map(modelKey => {
                     // Check if SFDiff
                     const isSFDiff = modelKey === 'sfdiff';
                     const modelDef = AVAILABLE_MODELS.find(m => m.id === modelKey);
                     const displayName = isSFDiff ? 'SFDiff (Ours)' : (modelDef?.name || modelKey.toUpperCase());
                     const rowColorClass = isSFDiff ? 'bg-white/10 border-b border-white/5' : 'bg-white/5 border-b border-white/5';
                     const textColorClass = isSFDiff ? 'text-accent font-bold' : (modelDef ? `text-[${modelDef.color}] font-bold` : 'text-gray-300 font-bold');
                     
                     // Helper for dynamic colors via style since Tailwind class interpolation is tricky
                     const nameStyle = isSFDiff ? { color: '#f43f5e' } : { color: modelDef?.color || '#cbd5e1' };

                     return (
                        <tr key={modelKey} className={`${rowColorClass} hover:bg-white/20 transition-colors`}>
                            <td className="px-4 py-4" style={nameStyle}>
                                {displayName}
                                {isSFDiff && <span className="ml-2 px-2 py-0.5 rounded-full bg-accent/20 text-accent text-xs font-normal">Best</span>}
                            </td>
                            {displayedMetrics.map(mId => (
                                <td key={mId} className="px-4 py-4 font-mono text-white">
                                    {metrics[modelKey][mId] !== undefined ? metrics[modelKey][mId].toFixed(4) : '-'}
                                </td>
                            ))}
                            <td className="px-4 py-4"></td>
                        </tr>
                     );
                })}
                {/* Add Model Row */}
                <tr>
                    <td colSpan={displayedMetrics.length + 2} className="px-4 py-3 text-center border-t border-white/5">
                         <div className="relative inline-block">
                            <button 
                                onClick={() => setIsAddingModel(!isAddingModel)}
                                className="flex items-center gap-2 text-primary hover:text-blue-400 transition-colors text-xs font-bold uppercase tracking-wider mx-auto"
                            >
                                <span className="text-lg">+</span> Add Model to Comparison
                            </button>
                            {isAddingModel && (
                                <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-48 bg-surface border border-white/10 rounded shadow-xl z-50">
                                    {unusedModels.length === 0 ? (
                                        <div className="px-4 py-2 text-xs">All models added</div>
                                    ) : (
                                        unusedModels.map(m => (
                                            <button 
                                                key={m.id}
                                                onClick={() => {
                                                    onAddModel(m.id);
                                                    setIsAddingModel(false);
                                                }}
                                                className="block w-full text-left px-4 py-2 hover:bg-white/5 text-xs text-gray-300"
                                            >
                                                {m.name}
                                            </button>
                                        ))
                                    )}
                                </div>
                            )}
                         </div>
                    </td>
                </tr>
            </tbody>
        </table>
      </div>
      
      <p className="text-xs text-gray-500 mt-4 italic">
        * Comparison generated using current simulation context.
      </p>
    </div>
  );
};

export default MetricsCard;