export interface Coordinate {
  lat: number;
  lon: number;
}

export interface TimeStepData {
  time: number; // Hours from start
  lat: number;
  lon: number;
  windSpeed: number | null; // knots, null if model doesn't support
  pressure: number | null; // mb, null if model doesn't support
}

export type TrackData = Record<string, TimeStepData[]>;

// Dynamic metrics dictionary: { "ModelName": { "MetricName": value } }
export type ModelMetrics = Record<string, Record<string, number>>;

export interface ScenarioData {
  id: string;
  name: string;
  description: string;
  track: TrackData;
  metrics: ModelMetrics;
  generatedAt: string;
}

export type SFDiffDimension = '2D' | '3D' | '4D';
export type SFDiffVariant = 'S4' | 'S5' | 'ResNet' | 'Featured';

export interface GenerateScenarioRequest {
  sim_config: {
    sfdiffVariant: SFDiffVariant;
    sfdiffDimension: SFDiffDimension;
    activeModels: string[];
  };
  model_id: string;
  dataset_id: string;
}

export const AVAILABLE_MODELS = [
  { id: 'ar', name: 'AR (Baseline)', color: '#a855f7' },
  { id: 'kf', name: 'KF (Filter)', color: '#f59e0b' },
  { id: 'hwrf', name: 'HWRF (Physics)', color: '#3b82f6' },
  { id: 'gfs', name: 'GFS (Global)', color: '#06b6d4' },
  { id: 'cliper', name: 'CLIPER', color: '#64748b' }
];

export const AVAILABLE_METRICS = [
  { id: 'mse', name: 'MSE' },
  { id: 'crps', name: 'CRPS' },
  { id: 'ade', name: 'ADE (km)' },
  { id: 'fde', name: 'FDE (km)' },
  { id: 'cte', name: 'Cross Track Err' },
  { id: 'ate', name: 'Along Track Err' }
];
