/**
 * Types for ColdStore Diagnostics
 */

export type StorageClassId = 0 | 1 | 2 | 3 | 4;

export interface StorageClassInfo {
  id: StorageClassId;
  name: string;
  shortName: string;
  color: string;
  bgLight: string;
  borderCol: string;
  description: string;
  operatorAction: string;
  rootCauseDetails: string;
}

export const STORAGE_CLASSES: Record<StorageClassId, StorageClassInfo> = {
  0: {
    id: 0,
    name: 'Normal (No Excursion)',
    shortName: 'Normal',
    color: '#059669', // Emerald
    bgLight: '#ecfdf5',
    borderCol: '#10b981',
    description: 'Refrigeration unit operating within standard hysteresis cycle (±0.3 °C) around setpoint.',
    operatorAction: 'No intervention required. Potato tuber respiration and dormancy maintained.',
    rootCauseDetails: 'Compressor duty cycle and fan circulation are operating normally. Temperature remains safely below threshold.',
  },
  1: {
    id: 1,
    name: 'Door Left Open',
    shortName: 'Door Open',
    color: '#d97706', // Amber
    bgLight: '#fffbeb',
    borderCol: '#f59e0b',
    description: 'Rapid ingress of ambient air caused by loading bay doors open or worn strip curtains.',
    operatorAction: 'Inspect bay doors, verify automatic closer mechanisms, and audit forklift operating protocol.',
    rootCauseDetails: 'Sharp temperature rise (1.5–4.0 °C) within 5–20 minutes, followed by fast pull-down once doors closed.',
  },
  2: {
    id: 2,
    name: 'Compressor / Cooling Failure',
    shortName: 'Cooling Failure',
    color: '#dc2626', // Red
    bgLight: '#fef2f2',
    borderCol: '#ef4444',
    description: 'Loss of cooling capacity due to mechanical compressor trip, refrigerant loss, or coil freeze.',
    operatorAction: 'Urgent HVAC dispatch: check compressor contactors, refrigerant sight glass, and defrost cycle.',
    rootCauseDetails: 'Monotonic, steady temperature rise (0.3–1.2 °C/h) with zero refrigeration pull-down recovery.',
  },
  3: {
    id: 3,
    name: 'Ambient Temperature Change',
    shortName: 'Ambient Shift',
    color: '#2563eb', // Blue
    bgLight: '#eff6ff',
    borderCol: '#3b82f6',
    description: 'Exterior diurnal heat gain leaking through storage envelope, louvers, or ventilation intake.',
    operatorAction: 'Audit building thermal envelope insulation, test louver sealing, and utilize night-air cooling flush.',
    rootCauseDetails: 'Gradual sinusoidal heat drift (0.1–0.5 °C/h) with smooth curvature and delayed, weak recovery.',
  },
  4: {
    id: 4,
    name: 'Sensor Malfunction',
    shortName: 'Sensor Error',
    color: '#7c3aed', // Purple
    bgLight: '#f5f3ff',
    borderCol: '#8b5cf6',
    description: 'Erratic RTD/thermistor wiring, analog telemetry dropouts, ADC pin float, or calibration drift.',
    operatorAction: 'Inspect sensor probe housing for condensation/corrosion, test resistance, or swap PT100 probe.',
    rootCauseDetails: 'Physically impossible transients (>1 °C/min instantaneous delta), zero-variance flatlines, or telemetry dropouts.',
  },
};

export interface DataPoint {
  minute: number; // 0 to 359
  timeLabel: string; // e.g. "00:00", "02:15"
  rawTemp: number;
  smoothedTemp: number;
  isExcursion: boolean;
}

export interface ExtractedFeatures {
  mean_temp: number;
  std_temp: number;
  mean_rate_rise: number;
  duration_above_thresh: number;
  max_slope: number;
  recovery_time: number;
  peak_temp: number;
  num_spikes: number;
  flatline_fraction: number;
}

export const FEATURE_NAMES: (keyof ExtractedFeatures)[] = [
  'mean_temp',
  'std_temp',
  'mean_rate_rise',
  'duration_above_thresh',
  'max_slope',
  'recovery_time',
  'peak_temp',
  'num_spikes',
  'flatline_fraction',
];

export const FEATURE_LABELS: Record<keyof ExtractedFeatures, { label: string; unit: string; description: string }> = {
  mean_temp: { label: 'Mean Temperature', unit: '°C', description: 'Average temperature across the 6-hour window' },
  std_temp: { label: 'Temperature Std Dev', unit: '°C', description: 'Variability / noise magnitude in temperature' },
  mean_rate_rise: { label: 'Mean Rate of Rise', unit: '°C/min', description: 'Average positive rate of temperature climb' },
  duration_above_thresh: { label: 'Excursion Duration', unit: 'min', description: 'Total minutes spent exceeding excursion threshold' },
  max_slope: { label: 'Max Rate of Rise', unit: '°C/min', description: 'Maximum positive temperature change per minute' },
  recovery_time: { label: 'Recovery Time', unit: 'min', description: 'Time taken to fall back below threshold (or 360 if none)' },
  peak_temp: { label: 'Peak Temperature', unit: '°C', description: 'Maximum recorded temperature in the window' },
  num_spikes: { label: 'Spike Count', unit: 'spikes', description: 'Count of rapid jumps exceeding 1.0 °C in one minute' },
  flatline_fraction: { label: 'Flatline Fraction', unit: 'ratio', description: 'Fraction of consecutive unchanged readings (sensor lock)' },
};

export interface StorageWindowSample {
  id: string;
  classId: StorageClassId;
  points: number[]; // 360 raw temperature values
  features: ExtractedFeatures;
}

export interface StorageSettings {
  setpoint: number; // default 5.0 °C
  threshold: number; // default 7.0 °C
  medianWindow: number; // default 5
  maWindow: number; // default 7
  datasetSize: number; // default 2000
}

export interface ModelHyperparameters {
  numTrees: number; // default 100
  maxDepth: number; // default 4
  learningRate: number; // default 0.1
  lambda: number; // default 1.0 (L2)
  minChildWeight: number; // default 1.0
  subsample: number; // default 0.85
  colsample: number; // default 0.85
}

export interface PredictionResult {
  classId: StorageClassId;
  className: string;
  confidence: number; // 0 to 1
  probabilities: Record<StorageClassId, number>;
  features: ExtractedFeatures;
  isLowConfidence: boolean;
}

export interface EvaluationMetrics {
  accuracy: number;
  macroF1: number;
  classMetrics: Record<
    StorageClassId,
    {
      precision: number;
      recall: number;
      f1: number;
      support: number;
    }
  >;
  confusionMatrix: number[][]; // 5x5: [actual][predicted]
  featureImportance: Record<keyof ExtractedFeatures, number>; // Normalized gain 0..100
  testLoss: number;
  sampleCount: {
    train: number;
    test: number;
  };
}
