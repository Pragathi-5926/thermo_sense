import React, { useState } from 'react';
import {
  StorageClassId,
  STORAGE_CLASSES,
  StorageSettings,
  ModelHyperparameters,
  StorageWindowSample,
  FEATURE_NAMES,
  FEATURE_LABELS,
} from '../types/storage';
import { exportFeaturesToCSV } from '../utils/syntheticData';
import { TrainingProgress } from '../ml/trainerClient';
import { DEFAULT_HYPERPARAMETERS } from '../ml/storageCache';
import {
  Database,
  Cpu,
  Download,
  Play,
  RotateCcw,
  Layers,
  Sparkles,
  BarChart,
  CheckCircle,
  ExternalLink,
} from 'lucide-react';

interface DataModelTabProps {
  settings: StorageSettings;
  samples: StorageWindowSample[];
  hyperparameters: ModelHyperparameters;
  onUpdateHyperparameters: (params: ModelHyperparameters) => void;
  onGenerateDataset: (count: number) => void;
  onTrainModel: () => void;
  isTraining: boolean;
  trainingProgress?: TrainingProgress;
  isGenerating: boolean;
  onSelectSampleForDashboard?: (sample: StorageWindowSample) => void;
}

export const DataModelTab: React.FC<DataModelTabProps> = ({
  settings,
  samples,
  hyperparameters,
  onUpdateHyperparameters,
  onGenerateDataset,
  onTrainModel,
  isTraining,
  trainingProgress,
  isGenerating,
  onSelectSampleForDashboard,
}) => {
  const [datasetSizeInput, setDatasetSizeInput] = useState<number>(samples.length || 2000);
  const [filterClass, setFilterClass] = useState<StorageClassId | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Handle CSV export
  const handleExportCSV = () => {
    if (samples.length === 0) return;
    const csvContent = exportFeaturesToCSV(samples);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `coldstore_features_${samples.length}_samples.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Class counts
  const classCounts = React.useMemo(() => {
    const counts: Record<StorageClassId, number> = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    for (const s of samples) {
      counts[s.classId] = (counts[s.classId] || 0) + 1;
    }
    return counts;
  }, [samples]);

  // Filtered samples for preview table (first 25 matches)
  const filteredSamples = React.useMemo(() => {
    return samples
      .filter((s) => {
        if (filterClass !== 'all' && s.classId !== filterClass) return false;
        if (searchQuery && !s.id.toLowerCase().includes(searchQuery.toLowerCase())) return false;
        return true;
      })
      .slice(0, 30);
  }, [samples, filterClass, searchQuery]);

  return (
    <div className="space-y-6 pb-12">
      {/* Top Grid: Synthetic Generator on Left, GBDT Hyperparameters on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Synthetic Dataset Card (5 cols) */}
        <div className="lg:col-span-5 bg-white rounded-xl p-5 border border-slate-200 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Database className="w-5 h-5 text-emerald-600" />
              <h3 className="font-bold text-slate-900 text-base">Synthetic Telemetry Generator</h3>
            </div>
            <p className="text-xs text-slate-500 mb-4 leading-relaxed">
              Synthesizes realistic 6-hour windows (360 min) with Gaussian noise (&sigma; &approx; 0.05 - 0.15 °C),
              compressor duty cycles, door thermal transients, refrigeration failures, and unphysical sensor faults.
            </p>

            {/* Dataset Size Input */}
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-slate-700">Dataset Size</label>
                <div className="flex items-center gap-2">
                  {[1000, 2000, 3000].map((count) => (
                    <button
                      key={count}
                      type="button"
                      onClick={() => setDatasetSizeInput(count)}
                      className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                        datasetSizeInput === count
                          ? 'bg-emerald-600 text-white'
                          : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-100'
                      }`}
                    >
                      {count}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min="250"
                  max="5000"
                  step="250"
                  value={datasetSizeInput}
                  onChange={(e) => setDatasetSizeInput(parseInt(e.target.value, 10) || 2000)}
                  className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white text-slate-800 font-mono"
                />
                <span className="text-xs text-slate-500 shrink-0">samples</span>
              </div>
            </div>

            {/* Class Balance Summary */}
            <div className="mt-4">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Class Distribution ({samples.length} total)
              </span>
              <div className="grid grid-cols-5 gap-1.5 mt-2">
                {([0, 1, 2, 3, 4] as StorageClassId[]).map((cId) => {
                  const info = STORAGE_CLASSES[cId];
                  const count = classCounts[cId] || 0;
                  return (
                    <div
                      key={cId}
                      className="p-2 rounded-lg border text-center text-xs"
                      style={{
                        backgroundColor: info.bgLight,
                        borderColor: info.borderCol,
                      }}
                    >
                      <div className="text-[10px] font-bold truncate" style={{ color: info.color }}>
                        {info.shortName}
                      </div>
                      <div className="font-mono font-bold text-slate-800 mt-0.5">{count}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between gap-3">
            <button
              onClick={() => onGenerateDataset(datasetSizeInput)}
              disabled={isGenerating || isTraining}
              className="flex-1 flex items-center justify-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 rounded-lg shadow-xs transition"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>{isGenerating ? 'Generating...' : 'Generate Dataset'}</span>
            </button>

            <button
              onClick={handleExportCSV}
              disabled={samples.length === 0}
              className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg shadow-xs transition disabled:opacity-50"
              title="Download feature dataset as CSV"
            >
              <Download className="w-3.5 h-3.5 text-slate-500" />
              <span>Export CSV</span>
            </button>
          </div>
        </div>

        {/* XGBoost Model Training Panel (7 cols) */}
        <div className="lg:col-span-7 bg-white rounded-xl p-5 border border-slate-200 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Cpu className="w-5 h-5 text-blue-600" />
                <h3 className="font-bold text-slate-900 text-base">
                  XGBoost-Style Classifier Hyperparameters
                </h3>
              </div>
              <button
                onClick={() => onUpdateHyperparameters(DEFAULT_HYPERPARAMETERS)}
                className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-800 px-2 py-1 rounded hover:bg-slate-100 transition"
                title="Reset to defaults"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Defaults</span>
              </button>
            </div>
            <p className="text-xs text-slate-500 mb-4 leading-relaxed">
              Custom gradient boosted decision trees with second-order gradient + hessian weights, multiclass softmax
              loss, and Web Worker execution. Stratified 80/20 train/test evaluation.
            </p>

            {/* Hyperparameter Inputs */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
              <div>
                <label className="block font-medium text-slate-700 mb-1">
                  Num Trees (Rounds)
                </label>
                <input
                  type="number"
                  min="10"
                  max="300"
                  step="10"
                  value={hyperparameters.numTrees}
                  onChange={(e) =>
                    onUpdateHyperparameters({
                      ...hyperparameters,
                      numTrees: parseInt(e.target.value, 10) || 100,
                    })
                  }
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded-lg font-mono text-slate-800"
                />
                <span className="text-[10px] text-slate-400 mt-0.5 block">Default: 100</span>
              </div>

              <div>
                <label className="block font-medium text-slate-700 mb-1">Max Depth</label>
                <input
                  type="number"
                  min="2"
                  max="8"
                  value={hyperparameters.maxDepth}
                  onChange={(e) =>
                    onUpdateHyperparameters({
                      ...hyperparameters,
                      maxDepth: parseInt(e.target.value, 10) || 4,
                    })
                  }
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded-lg font-mono text-slate-800"
                />
                <span className="text-[10px] text-slate-400 mt-0.5 block">Default: 4</span>
              </div>

              <div>
                <label className="block font-medium text-slate-700 mb-1">
                  Learning Rate (&eta;)
                </label>
                <input
                  type="number"
                  step="0.02"
                  min="0.01"
                  max="0.5"
                  value={hyperparameters.learningRate}
                  onChange={(e) =>
                    onUpdateHyperparameters({
                      ...hyperparameters,
                      learningRate: parseFloat(e.target.value) || 0.1,
                    })
                  }
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded-lg font-mono text-slate-800"
                />
                <span className="text-[10px] text-slate-400 mt-0.5 block">Default: 0.10</span>
              </div>

              <div>
                <label className="block font-medium text-slate-700 mb-1">
                  L2 Regularization (&lambda;)
                </label>
                <input
                  type="number"
                  step="0.2"
                  min="0.1"
                  max="10"
                  value={hyperparameters.lambda}
                  onChange={(e) =>
                    onUpdateHyperparameters({
                      ...hyperparameters,
                      lambda: parseFloat(e.target.value) || 1.0,
                    })
                  }
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded-lg font-mono text-slate-800"
                />
                <span className="text-[10px] text-slate-400 mt-0.5 block">Default: 1.0</span>
              </div>

              <div>
                <label className="block font-medium text-slate-700 mb-1">Min Child Weight</label>
                <input
                  type="number"
                  step="0.5"
                  min="0.5"
                  max="10"
                  value={hyperparameters.minChildWeight}
                  onChange={(e) =>
                    onUpdateHyperparameters({
                      ...hyperparameters,
                      minChildWeight: parseFloat(e.target.value) || 1.0,
                    })
                  }
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded-lg font-mono text-slate-800"
                />
                <span className="text-[10px] text-slate-400 mt-0.5 block">Default: 1.0</span>
              </div>

              <div>
                <label className="block font-medium text-slate-700 mb-1">Row Subsample</label>
                <input
                  type="number"
                  step="0.05"
                  min="0.5"
                  max="1.0"
                  value={hyperparameters.subsample}
                  onChange={(e) =>
                    onUpdateHyperparameters({
                      ...hyperparameters,
                      subsample: parseFloat(e.target.value) || 0.85,
                    })
                  }
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded-lg font-mono text-slate-800"
                />
                <span className="text-[10px] text-slate-400 mt-0.5 block">Default: 0.85</span>
              </div>
            </div>

            {/* Live Progress Bar when Training */}
            {isTraining && trainingProgress && (
              <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-xl space-y-2 animate-in fade-in">
                <div className="flex justify-between items-center text-xs text-blue-900 font-semibold">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-blue-600 animate-ping" />
                    <span>
                      Training Tree Round {trainingProgress.round} / {trainingProgress.totalRounds}
                    </span>
                  </span>
                  <span>{trainingProgress.percentage}%</span>
                </div>
                <div className="w-full h-2 rounded-full bg-blue-200 overflow-hidden">
                  <div
                    className="h-full bg-blue-600 transition-all duration-150"
                    style={{ width: `${trainingProgress.percentage}%` }}
                  />
                </div>
                <div className="flex justify-between items-center text-[11px] text-blue-800 font-mono">
                  <span>Train Loss: {trainingProgress.trainLoss.toFixed(4)}</span>
                  {trainingProgress.valLoss !== undefined && (
                    <span>Val Loss: {trainingProgress.valLoss.toFixed(4)}</span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Train Button */}
          <div className="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between">
            <div className="text-xs text-slate-500">
              Stratified 80% Train ({Math.floor(samples.length * 0.8)}) / 20% Test ({Math.ceil(samples.length * 0.2)})
            </div>
            <button
              onClick={onTrainModel}
              disabled={isTraining || isGenerating || samples.length === 0}
              className="flex items-center gap-2 px-5 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded-lg shadow-xs transition"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{isTraining ? 'Training GBDT...' : 'Train XGBoost Model'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Dataset Feature Table Preview */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-600" />
            <h3 className="font-semibold text-slate-900 text-sm">
              Extracted Feature Matrix Preview ({samples.length} Samples)
            </h3>
          </div>

          {/* Filters */}
          <div className="flex items-center gap-2 text-xs">
            <select
              value={filterClass}
              onChange={(e) => setFilterClass(e.target.value as any)}
              className="px-2.5 py-1.5 border border-slate-300 rounded-lg bg-white text-slate-700"
            >
              <option value="all">All Classes</option>
              {([0, 1, 2, 3, 4] as StorageClassId[]).map((cId) => (
                <option key={cId} value={cId}>
                  {STORAGE_CLASSES[cId].name}
                </option>
              ))}
            </select>

            <input
              type="text"
              placeholder="Search sample ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="px-2.5 py-1.5 border border-slate-300 rounded-lg text-slate-800 placeholder-slate-400 text-xs"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-50 text-slate-600 uppercase tracking-wider font-sans border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3 font-semibold">Sample ID</th>
                <th className="py-2.5 px-3 font-semibold">Ground Truth Class</th>
                <th className="py-2.5 px-3 font-semibold">Mean (°C)</th>
                <th className="py-2.5 px-3 font-semibold">Std Dev</th>
                <th className="py-2.5 px-3 font-semibold">Max Slope</th>
                <th className="py-2.5 px-3 font-semibold">Duration (min)</th>
                <th className="py-2.5 px-3 font-semibold">Recovery (min)</th>
                <th className="py-2.5 px-3 font-semibold">Peak (°C)</th>
                <th className="py-2.5 px-3 font-semibold">Spikes</th>
                <th className="py-2.5 px-3 font-semibold">Flatline</th>
                <th className="py-2.5 px-3 font-semibold font-sans">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {filteredSamples.map((s) => {
                const info = STORAGE_CLASSES[s.classId];
                const f = s.features;
                return (
                  <tr key={s.id} className="hover:bg-slate-50/70 transition">
                    <td className="py-2 px-3 font-bold text-slate-900">{s.id}</td>
                    <td className="py-2 px-3 font-sans">
                      <span
                        className="text-[11px] font-semibold px-2 py-0.5 rounded-full border inline-block"
                        style={{
                          backgroundColor: info.bgLight,
                          borderColor: info.borderCol,
                          color: info.color,
                        }}
                      >
                        {info.shortName}
                      </span>
                    </td>
                    <td className="py-2 px-3">{f.mean_temp.toFixed(2)}</td>
                    <td className="py-2 px-3">{f.std_temp.toFixed(2)}</td>
                    <td className="py-2 px-3">{f.max_slope.toFixed(2)}</td>
                    <td className="py-2 px-3">{f.duration_above_thresh}</td>
                    <td className="py-2 px-3">{f.recovery_time}</td>
                    <td className="py-2 px-3">{f.peak_temp.toFixed(2)}</td>
                    <td className="py-2 px-3">{f.num_spikes}</td>
                    <td className="py-2 px-3">{(f.flatline_fraction * 100).toFixed(0)}%</td>
                    <td className="py-2 px-3 font-sans">
                      {onSelectSampleForDashboard && (
                        <button
                          onClick={() => onSelectSampleForDashboard(s)}
                          className="text-[11px] font-medium text-emerald-700 hover:text-emerald-800 flex items-center gap-1 hover:underline"
                          title="Load this window into Dashboard for live diagnosis"
                        >
                          <span>Diagnose</span>
                          <ExternalLink className="w-3 h-3" />
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {filteredSamples.length === 0 && (
          <div className="py-8 text-center text-xs text-slate-400">
            No matching samples found.
          </div>
        )}
      </div>
    </div>
  );
};
