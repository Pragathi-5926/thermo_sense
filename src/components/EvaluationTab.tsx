import React from 'react';
import {
  EvaluationMetrics,
  FEATURE_LABELS,
  FEATURE_NAMES,
  ModelHyperparameters,
  STORAGE_CLASSES,
  StorageClassId,
  StorageSettings,
} from '../types/storage';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  Award,
  CheckCircle2,
  Download,
  Flame,
  Grid,
  Layers,
  Sparkles,
  TrendingUp,
} from 'lucide-react';

interface EvaluationTabProps {
  metrics: EvaluationMetrics | null;
  hyperparameters: ModelHyperparameters;
  settings: StorageSettings;
}

export const EvaluationTab: React.FC<EvaluationTabProps> = ({
  metrics,
  hyperparameters,
  settings,
}) => {
  if (!metrics) {
    return (
      <div className="bg-white rounded-xl p-12 text-center border border-slate-200 shadow-xs max-w-xl mx-auto my-12">
        <Grid className="w-12 h-12 text-slate-300 mx-auto mb-3" />
        <h3 className="font-bold text-slate-800 text-base">Model Evaluation Pending</h3>
        <p className="text-xs text-slate-500 mt-1">
          Train the XGBoost model in the Data & Model tab to generate validation metrics, confusion
          matrix, and feature importance.
        </p>
      </div>
    );
  }

  // Feature importance data sorted descending
  const sortedFeatureImportance = React.useMemo(() => {
    return FEATURE_NAMES.map((name) => ({
      name,
      label: FEATURE_LABELS[name].label,
      gainScore: metrics.featureImportance[name] || 0,
      description: FEATURE_LABELS[name].description,
    })).sort((a, b) => b.gainScore - a.gainScore);
  }, [metrics.featureImportance]);

  // Export metrics JSON
  const handleDownloadJSON = () => {
    const report = {
      timestamp: new Date().toISOString(),
      model: 'ColdStore-XGBoost-Multiclass',
      settings,
      hyperparameters,
      overallMetrics: {
        accuracy: metrics.accuracy,
        macroF1: metrics.macroF1,
        testLoss: metrics.testLoss,
        testSampleCount: metrics.sampleCount.test,
        trainSampleCount: metrics.sampleCount.train,
      },
      classMetrics: metrics.classMetrics,
      confusionMatrix: metrics.confusionMatrix,
      featureImportanceGain: metrics.featureImportance,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `coldstore_model_report_${Date.now()}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const cm = metrics.confusionMatrix;

  return (
    <div className="space-y-6 pb-12">
      {/* KPI Cards Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Accuracy */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Overall Accuracy</span>
            <Award className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-3xl font-black text-slate-900">
            {(metrics.accuracy * 100).toFixed(1)}%
          </div>
          <div className="text-[11px] text-emerald-700 font-medium mt-1 flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Stratified 20% test holdout</span>
          </div>
        </div>

        {/* Macro F1 */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Macro F1-Score</span>
            <TrendingUp className="w-4 h-4 text-blue-600" />
          </div>
          <div className="text-3xl font-black text-slate-900">{metrics.macroF1.toFixed(3)}</div>
          <div className="text-[11px] text-slate-500 mt-1">Balanced across 5 classes</div>
        </div>

        {/* Test Loss */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Log Loss (CE)</span>
            <Flame className="w-4 h-4 text-amber-600" />
          </div>
          <div className="text-3xl font-black text-slate-900 font-mono">
            {metrics.testLoss.toFixed(3)}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Cross-entropy on test set</div>
        </div>

        {/* Holdout Samples */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Test Samples</span>
            <Layers className="w-4 h-4 text-purple-600" />
          </div>
          <div className="text-3xl font-black text-slate-900">
            {metrics.sampleCount.test.toLocaleString()}
          </div>
          <div className="flex items-center justify-between text-[11px] text-slate-500 mt-1">
            <span>Train: {metrics.sampleCount.train.toLocaleString()}</span>
            <button
              onClick={handleDownloadJSON}
              className="text-emerald-700 hover:text-emerald-800 font-semibold flex items-center gap-1 hover:underline"
              title="Download metrics as JSON"
            >
              <Download className="w-3 h-3" />
              <span>Report</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Row: Confusion Matrix Heatmap (7 cols) + Per-Class Metrics (5 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Confusion Matrix Heatmap: 7 cols */}
        <div className="lg:col-span-7 bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h3 className="font-bold text-slate-900 text-base">Confusion Matrix Heatmap</h3>
              <p className="text-xs text-slate-500">
                Ground Truth (Rows) vs. Model Prediction (Columns). Diagonal cells represent correct
                classifications.
              </p>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600">
              5 × 5 Matrix
            </span>
          </div>

          <div className="mt-4 overflow-x-auto">
            <div className="inline-block min-w-full">
              {/* Column Headers (Predicted) */}
              <div className="text-center text-[10px] uppercase font-bold tracking-wider text-slate-400 mb-1 ml-28">
                Predicted Root Cause Class →
              </div>

              <table className="w-full border-collapse text-xs">
                <thead>
                  <tr>
                    <th className="p-2 text-left font-semibold text-slate-500 uppercase text-[10px] tracking-wider w-28">
                      Ground Truth ↓
                    </th>
                    {([0, 1, 2, 3, 4] as StorageClassId[]).map((cId) => (
                      <th
                        key={cId}
                        className="p-2 text-center font-semibold text-slate-700 text-[11px] truncate max-w-[80px]"
                        title={STORAGE_CLASSES[cId].name}
                      >
                        {STORAGE_CLASSES[cId].shortName}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {([0, 1, 2, 3, 4] as StorageClassId[]).map((actualId) => {
                    const row = cm[actualId] || [0, 0, 0, 0, 0];
                    const rowSum = row.reduce((a, b) => a + b, 0) || 1;

                    return (
                      <tr key={actualId} className="border-t border-slate-100">
                        <td className="p-2 font-medium text-slate-900 text-[11px] truncate max-w-[120px]">
                          <div className="flex items-center gap-1.5">
                            <span
                              className="w-2 h-2 rounded-full shrink-0"
                              style={{ backgroundColor: STORAGE_CLASSES[actualId].color }}
                            />
                            <span title={STORAGE_CLASSES[actualId].name}>
                              {STORAGE_CLASSES[actualId].shortName}
                            </span>
                          </div>
                        </td>

                        {([0, 1, 2, 3, 4] as StorageClassId[]).map((predId) => {
                          const count = row[predId] ?? 0;
                          const rate = count / rowSum;
                          const isDiagonal = actualId === predId;

                          // Heatmap color logic
                          let bg = '#f8fafc';
                          let text = '#64748b';

                          if (isDiagonal) {
                            if (rate >= 0.9) {
                              bg = '#ecfdf5'; // emerald-50
                              text = '#047857'; // emerald-700
                            } else if (rate >= 0.7) {
                              bg = '#d1fae5'; // emerald-100
                              text = '#065f46';
                            } else {
                              bg = '#a7f3d0';
                              text = '#064e3b';
                            }
                          } else if (count > 0) {
                            bg = '#fff1f2'; // red-50
                            text = '#be123c'; // red-700
                          }

                          return (
                            <td
                              key={predId}
                              className="p-2 text-center transition-colors"
                              style={{ backgroundColor: bg }}
                              title={`Actual: ${STORAGE_CLASSES[actualId].name}\nPredicted: ${
                                STORAGE_CLASSES[predId].name
                              }\nCount: ${count} (${(rate * 100).toFixed(1)}%)`}
                            >
                              <div className="font-mono font-bold text-xs" style={{ color: text }}>
                                {count}
                              </div>
                              <div className="text-[10px] text-slate-400 font-mono">
                                {(rate * 100).toFixed(0)}%
                              </div>
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Per-Class Metrics Table: 5 cols */}
        <div className="lg:col-span-5 bg-white rounded-xl p-5 border border-slate-200 shadow-xs flex flex-col justify-between">
          <div>
            <h3 className="font-bold text-slate-900 text-base mb-1">Per-Class Diagnostic Performance</h3>
            <p className="text-xs text-slate-500 mb-4">
              Precision, Recall, and F1 harmonic score per failure mode.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-slate-50 text-slate-600 uppercase tracking-wider font-sans border-b border-slate-200">
                  <tr>
                    <th className="py-2 px-2.5 font-semibold">Class</th>
                    <th className="py-2 px-2.5 font-semibold text-right">Prec.</th>
                    <th className="py-2 px-2.5 font-semibold text-right">Recall</th>
                    <th className="py-2 px-2.5 font-semibold text-right">F1</th>
                    <th className="py-2 px-2.5 font-semibold text-right">Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  {([0, 1, 2, 3, 4] as StorageClassId[]).map((cId) => {
                    const info = STORAGE_CLASSES[cId];
                    const m = metrics.classMetrics[cId] || {
                      precision: 0,
                      recall: 0,
                      f1: 0,
                      support: 0,
                    };

                    return (
                      <tr key={cId} className="hover:bg-slate-50 transition">
                        <td className="py-2.5 px-2.5 font-sans font-medium text-slate-900">
                          <div className="flex items-center gap-1.5">
                            <span
                              className="w-2 h-2 rounded-full shrink-0"
                              style={{ backgroundColor: info.color }}
                            />
                            <span>{info.shortName}</span>
                          </div>
                        </td>
                        <td className="py-2.5 px-2.5 text-right font-bold text-slate-800">
                          {m.precision.toFixed(3)}
                        </td>
                        <td className="py-2.5 px-2.5 text-right font-bold text-slate-800">
                          {m.recall.toFixed(3)}
                        </td>
                        <td className="py-2.5 px-2.5 text-right font-bold text-emerald-700">
                          {m.f1.toFixed(3)}
                        </td>
                        <td className="py-2.5 px-2.5 text-right text-slate-500">{m.support}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Macro Average F1:</span>
            <span className="font-bold font-mono text-emerald-700 text-sm">
              {metrics.macroF1.toFixed(3)}
            </span>
          </div>
        </div>
      </div>

      {/* Feature Importance Bar Chart (Gain-Based) */}
      <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="font-bold text-slate-900 text-base">
              Feature Importance (Gain-Based)
            </h3>
            <p className="text-xs text-slate-500">
              Aggregated second-order split gain across all boosting trees, normalized to 100%.
            </p>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            Metric: Split Gain &Delta;L
          </span>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={sortedFeatureImportance}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
              <XAxis
                type="number"
                domain={[0, 100]}
                unit="%"
                stroke="#94a3b8"
                tick={{ fontSize: 11 }}
              />
              <YAxis
                type="category"
                dataKey="label"
                stroke="#64748b"
                tick={{ fontSize: 11 }}
                width={120}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="bg-slate-900 text-white p-2.5 rounded-lg text-xs shadow-lg max-w-xs">
                        <div className="font-bold text-slate-200">{d.label}</div>
                        <div className="text-emerald-400 font-mono font-bold mt-0.5">
                          Gain Share: {d.gainScore.toFixed(2)}%
                        </div>
                        <div className="text-slate-400 text-[11px] mt-1">{d.description}</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="gainScore" fill="#0d9488" radius={[0, 4, 4, 0]}>
                {sortedFeatureImportance.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={index === 0 ? '#059669' : index < 3 ? '#0d9488' : '#38bdf8'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Feature definitions summary */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4 pt-4 border-t border-slate-100 text-xs text-slate-600">
          <div>
            <strong className="text-slate-800">Spikes & Flatline:</strong> Directly isolate sensor
            ADC freezes and electrical glitches from physical refrigeration heat transfer.
          </div>
          <div>
            <strong className="text-slate-800">Max Slope & Recovery:</strong> Distinguish rapid door
            openings (fast rise + fast pull-down) from ambient diurnal ramps.
          </div>
          <div>
            <strong className="text-slate-800">Duration & Peak:</strong> Differentiate non-recovering
            compressor failures from temporary excursions.
          </div>
        </div>
      </div>
    </div>
  );
};
