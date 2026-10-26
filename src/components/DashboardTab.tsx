import React, { useState, useMemo, useRef } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  Legend,
} from 'recharts';
import {
  StorageClassId,
  STORAGE_CLASSES,
  StorageSettings,
  StorageWindowSample,
  FEATURE_LABELS,
  FEATURE_NAMES,
  PredictionResult,
  ExtractedFeatures,
} from '../types/storage';
import { generateWindowForClass } from '../utils/syntheticData';
import { smoothTemperature, extractFeatures } from '../utils/signalProcessing';
import { XGBoostClassifier } from '../ml/xgboost';
import {
  AlertTriangle,
  CheckCircle2,
  Upload,
  RefreshCw,
  Shuffle,
  Info,
  ShieldAlert,
  ArrowRight,
  Eye,
  EyeOff,
  SlidersHorizontal,
} from 'lucide-react';

interface DashboardTabProps {
  settings: StorageSettings;
  model: XGBoostClassifier | null;
  onOpenSettings: () => void;
  selectedSample?: StorageWindowSample | null;
}

export const DashboardTab: React.FC<DashboardTabProps> = ({
  settings,
  model,
  onOpenSettings,
  selectedSample,
}) => {
  // Current active 360-minute window
  const [activeRawData, setActiveRawData] = useState<number[]>(() =>
    selectedSample ? selectedSample.points : generateWindowForClass(1, settings)
  );
  const [activeClassId, setActiveClassId] = useState<StorageClassId | 'custom'>(
    selectedSample ? selectedSample.classId : 1
  );
  const [showRaw, setShowRaw] = useState(true);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Sync when selectedSample prop changes
  React.useEffect(() => {
    if (selectedSample) {
      setActiveRawData(selectedSample.points);
      setActiveClassId(selectedSample.classId);
    }
  }, [selectedSample]);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Compute smoothed and features dynamically whenever raw data or settings change
  const { smoothedData, features, chartData, excursionIntervals } = useMemo(() => {
    const smoothed = smoothTemperature(
      activeRawData,
      settings.medianWindow,
      settings.maWindow
    );
    const feat = extractFeatures(activeRawData, smoothed, settings.threshold);

    // Build chart data
    const points = activeRawData.map((raw, idx) => {
      const h = Math.floor(idx / 60);
      const m = idx % 60;
      const timeLabel = `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
      const sTemp = smoothed[idx];
      const isExcursion = sTemp > settings.threshold;

      return {
        minute: idx,
        timeLabel,
        rawTemp: raw,
        smoothedTemp: Number(sTemp.toFixed(2)),
        isExcursion,
      };
    });

    // Detect excursion intervals for shaded background
    const intervals: { start: number; end: number }[] = [];
    let curStart: number | null = null;

    for (let i = 0; i < smoothed.length; i++) {
      if (smoothed[i] > settings.threshold) {
        if (curStart === null) curStart = i;
      } else {
        if (curStart !== null) {
          intervals.push({ start: curStart, end: i - 1 });
          curStart = null;
        }
      }
    }
    if (curStart !== null) {
      intervals.push({ start: curStart, end: smoothed.length - 1 });
    }

    return {
      smoothedData: smoothed,
      features: feat,
      chartData: points,
      excursionIntervals: intervals,
    };
  }, [activeRawData, settings]);

  // Model prediction on the current window
  const prediction: PredictionResult | null = useMemo(() => {
    if (!model) return null;
    return model.predict(features);
  }, [model, features]);

  // Handler to generate a specific synthetic class scenario
  const handleSimulateClass = (classId: StorageClassId) => {
    const raw = generateWindowForClass(classId, settings);
    setActiveRawData(raw);
    setActiveClassId(classId);
    setUploadError(null);
  };

  // Random scenario
  const handleSimulateRandom = () => {
    const randomClass = Math.floor(Math.random() * 5) as StorageClassId;
    handleSimulateClass(randomClass);
  };

  // CSV file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        if (!text) throw new Error('File is empty.');

        const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
        if (lines.length < 10) {
          throw new Error('CSV must contain at least 10 temperature readings.');
        }

        const parsedValues: number[] = [];

        for (let i = 0; i < lines.length; i++) {
          const line = lines[i].trim();
          // Check if header line
          if (i === 0 && (line.toLowerCase().includes('temp') || line.toLowerCase().includes('time'))) {
            continue;
          }
          const parts = line.split(/[,\t;]/);
          // Look for temperature number in columns
          let val = NaN;
          if (parts.length >= 2) {
            val = parseFloat(parts[1]);
            if (isNaN(val)) val = parseFloat(parts[0]);
          } else {
            val = parseFloat(parts[0]);
          }

          if (!isNaN(val)) {
            parsedValues.push(val);
          }
        }

        if (parsedValues.length < 10) {
          throw new Error('Could not parse valid temperature numbers from CSV.');
        }

        // Resample / fit to 360 points
        let resampled: number[];
        if (parsedValues.length === 360) {
          resampled = parsedValues;
        } else {
          resampled = new Array<number>(360);
          for (let i = 0; i < 360; i++) {
            const srcIdx = (i / 359) * (parsedValues.length - 1);
            const lower = Math.floor(srcIdx);
            const upper = Math.ceil(srcIdx);
            const frac = srcIdx - lower;
            resampled[i] =
              lower === upper
                ? parsedValues[lower]
                : parsedValues[lower] * (1 - frac) + parsedValues[upper] * frac;
            resampled[i] = Number(resampled[i].toFixed(2));
          }
        }

        setActiveRawData(resampled);
        setActiveClassId('custom');
        setUploadError(null);
      } catch (err: any) {
        setUploadError(err?.message || 'Failed to parse CSV file');
      }
    };
    reader.readAsText(file);
    // Reset file input value so re-uploading same file works
    e.target.value = '';
  };

  const currentClassInfo = prediction ? STORAGE_CLASSES[prediction.classId] : null;

  return (
    <div className="space-y-6 pb-12">
      {/* Top Banner / Scenarios Bar */}
      <div className="bg-white rounded-xl p-4 sm:p-5 border border-slate-200 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-500">
              Diagnostic Scenario Generator
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Select an excursion event to simulate sensor telemetry, or upload field telemetry CSV:
            </p>
          </div>

          {/* Scenario Buttons */}
          <div className="flex flex-wrap items-center gap-2">
            {([0, 1, 2, 3, 4] as StorageClassId[]).map((cId) => {
              const info = STORAGE_CLASSES[cId];
              const isSelected = activeClassId === cId;
              return (
                <button
                  key={cId}
                  onClick={() => handleSimulateClass(cId)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition ${
                    isSelected
                      ? 'bg-slate-900 text-white border-slate-900 shadow-xs'
                      : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50 hover:border-slate-300'
                  }`}
                >
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: info.color }}
                  />
                  <span>{info.shortName}</span>
                </button>
              );
            })}

            <button
              onClick={handleSimulateRandom}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition"
              title="Generate a randomized scenario"
            >
              <Shuffle className="w-3.5 h-3.5 text-slate-600" />
              <span>Random</span>
            </button>

            <label className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-200 cursor-pointer transition">
              <Upload className="w-3.5 h-3.5 text-emerald-600" />
              <span>Upload CSV</span>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.txt"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
          </div>
        </div>

        {uploadError && (
          <div className="mt-3 p-2 text-xs bg-red-50 border border-red-200 text-red-700 rounded-lg flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{uploadError}</span>
          </div>
        )}
      </div>

      {/* Main Grid: Chart on Left, Diagnosis Card on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Telemetry Chart: 8 cols */}
        <div className="lg:col-span-7 xl:col-span-8 bg-white rounded-xl p-5 border border-slate-200 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="font-bold text-slate-900 text-base">
                    6-Hour Cold Storage Telemetry Window
                  </h3>
                  {activeClassId === 'custom' ? (
                    <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
                      Uploaded Telemetry
                    </span>
                  ) : (
                    <span
                      className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full border"
                      style={{
                        backgroundColor: STORAGE_CLASSES[activeClassId].bgLight,
                        borderColor: STORAGE_CLASSES[activeClassId].borderCol,
                        color: STORAGE_CLASSES[activeClassId].color,
                      }}
                    >
                      True: {STORAGE_CLASSES[activeClassId].name}
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-500 mt-0.5">
                  Sampled at 1-min intervals (360 points). Excursion shaded in light red when &gt; {settings.threshold.toFixed(1)} °C.
                </p>
              </div>

              {/* View Toggle Controls */}
              <div className="flex items-center gap-3 text-xs">
                <button
                  onClick={() => setShowRaw(!showRaw)}
                  className={`flex items-center gap-1 px-2.5 py-1 rounded-md border transition ${
                    showRaw
                      ? 'bg-slate-100 text-slate-700 border-slate-300'
                      : 'bg-white text-slate-400 border-slate-200 line-through'
                  }`}
                >
                  {showRaw ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                  <span>Raw Signal</span>
                </button>
                <button
                  onClick={() => setShowSmoothed(!showSmoothed)}
                  className={`flex items-center gap-1 px-2.5 py-1 rounded-md border transition ${
                    showSmoothed
                      ? 'bg-emerald-50 text-emerald-800 border-emerald-300 font-medium'
                      : 'bg-white text-slate-400 border-slate-200 line-through'
                  }`}
                >
                  {showSmoothed ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                  <span>Smoothed (Median+MA)</span>
                </button>
              </div>
            </div>

            {/* Recharts Telemetry Plot */}
            <div className="h-72 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  data={chartData}
                  margin={{ top: 10, right: 20, left: -10, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis
                    dataKey="minute"
                    tickFormatter={(val) => {
                      const h = Math.floor(val / 60);
                      const m = val % 60;
                      return `${h}h${m > 0 ? `${m}m` : ''}`;
                    }}
                    stroke="#94a3b8"
                    tick={{ fontSize: 11 }}
                    ticks={[0, 60, 120, 180, 240, 300, 359]}
                  />
                  <YAxis
                    domain={[
                      (dataMin: number) => Math.min(2.0, Math.floor(dataMin - 0.5)),
                      (dataMax: number) => Math.max(10.0, Math.ceil(dataMax + 1.0)),
                    ]}
                    unit="°C"
                    stroke="#94a3b8"
                    tick={{ fontSize: 11 }}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-slate-900/95 text-white p-3 rounded-lg shadow-xl text-xs space-y-1.5 border border-slate-800 backdrop-blur-xs">
                            <div className="font-semibold text-slate-300 border-b border-slate-800 pb-1 flex justify-between gap-4">
                              <span>Minute {data.minute} ({data.timeLabel})</span>
                              {data.isExcursion && (
                                <span className="text-red-400 font-bold uppercase tracking-wider text-[10px]">
                                  Excursion
                                </span>
                              )}
                            </div>
                            <div className="flex justify-between gap-4">
                              <span className="text-slate-400">Raw Temp:</span>
                              <span className="font-mono font-medium text-slate-200">
                                {data.rawTemp.toFixed(2)} °C
                              </span>
                            </div>
                            <div className="flex justify-between gap-4">
                              <span className="text-emerald-400 font-medium">Smoothed Temp:</span>
                              <span className="font-mono font-bold text-emerald-300">
                                {data.smoothedTemp.toFixed(2)} °C
                              </span>
                            </div>
                            <div className="flex justify-between gap-4 text-[10px] text-slate-400 border-t border-slate-800/60 pt-1">
                              <span>Setpoint: {settings.setpoint.toFixed(1)} °C</span>
                              <span>Threshold: {settings.threshold.toFixed(1)} °C</span>
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />

                  {/* Reference Lines */}
                  <ReferenceLine
                    y={settings.setpoint}
                    stroke="#059669"
                    strokeDasharray="4 4"
                    label={{
                      value: `Setpoint (${settings.setpoint}°C)`,
                      position: 'insideBottomLeft',
                      fill: '#059669',
                      fontSize: 10,
                      fontWeight: 600,
                    }}
                  />
                  <ReferenceLine
                    y={settings.threshold}
                    stroke="#dc2626"
                    strokeDasharray="4 4"
                    label={{
                      value: `Excursion Threshold (${settings.threshold}°C)`,
                      position: 'insideTopLeft',
                      fill: '#dc2626',
                      fontSize: 10,
                      fontWeight: 600,
                    }}
                  />

                  {/* Shaded Excursion Intervals */}
                  {excursionIntervals.map((interval, i) => (
                    <ReferenceArea
                      key={i}
                      x1={interval.start}
                      x2={interval.end}
                      y1={settings.threshold}
                      fill="#fee2e2"
                      fillOpacity={0.45}
                    />
                  ))}

                  {/* Raw Telemetry Line */}
                  {showRaw && (
                    <Line
                      type="monotone"
                      dataKey="rawTemp"
                      stroke="#94a3b8"
                      strokeWidth={1.2}
                      dot={false}
                      isAnimationActive={false}
                      name="Raw Sensor"
                    />
                  )}

                  {/* Filtered / Smoothed Temperature Line */}
                  {showSmoothed && (
                    <Line
                      type="monotone"
                      dataKey="smoothedTemp"
                      stroke="#0d9488"
                      strokeWidth={2.4}
                      dot={false}
                      isAnimationActive={false}
                      name="Smoothed (Filtered)"
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Excursion Summary Strip */}
          <div className="mt-4 pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between text-xs text-slate-600 gap-2">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-xs bg-red-100 border border-red-300" />
                <span>Excursion Duration: <strong>{features.duration_above_thresh} min</strong></span>
              </span>
              <span>Peak: <strong>{features.peak_temp.toFixed(1)} °C</strong></span>
              <span>Max Rise Rate: <strong>{features.max_slope.toFixed(2)} °C/min</strong></span>
            </div>
            <div className="text-[11px] text-slate-400">
              Filter: Median ({settings.medianWindow}m) + MA ({settings.maWindow}m)
            </div>
          </div>
        </div>

        {/* Prediction & Root Cause: 5 cols */}
        <div className="lg:col-span-5 xl:col-span-4 space-y-4">
          {/* Warning Banner when confidence < 60% */}
          {prediction && prediction.isLowConfidence && (
            <div className="bg-amber-50 border-l-4 border-amber-500 p-3 rounded-r-xl shadow-xs text-amber-900 text-xs">
              <div className="flex items-center gap-2 font-bold mb-0.5">
                <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                <span>Low Confidence Prediction ({Math.round(prediction.confidence * 100)}%)</span>
              </div>
              <p className="text-amber-800 text-[11px] leading-relaxed">
                Telemetry shows ambiguous transient characteristics. Review sensor calibration and conduct visual inspection.
              </p>
            </div>
          )}

          {/* Large Prediction Card */}
          <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs relative overflow-hidden">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                  Predicted Root Cause
                </span>
                <h3 className="text-xl font-bold text-slate-900 mt-1 flex items-center gap-2">
                  {currentClassInfo ? (
                    <>
                      <span
                        className="w-3 h-3 rounded-full shrink-0"
                        style={{ backgroundColor: currentClassInfo.color }}
                      />
                      <span>{currentClassInfo.name}</span>
                    </>
                  ) : (
                    <span className="text-slate-400">Model not trained</span>
                  )}
                </h3>
              </div>

              {prediction && (
                <div className="text-right">
                  <div className="text-2xl font-black text-slate-900">
                    {Math.round(prediction.confidence * 100)}%
                  </div>
                  <span className="text-[10px] uppercase font-semibold text-slate-400">
                    Confidence
                  </span>
                </div>
              )}
            </div>

            {/* Horizontal Probability Bars */}
            {prediction && (
              <div className="mt-5 space-y-2.5">
                <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Posterior Probability Distribution
                </div>
                {([0, 1, 2, 3, 4] as StorageClassId[]).map((cId) => {
                  const info = STORAGE_CLASSES[cId];
                  const prob = prediction.probabilities[cId] || 0;
                  const isTop = prediction.classId === cId;
                  const pct = Math.round(prob * 100);

                  return (
                    <div key={cId} className="space-y-1">
                      <div className="flex justify-between items-center text-xs">
                        <span
                          className={`font-medium ${
                            isTop ? 'text-slate-900 font-bold' : 'text-slate-600'
                          }`}
                        >
                          {info.shortName}
                        </span>
                        <span
                          className={`font-mono text-xs ${
                            isTop ? 'text-slate-900 font-bold' : 'text-slate-500'
                          }`}
                        >
                          {pct}%
                        </span>
                      </div>
                      <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden relative">
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{
                            width: `${pct}%`,
                            backgroundColor: isTop ? info.color : '#cbd5e1',
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Operator Guidance & Action Box */}
            {currentClassInfo && (
              <div className="mt-5 pt-4 border-t border-slate-100 space-y-2.5">
                <div>
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-600 flex items-center gap-1.5">
                    <ShieldAlert className="w-3.5 h-3.5 text-slate-500" />
                    <span>Operator Action Playbook</span>
                  </h4>
                  <div className="mt-1.5 p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700 leading-relaxed font-medium">
                    {currentClassInfo.operatorAction}
                  </div>
                </div>

                <div className="text-[11px] text-slate-500 leading-snug">
                  <strong>Diagnostic Rationale:</strong> {currentClassInfo.rootCauseDetails}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Extracted Diagnostic Features Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-emerald-600" />
            <h3 className="font-semibold text-slate-900 text-sm">
              Extracted Telemetry Features (Active Window)
            </h3>
          </div>
          <span className="text-xs text-slate-500">
            Window length: 360 min (6 hours)
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50/80 text-slate-600 uppercase tracking-wider border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-4 font-semibold">Feature</th>
                <th className="py-2.5 px-4 font-semibold">Computed Value</th>
                <th className="py-2.5 px-4 font-semibold">Unit</th>
                <th className="py-2.5 px-4 font-semibold">Diagnostic Significance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {FEATURE_NAMES.map((fName) => {
                const meta = FEATURE_LABELS[fName];
                const val = features[fName];
                const isHighlight =
                  (fName === 'duration_above_thresh' && val > 0) ||
                  (fName === 'num_spikes' && val > 0) ||
                  (fName === 'flatline_fraction' && val > 0.05) ||
                  (fName === 'peak_temp' && val > settings.threshold);

                return (
                  <tr
                    key={fName}
                    className={`hover:bg-slate-50/50 transition ${
                      isHighlight ? 'bg-amber-50/30' : ''
                    }`}
                  >
                    <td className="py-2 px-4 font-medium text-slate-900">
                      {meta.label}
                    </td>
                    <td className="py-2 px-4 font-mono font-bold text-slate-800">
                      {typeof val === 'number' ? (
                        fName === 'flatline_fraction'
                          ? `${(val * 100).toFixed(1)}%`
                          : val
                      ) : (
                        val
                      )}
                    </td>
                    <td className="py-2 px-4 text-slate-500">{meta.unit}</td>
                    <td className="py-2 px-4 text-slate-500">{meta.description}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
