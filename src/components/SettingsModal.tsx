import React, { useState } from 'react';
import { StorageSettings } from '../types/storage';
import { Sliders, X, RotateCcw, Check } from 'lucide-react';
import { DEFAULT_SETTINGS } from '../ml/storageCache';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: StorageSettings;
  onSave: (newSettings: StorageSettings) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSave,
}) => {
  const [formData, setFormData] = useState<StorageSettings>({ ...settings });

  if (!isOpen) return null;

  const handleReset = () => {
    setFormData({ ...DEFAULT_SETTINGS });
  };

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(formData);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4 animate-in fade-in duration-150">
      <div className="bg-white rounded-xl shadow-2xl border border-slate-200 w-full max-w-lg overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 bg-slate-50/50">
          <div className="flex items-center gap-2 text-slate-800">
            <Sliders className="w-5 h-5 text-emerald-600" />
            <h2 className="font-semibold text-lg">Storage & Filter Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 transition-colors p-1 rounded-lg hover:bg-slate-100"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form Content */}
        <form onSubmit={handleSave} className="p-6 space-y-5">
          <div className="space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Cold Storage Temperature Targets
            </h3>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Setpoint (°C)
                </label>
                <div className="relative">
                  <input
                    type="number"
                    step="0.1"
                    min="2"
                    max="10"
                    value={formData.setpoint}
                    onChange={(e) =>
                      setFormData({ ...formData, setpoint: parseFloat(e.target.value) || 5.0 })
                    }
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition"
                  />
                  <span className="absolute right-3 top-2.5 text-xs text-slate-400">°C</span>
                </div>
                <p className="text-xs text-slate-500 mt-1">Recommended for table potatoes: 4.0–8.0 °C</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Excursion Threshold (°C)
                </label>
                <div className="relative">
                  <input
                    type="number"
                    step="0.1"
                    min="5"
                    max="14"
                    value={formData.threshold}
                    onChange={(e) =>
                      setFormData({ ...formData, threshold: parseFloat(e.target.value) || 7.0 })
                    }
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-800 focus:ring-2 focus:ring-amber-500 focus:border-amber-500 outline-none transition"
                  />
                  <span className="absolute right-3 top-2.5 text-xs text-slate-400">°C</span>
                </div>
                <p className="text-xs text-slate-500 mt-1">Triggers diagnostic alarm above this</p>
              </div>
            </div>
          </div>

          <div className="space-y-4 pt-2 border-t border-slate-100">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Preprocessing Filter Windows
            </h3>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Median Filter Size
                </label>
                <input
                  type="number"
                  step="2"
                  min="3"
                  max="15"
                  value={formData.medianWindow}
                  onChange={(e) => {
                    let val = parseInt(e.target.value, 10) || 5;
                    if (val % 2 === 0) val += 1;
                    setFormData({ ...formData, medianWindow: val });
                  }}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition"
                />
                <p className="text-xs text-slate-500 mt-1">Removes unphysical sensor spikes</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Moving Average Size
                </label>
                <input
                  type="number"
                  step="1"
                  min="1"
                  max="21"
                  value={formData.maWindow}
                  onChange={(e) =>
                    setFormData({ ...formData, maWindow: parseInt(e.target.value, 10) || 7 })
                  }
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition"
                />
                <p className="text-xs text-slate-500 mt-1">Smoothes high-frequency noise</p>
              </div>
            </div>
          </div>

          {/* Footer Actions */}
          <div className="flex items-center justify-between pt-4 border-t border-slate-100">
            <button
              type="button"
              onClick={handleReset}
              className="flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-slate-800 px-3 py-1.5 rounded-lg hover:bg-slate-100 transition"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Defaults
            </button>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-800 rounded-lg hover:bg-slate-100 transition"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="flex items-center gap-1.5 px-4 py-2 text-sm font-medium text-white bg-emerald-600 hover:bg-emerald-700 rounded-lg shadow-xs transition"
              >
                <Check className="w-4 h-4" />
                Apply Changes
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};
