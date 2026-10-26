import React from 'react';
import { Activity, Database, BarChart3, Settings, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { StorageSettings } from '../types/storage';

export type AppTab = 'dashboard' | 'datamodel' | 'evaluation';

interface NavbarProps {
  currentTab: AppTab;
  onTabChange: (tab: AppTab) => void;
  settings: StorageSettings;
  onOpenSettings: () => void;
  modelReady: boolean;
  isTraining: boolean;
  trainingProgress?: { percentage: number };
  sampleCount: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  onTabChange,
  settings,
  onOpenSettings,
  modelReady,
  isTraining,
  trainingProgress,
  sampleCount,
}) => {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-600 to-teal-700 flex items-center justify-center text-white shadow-md shadow-emerald-500/10">
              <span className="text-xl font-black">🥔</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-lg text-slate-900 tracking-tight leading-none">
                  ColdStore <span className="text-emerald-700 font-semibold">Diagnostics</span>
                </h1>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                  v1.0 GBDT
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-0.5">
                Table-Potato Cold Storage Root-Cause Classification
              </p>
            </div>
          </div>

          {/* Quick Metrics & Badges */}
          <div className="hidden lg:flex items-center gap-2 text-xs">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-50 border border-slate-200 text-slate-700">
              <span className="text-slate-400">Setpoint:</span>
              <span className="font-semibold text-emerald-700">{settings.setpoint.toFixed(1)} °C</span>
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-50 border border-slate-200 text-slate-700">
              <span className="text-slate-400">Threshold:</span>
              <span className="font-semibold text-amber-700">{settings.threshold.toFixed(1)} °C</span>
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-50 border border-slate-200 text-slate-700">
              <span className="text-slate-400">Dataset:</span>
              <span className="font-semibold text-slate-900">{sampleCount.toLocaleString()} windows</span>
            </div>

            {/* Model Status Indicator */}
            {isTraining ? (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-blue-50 border border-blue-200 text-blue-700 font-medium animate-pulse">
                <span className="w-2 h-2 rounded-full bg-blue-600 animate-ping" />
                Training {trainingProgress?.percentage ?? 0}%
              </div>
            ) : modelReady ? (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-50 border border-emerald-200 text-emerald-800 font-medium">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                Model Active
              </div>
            ) : (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-50 border border-amber-200 text-amber-800 font-medium">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                Untrained
              </div>
            )}
          </div>

          {/* Tab Navigation */}
          <nav className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200">
            <button
              onClick={() => onTabChange('dashboard')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition ${
                currentTab === 'dashboard'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Activity className="w-3.5 h-3.5 text-emerald-600" />
              <span>Dashboard</span>
            </button>
            <button
              onClick={() => onTabChange('datamodel')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition ${
                currentTab === 'datamodel'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Database className="w-3.5 h-3.5 text-blue-600" />
              <span>Data & Model</span>
            </button>
            <button
              onClick={() => onTabChange('evaluation')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition ${
                currentTab === 'evaluation'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5 text-purple-600" />
              <span>Evaluation</span>
            </button>
          </nav>

          {/* Settings Trigger */}
          <button
            onClick={onOpenSettings}
            className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition border border-transparent hover:border-slate-200"
            title="Configure Setpoint & Filters"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
};
