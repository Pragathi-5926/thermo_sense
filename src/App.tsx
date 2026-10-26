/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  EvaluationMetrics,
  ModelHyperparameters,
  StorageSettings,
  StorageWindowSample,
} from './types/storage';
import {
  DEFAULT_HYPERPARAMETERS,
  DEFAULT_SETTINGS,
  loadSettings,
  saveSettings,
  loadHyperparameters,
  saveHyperparameters,
  cacheDataset,
  loadCachedDataset,
  cacheModelAndMetrics,
  loadCachedModelAndMetrics,
} from './ml/storageCache';
import { generateSyntheticDataset } from './utils/syntheticData';
import { XGBoostClassifier } from './ml/xgboost';
import { trainModelAsync, TrainingProgress } from './ml/trainerClient';
import { Navbar, AppTab } from './components/Navbar';
import { DashboardTab } from './components/DashboardTab';
import { DataModelTab } from './components/DataModelTab';
import { EvaluationTab } from './components/EvaluationTab';
import { SettingsModal } from './components/SettingsModal';
import { AlertCircle, CheckCircle, Sparkles, Loader2 } from 'lucide-react';

export default function App() {
  const [currentTab, setCurrentTab] = useState<AppTab>('dashboard');
  const [settings, setSettings] = useState<StorageSettings>(loadSettings);
  const [hyperparameters, setHyperparameters] = useState<ModelHyperparameters>(loadHyperparameters);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Dataset & Model State
  const [samples, setSamples] = useState<StorageWindowSample[]>([]);
  const [model, setModel] = useState<XGBoostClassifier | null>(null);
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [selectedSample, setSelectedSample] = useState<StorageWindowSample | null>(null);

  // Loading / Training states
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | undefined>(undefined);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // Keep a ref to avoid double initializations in React StrictMode
  const initialLoadExecuted = useRef(false);

  // Function to train model
  const trainModel = useCallback(
    async (datasetToTrain: StorageWindowSample[], params: ModelHyperparameters) => {
      if (datasetToTrain.length === 0) return;
      setIsTraining(true);
      setStatusMessage('Training XGBoost model in Web Worker...');

      try {
        const result = await trainModelAsync(datasetToTrain, params, (progress) => {
          setTrainingProgress(progress);
        });

        setModel(result.model);
        setMetrics(result.metrics);
        setStatusMessage(
          `Model trained successfully! Test Accuracy: ${(result.metrics.accuracy * 100).toFixed(1)}%`
        );

        // Cache model and metrics for subsequent visits
        await cacheModelAndMetrics(result.serialized, result.metrics);
      } catch (err: any) {
        console.error('Training failed:', err);
        setStatusMessage(`Training failed: ${err?.message || 'Unknown error'}`);
      } finally {
        setIsTraining(false);
        setTrainingProgress(undefined);
        setTimeout(() => setStatusMessage(null), 5000);
      }
    },
    []
  );

  // Function to generate new synthetic dataset
  const generateNewDataset = useCallback(
    async (count: number) => {
      setIsGenerating(true);
      setStatusMessage(`Generating ${count} synthetic 6-hour cold storage windows...`);

      // Yield frame so UI updates immediately
      await new Promise((r) => setTimeout(r, 50));

      try {
        const newSamples = generateSyntheticDataset(count, settings);
        setSamples(newSamples);
        await cacheDataset(newSamples);
        setStatusMessage(`Generated ${newSamples.length} samples. Training model...`);

        // Automatically train model on the newly generated dataset
        await trainModel(newSamples, hyperparameters);
      } catch (err: any) {
        console.error('Generation failed:', err);
        setStatusMessage('Error generating dataset.');
      } finally {
        setIsGenerating(false);
      }
    },
    [settings, hyperparameters, trainModel]
  );

  // First-load initialization lifecycle
  useEffect(() => {
    if (initialLoadExecuted.current) return;
    initialLoadExecuted.current = true;

    async function initialize() {
      // 1. Check if cached model & dataset exist
      const cachedModelData = await loadCachedModelAndMetrics();
      const cachedSamples = await loadCachedDataset();

      if (cachedModelData && cachedSamples && cachedSamples.length > 0) {
        setSamples(cachedSamples);
        const restoredModel = XGBoostClassifier.fromJSON(cachedModelData.modelData);
        setModel(restoredModel);
        setMetrics(cachedModelData.metrics);
        return;
      }

      // 2. If no cache, auto-generate dataset and train immediately
      const initialCount = settings.datasetSize || 2000;
      const initialSamples = generateSyntheticDataset(initialCount, settings);
      setSamples(initialSamples);
      await cacheDataset(initialSamples);

      // Train model
      await trainModel(initialSamples, hyperparameters);
    }

    initialize();
  }, [settings, hyperparameters, trainModel]);

  // Handle saving storage settings
  const handleSaveSettings = (newSettings: StorageSettings) => {
    setSettings(newSettings);
    saveSettings(newSettings);
    // Note: features for active sample will re-compute automatically in DashboardTab
  };

  // Handle updating hyperparameters
  const handleUpdateHyperparameters = (newParams: ModelHyperparameters) => {
    setHyperparameters(newParams);
    saveHyperparameters(newParams);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans selection:bg-emerald-100 selection:text-emerald-900">
      {/* Top Navigation */}
      <Navbar
        currentTab={currentTab}
        onTabChange={setCurrentTab}
        settings={settings}
        onOpenSettings={() => setIsSettingsOpen(true)}
        modelReady={model !== null}
        isTraining={isTraining}
        trainingProgress={trainingProgress}
        sampleCount={samples.length}
      />

      {/* Floating Status Notification Toast */}
      {statusMessage && (
        <div className="fixed bottom-12 right-6 z-50 bg-slate-900 text-white text-xs px-4 py-2.5 rounded-xl shadow-2xl flex items-center gap-2.5 border border-slate-700 animate-in slide-in-from-bottom-3 duration-200">
          {isTraining || isGenerating ? (
            <Loader2 className="w-4 h-4 text-emerald-400 animate-spin shrink-0" />
          ) : (
            <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
          )}
          <span>{statusMessage}</span>
        </div>
      )}

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-6">
        {currentTab === 'dashboard' && (
          <DashboardTab
            settings={settings}
            model={model}
            onOpenSettings={() => setIsSettingsOpen(true)}
            selectedSample={selectedSample}
          />
        )}

        {currentTab === 'datamodel' && (
          <DataModelTab
            settings={settings}
            samples={samples}
            hyperparameters={hyperparameters}
            onUpdateHyperparameters={handleUpdateHyperparameters}
            onGenerateDataset={(count) => generateNewDataset(count)}
            onTrainModel={() => trainModel(samples, hyperparameters)}
            isTraining={isTraining}
            trainingProgress={trainingProgress}
            isGenerating={isGenerating}
            onSelectSampleForDashboard={(sample) => {
              setSelectedSample(sample);
              setCurrentTab('dashboard');
            }}
          />
        )}

        {currentTab === 'evaluation' && (
          <EvaluationTab
            metrics={metrics}
            hyperparameters={hyperparameters}
            settings={settings}
          />
        )}
      </main>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSave={handleSaveSettings}
      />

      {/* Required Operational Footer */}
      <footer className="border-t border-slate-200 bg-white/70 py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-700">ColdStore Diagnostics</span>
            <span>•</span>
            <span>Table-Potato Storage Excursion Classifier</span>
          </div>
          <div className="text-[11px] text-slate-400 bg-slate-50 px-2.5 py-1 rounded-md border border-slate-200">
            Trained on simulated data. Not validated on operational facilities.
          </div>
        </div>
      </footer>
    </div>
  );
}
