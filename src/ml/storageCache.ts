import {
  EvaluationMetrics,
  ModelHyperparameters,
  StorageSettings,
  StorageWindowSample,
} from '../types/storage';
import { SerializedModel } from './xgboost';

const DB_NAME = 'ColdStoreDiagnosticsDB';
const DB_VERSION = 1;
const STORE_DATASET = 'dataset';
const STORE_MODEL = 'model';

const SETTINGS_KEY = 'coldstore_settings_v1';
const HYPERPARAMS_KEY = 'coldstore_hyperparams_v1';

export const DEFAULT_SETTINGS: StorageSettings = {
  setpoint: 5.0,
  threshold: 7.0,
  medianWindow: 5,
  maWindow: 7,
  datasetSize: 2000,
};

export const DEFAULT_HYPERPARAMETERS: ModelHyperparameters = {
  numTrees: 100,
  maxDepth: 4,
  learningRate: 0.1,
  lambda: 1.0,
  minChildWeight: 1.0,
  subsample: 0.85,
  colsample: 0.85,
};

/**
 * Open or upgrade IndexedDB
 */
function openDB(): Promise<IDBDatabase | null> {
  if (typeof window === 'undefined' || !window.indexedDB) {
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    try {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = (e) => {
        const db = (e.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(STORE_DATASET)) {
          db.createObjectStore(STORE_DATASET);
        }
        if (!db.objectStoreNames.contains(STORE_MODEL)) {
          db.createObjectStore(STORE_MODEL);
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

/**
 * Cache dataset
 */
export async function cacheDataset(samples: StorageWindowSample[]): Promise<void> {
  const db = await openDB();
  if (db) {
    try {
      const tx = db.transaction(STORE_DATASET, 'readwrite');
      const store = tx.objectStore(STORE_DATASET);
      store.put(samples, 'current_dataset');
      return new Promise((res) => {
        tx.oncomplete = () => res();
        tx.onerror = () => res();
      });
    } catch {
      // ignore
    }
  }

  // Fallback: localStorage with sampling if needed
  try {
    const subset = samples.slice(0, 500); // smaller subset if in localStorage
    localStorage.setItem('coldstore_cached_samples', JSON.stringify(subset));
  } catch {
    // storage limit reached, pass
  }
}

/**
 * Load cached dataset
 */
export async function loadCachedDataset(): Promise<StorageWindowSample[] | null> {
  const db = await openDB();
  if (db) {
    try {
      const tx = db.transaction(STORE_DATASET, 'readonly');
      const store = tx.objectStore(STORE_DATASET);
      const req = store.get('current_dataset');
      return new Promise((resolve) => {
        req.onsuccess = () => {
          if (req.result && Array.isArray(req.result) && req.result.length > 0) {
            resolve(req.result);
          } else {
            resolve(null);
          }
        };
        req.onerror = () => resolve(null);
      });
    } catch {
      // continue to fallback
    }
  }

  try {
    const raw = localStorage.getItem('coldstore_cached_samples');
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch {
    // ignore
  }

  return null;
}

/**
 * Cache trained model and evaluation metrics
 */
export async function cacheModelAndMetrics(
  modelData: SerializedModel,
  metrics: EvaluationMetrics
): Promise<void> {
  const db = await openDB();
  if (db) {
    try {
      const tx = db.transaction(STORE_MODEL, 'readwrite');
      const store = tx.objectStore(STORE_MODEL);
      store.put({ modelData, metrics, timestamp: Date.now() }, 'current_model');
      return new Promise((res) => {
        tx.oncomplete = () => res();
        tx.onerror = () => res();
      });
    } catch {
      // ignore
    }
  }

  try {
    localStorage.setItem(
      'coldstore_cached_model',
      JSON.stringify({ modelData, metrics, timestamp: Date.now() })
    );
  } catch {
    // quota exceeded
  }
}

/**
 * Load cached model & metrics
 */
export async function loadCachedModelAndMetrics(): Promise<{
  modelData: SerializedModel;
  metrics: EvaluationMetrics;
} | null> {
  const db = await openDB();
  if (db) {
    try {
      const tx = db.transaction(STORE_MODEL, 'readonly');
      const store = tx.objectStore(STORE_MODEL);
      const req = store.get('current_model');
      return new Promise((resolve) => {
        req.onsuccess = () => {
          if (req.result && req.result.modelData && req.result.metrics) {
            resolve(req.result);
          } else {
            resolve(null);
          }
        };
        req.onerror = () => resolve(null);
      });
    } catch {
      // continue to fallback
    }
  }

  try {
    const raw = localStorage.getItem('coldstore_cached_model');
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed?.modelData && parsed?.metrics) return parsed;
    }
  } catch {
    // ignore
  }

  return null;
}

/**
 * Settings persistence
 */
export function loadSettings(): StorageSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    // ignore
  }
  return DEFAULT_SETTINGS;
}

export function saveSettings(settings: StorageSettings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // ignore
  }
}

/**
 * Hyperparameters persistence
 */
export function loadHyperparameters(): ModelHyperparameters {
  try {
    const raw = localStorage.getItem(HYPERPARAMS_KEY);
    if (raw) return { ...DEFAULT_HYPERPARAMETERS, ...JSON.parse(raw) };
  } catch {
    // ignore
  }
  return DEFAULT_HYPERPARAMETERS;
}

export function saveHyperparameters(params: ModelHyperparameters): void {
  try {
    localStorage.setItem(HYPERPARAMS_KEY, JSON.stringify(params));
  } catch {
    // ignore
  }
}
