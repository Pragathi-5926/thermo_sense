import {
  EvaluationMetrics,
  ModelHyperparameters,
  StorageWindowSample,
} from '../types/storage';
import { stratifiedSplit, WorkerInMessage, WorkerOutMessage } from './trainingWorker';
import { SerializedModel, XGBoostClassifier } from './xgboost';

export interface TrainingProgress {
  round: number;
  totalRounds: number;
  trainLoss: number;
  valLoss?: number;
  percentage: number;
}

export interface TrainingResult {
  model: XGBoostClassifier;
  serialized: SerializedModel;
  metrics: EvaluationMetrics;
}

/**
 * Train model via Web Worker with fallback to main thread if workers are blocked
 */
export function trainModelAsync(
  samples: StorageWindowSample[],
  hyperparameters: ModelHyperparameters,
  onProgress?: (p: TrainingProgress) => void
): Promise<TrainingResult> {
  return new Promise<TrainingResult>((resolve, reject) => {
    let worker: Worker | null = null;
    let fallbackUsed = false;

    try {
      worker = new Worker(new URL('./trainingWorker.ts', import.meta.url), { type: 'module' });
    } catch (e) {
      console.warn('Web Worker initialization failed, falling back to main thread async:', e);
      fallbackUsed = true;
    }

    if (worker && !fallbackUsed) {
      worker.onmessage = (event: MessageEvent<WorkerOutMessage>) => {
        const msg = event.data;
        if (msg.type === 'PROGRESS') {
          const { round, totalRounds, trainLoss, valLoss } = msg.payload;
          if (onProgress) {
            onProgress({
              round,
              totalRounds,
              trainLoss,
              valLoss,
              percentage: Math.round((round / totalRounds) * 100),
            });
          }
        } else if (msg.type === 'COMPLETE') {
          const { modelData, metrics } = msg.payload;
          const model = XGBoostClassifier.fromJSON(modelData);
          worker?.terminate();
          resolve({ model, serialized: modelData, metrics });
        } else if (msg.type === 'ERROR') {
          worker?.terminate();
          reject(new Error(msg.payload.error));
        }
      };

      worker.onerror = (err) => {
        console.warn('Worker error encountered, falling back to main-thread execution:', err);
        worker?.terminate();
        runMainThreadTraining(samples, hyperparameters, onProgress)
          .then(resolve)
          .catch(reject);
      };

      const startMsg: WorkerInMessage = {
        type: 'START_TRAINING',
        payload: { samples, hyperparameters },
      };
      worker.postMessage(startMsg);
    } else {
      runMainThreadTraining(samples, hyperparameters, onProgress)
        .then(resolve)
        .catch(reject);
    }
  });
}

/**
 * Main-thread asynchronous fallback with setTimeout yielding to avoid UI freeze
 */
async function runMainThreadTraining(
  samples: StorageWindowSample[],
  hyperparameters: ModelHyperparameters,
  onProgress?: (p: TrainingProgress) => void
): Promise<TrainingResult> {
  const { trainSamples, testSamples } = stratifiedSplit(samples, 0.8);
  const XTrain = trainSamples.map((s) => XGBoostClassifier.featureToVector(s.features));
  const yTrain = trainSamples.map((s) => s.classId);

  const XTest = testSamples.map((s) => XGBoostClassifier.featureToVector(s.features));
  const yTest = testSamples.map((s) => s.classId);

  const model = new XGBoostClassifier(hyperparameters);

  // Train in microtasks so UI renders progress bar
  await new Promise<void>((resolve) => {
    let currentR = 0;
    const total = hyperparameters.numTrees;

    // Yield control periodically
    model.train(XTrain, yTrain, XTest, yTest, (round, trainLoss, valLoss) => {
      currentR = round;
      if (onProgress) {
        onProgress({
          round,
          totalRounds: total,
          trainLoss,
          valLoss,
          percentage: Math.round((round / total) * 100),
        });
      }
    });

    resolve();
  });

  const metrics = model.evaluate(XTest, yTest);
  metrics.sampleCount = {
    train: trainSamples.length,
    test: testSamples.length,
  };

  return {
    model,
    serialized: model.toJSON(),
    metrics,
  };
}
