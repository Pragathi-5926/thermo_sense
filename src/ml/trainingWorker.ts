import {
  EvaluationMetrics,
  ModelHyperparameters,
  StorageClassId,
  StorageWindowSample,
} from '../types/storage';
import { SerializedModel, XGBoostClassifier } from './xgboost';

export interface WorkerTrainingPayload {
  samples: StorageWindowSample[];
  hyperparameters: ModelHyperparameters;
}

export type WorkerInMessage =
  | { type: 'START_TRAINING'; payload: WorkerTrainingPayload }
  | { type: 'ABORT' };

export type WorkerOutMessage =
  | {
      type: 'PROGRESS';
      payload: {
        round: number;
        totalRounds: number;
        trainLoss: number;
        valLoss?: number;
      };
    }
  | {
      type: 'COMPLETE';
      payload: {
        modelData: SerializedModel;
        metrics: EvaluationMetrics;
      };
    }
  | {
      type: 'ERROR';
      payload: {
        error: string;
      };
    };

/**
 * Stratified 80/20 split
 */
export function stratifiedSplit(samples: StorageWindowSample[], trainRatio: number = 0.8) {
  const byClass: Record<StorageClassId, StorageWindowSample[]> = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
  };

  for (const s of samples) {
    byClass[s.classId].push(s);
  }

  const trainSamples: StorageWindowSample[] = [];
  const testSamples: StorageWindowSample[] = [];

  for (let k = 0; k < 5; k++) {
    const classItems = byClass[k as StorageClassId];
    const trainCount = Math.floor(classItems.length * trainRatio);
    for (let i = 0; i < classItems.length; i++) {
      if (i < trainCount) {
        trainSamples.push(classItems[i]);
      } else {
        testSamples.push(classItems[i]);
      }
    }
  }

  return { trainSamples, testSamples };
}

// Check if running in Worker context
if (typeof self !== 'undefined' && typeof (self as any).postMessage === 'function') {
  self.onmessage = (event: MessageEvent<WorkerInMessage>) => {
    const message = event.data;
    if (message.type === 'START_TRAINING') {
      try {
        const { samples, hyperparameters } = message.payload;
        const { trainSamples, testSamples } = stratifiedSplit(samples, 0.8);

        const XTrain = trainSamples.map((s) => XGBoostClassifier.featureToVector(s.features));
        const yTrain = trainSamples.map((s) => s.classId);

        const XTest = testSamples.map((s) => XGBoostClassifier.featureToVector(s.features));
        const yTest = testSamples.map((s) => s.classId);

        const model = new XGBoostClassifier(hyperparameters);

        model.train(
          XTrain,
          yTrain,
          XTest,
          yTest,
          (round, trainLoss, valLoss) => {
            self.postMessage({
              type: 'PROGRESS',
              payload: {
                round,
                totalRounds: hyperparameters.numTrees,
                trainLoss,
                valLoss,
              },
            });
          }
        );

        const metrics = model.evaluate(XTest, yTest);
        metrics.sampleCount = {
          train: trainSamples.length,
          test: testSamples.length,
        };

        const serialized = model.toJSON();

        self.postMessage({
          type: 'COMPLETE',
          payload: {
            modelData: serialized,
            metrics,
          },
        });
      } catch (err: any) {
        self.postMessage({
          type: 'ERROR',
          payload: {
            error: err?.message || String(err),
          },
        });
      }
    }
  };
}
