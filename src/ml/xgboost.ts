import {
  EvaluationMetrics,
  ExtractedFeatures,
  FEATURE_NAMES,
  ModelHyperparameters,
  PredictionResult,
  STORAGE_CLASSES,
  StorageClassId,
} from '../types/storage';

export interface TreeNode {
  isLeaf: boolean;
  leafValue?: number;
  featureIndex?: number;
  featureName?: keyof ExtractedFeatures;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  gain?: number;
}

export interface ClassTree {
  classId: StorageClassId;
  root: TreeNode;
}

export interface BoostingRound {
  trees: ClassTree[];
}

export interface SerializedModel {
  hyperparameters: ModelHyperparameters;
  rounds: BoostingRound[];
  numClasses: number;
  featureNames: (keyof ExtractedFeatures)[];
  featureGainSum: Record<keyof ExtractedFeatures, number>;
}

/**
 * Gradient Boosted Decision Tree (XGBoost multiclass softmax formulation)
 */
export class XGBoostClassifier {
  hyperparameters: ModelHyperparameters;
  rounds: BoostingRound[] = [];
  numClasses: number = 5;
  featureNames: (keyof ExtractedFeatures)[] = [...FEATURE_NAMES];
  featureGainSum: Record<keyof ExtractedFeatures, number> = {
    mean_temp: 0,
    std_temp: 0,
    mean_rate_rise: 0,
    duration_above_thresh: 0,
    max_slope: 0,
    recovery_time: 0,
    peak_temp: 0,
    num_spikes: 0,
    flatline_fraction: 0,
  };

  constructor(params?: Partial<ModelHyperparameters>) {
    this.hyperparameters = {
      numTrees: params?.numTrees ?? 100,
      maxDepth: params?.maxDepth ?? 4,
      learningRate: params?.learningRate ?? 0.1,
      lambda: params?.lambda ?? 1.0,
      minChildWeight: params?.minChildWeight ?? 1.0,
      subsample: params?.subsample ?? 0.85,
      colsample: params?.colsample ?? 0.85,
    };
  }

  /**
   * Convert feature object to vector
   */
  static featureToVector(f: ExtractedFeatures): number[] {
    return [
      f.mean_temp,
      f.std_temp,
      f.mean_rate_rise,
      f.duration_above_thresh,
      f.max_slope,
      f.recovery_time,
      f.peak_temp,
      f.num_spikes,
      f.flatline_fraction,
    ];
  }

  /**
   * Calculate softmax probabilities from logits
   */
  static softmax(logits: number[]): number[] {
    let maxLogit = -Infinity;
    for (const v of logits) {
      if (v > maxLogit) maxLogit = v;
    }
    let sumExp = 0;
    const expVals = new Array<number>(logits.length);
    for (let k = 0; k < logits.length; k++) {
      const e = Math.exp(logits[k] - maxLogit);
      expVals[k] = e;
      sumExp += e;
    }
    return expVals.map((e) => (sumExp > 0 ? e / sumExp : 1 / logits.length));
  }

  /**
   * Train model with optional progress callback
   */
  train(
    XTrain: number[][],
    yTrain: StorageClassId[],
    XVal?: number[][],
    yVal?: StorageClassId[],
    onProgress?: (round: number, trainLoss: number, valLoss?: number) => void
  ): void {
    const N = XTrain.length;
    const K = this.numClasses;
    const { numTrees, maxDepth, learningRate, lambda, minChildWeight, subsample, colsample } =
      this.hyperparameters;

    this.rounds = [];
    // Reset feature gain sum
    for (const name of this.featureNames) {
      this.featureGainSum[name] = 0;
    }

    // Initialize raw logits to 0
    const logits: number[][] = Array.from({ length: N }, () => new Array(K).fill(0));
    const valLogits: number[][] | null = XVal
      ? Array.from({ length: XVal.length }, () => new Array(K).fill(0))
      : null;

    // Feature matrix transpose or column arrays for faster split finding
    const numFeatures = this.featureNames.length;

    for (let r = 0; r < numTrees; r++) {
      const currentRoundTrees: ClassTree[] = [];

      // 1. Compute current probabilities and gradients for all training samples
      const probs: number[][] = new Array(N);
      const grads: number[][] = Array.from({ length: K }, () => new Array(N));
      const hess: number[][] = Array.from({ length: K }, () => new Array(N));

      let trainLoss = 0;
      for (let i = 0; i < N; i++) {
        const p = XGBoostClassifier.softmax(logits[i]);
        probs[i] = p;
        const actualClass = yTrain[i];
        trainLoss -= Math.log(Math.max(p[actualClass], 1e-15));

        for (let k = 0; k < K; k++) {
          const y_ik = actualClass === k ? 1 : 0;
          grads[k][i] = p[k] - y_ik;
          hess[k][i] = Math.max(p[k] * (1 - p[k]), 1e-16);
        }
      }
      trainLoss /= N;

      // 2. Build one tree for each class k
      for (let k = 0; k < K; k++) {
        // Row subsampling
        let sampleIndices: number[] = [];
        if (subsample >= 1.0) {
          sampleIndices = Array.from({ length: N }, (_, i) => i);
        } else {
          for (let i = 0; i < N; i++) {
            if (Math.random() < subsample) {
              sampleIndices.push(i);
            }
          }
          if (sampleIndices.length < 10) {
            sampleIndices = Array.from({ length: N }, (_, i) => i);
          }
        }

        // Build tree
        const root = this.buildNode(
          XTrain,
          grads[k],
          hess[k],
          sampleIndices,
          0,
          maxDepth,
          lambda,
          minChildWeight,
          colsample,
          numFeatures
        );

        currentRoundTrees.push({
          classId: k as StorageClassId,
          root,
        });

        // Update logits for training set
        for (let i = 0; i < N; i++) {
          const update = this.predictTree(root, XTrain[i]);
          logits[i][k] += learningRate * update;
        }

        // Update validation logits if available
        if (XVal && valLogits) {
          for (let i = 0; i < XVal.length; i++) {
            const update = this.predictTree(root, XVal[i]);
            valLogits[i][k] += learningRate * update;
          }
        }
      }

      this.rounds.push({ trees: currentRoundTrees });

      // Compute validation loss if requested
      let valLoss: number | undefined;
      if (XVal && yVal && valLogits) {
        let vLoss = 0;
        for (let i = 0; i < XVal.length; i++) {
          const vp = XGBoostClassifier.softmax(valLogits[i]);
          vLoss -= Math.log(Math.max(vp[yVal[i]], 1e-15));
        }
        valLoss = vLoss / XVal.length;
      }

      if (onProgress && (r % 5 === 0 || r === numTrees - 1)) {
        onProgress(r + 1, trainLoss, valLoss);
      }
    }
  }

  /**
   * Recursive tree construction with second-order gradient split search
   */
  private buildNode(
    X: number[][],
    g: number[],
    h: number[],
    indices: number[],
    depth: number,
    maxDepth: number,
    lambda: number,
    minChildWeight: number,
    colsample: number,
    numFeatures: number
  ): TreeNode {
    // Sum gradients & hessians in current node
    let G = 0;
    let H = 0;
    for (const idx of indices) {
      G += g[idx];
      H += h[idx];
    }

    // Leaf weight calculation
    const leafValue = -G / (H + lambda);

    // Stop conditions
    if (depth >= maxDepth || indices.length <= 5 || H < minChildWeight) {
      return { isLeaf: true, leafValue };
    }

    // Column subsampling
    const candidateFeatures: number[] = [];
    for (let f = 0; f < numFeatures; f++) {
      if (Math.random() <= colsample) {
        candidateFeatures.push(f);
      }
    }
    if (candidateFeatures.length === 0) {
      candidateFeatures.push(Math.floor(Math.random() * numFeatures));
    }

    let bestGain = 0;
    let bestFeature = -1;
    let bestThreshold = 0;
    let bestLeftIndices: number[] = [];
    let bestRightIndices: number[] = [];

    const parentScore = (G * G) / (H + lambda);

    // Evaluate splits for selected features
    for (const f of candidateFeatures) {
      // Collect values and sort
      const pairs: { val: number; idx: number }[] = new Array(indices.length);
      for (let i = 0; i < indices.length; i++) {
        const rowIdx = indices[i];
        pairs[i] = { val: X[rowIdx][f], idx: rowIdx };
      }
      pairs.sort((a, b) => a.val - b.val);

      let GL = 0;
      let HL = 0;

      // Scan through sorted values
      for (let i = 0; i < pairs.length - 1; i++) {
        const rowIdx = pairs[i].idx;
        GL += g[rowIdx];
        HL += h[rowIdx];

        // Skip identical consecutive values
        if (pairs[i].val === pairs[i + 1].val) continue;

        const GR = G - GL;
        const HR = H - HL;

        if (HL < minChildWeight || HR < minChildWeight) continue;

        const leftScore = (GL * GL) / (HL + lambda);
        const rightScore = (GR * GR) / (HR + lambda);
        const gain = 0.5 * (leftScore + rightScore - parentScore);

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = f;
          bestThreshold = (pairs[i].val + pairs[i + 1].val) / 2;
        }
      }
    }

    // If no gain found, return leaf
    if (bestGain <= 1e-6 || bestFeature === -1) {
      return { isLeaf: true, leafValue };
    }

    // Partition indices with best split
    for (const idx of indices) {
      if (X[idx][bestFeature] <= bestThreshold) {
        bestLeftIndices.push(idx);
      } else {
        bestRightIndices.push(idx);
      }
    }

    // Track feature importance gain
    const featName = this.featureNames[bestFeature];
    this.featureGainSum[featName] = (this.featureGainSum[featName] || 0) + bestGain;

    const left = this.buildNode(
      X,
      g,
      h,
      bestLeftIndices,
      depth + 1,
      maxDepth,
      lambda,
      minChildWeight,
      colsample,
      numFeatures
    );
    const right = this.buildNode(
      X,
      g,
      h,
      bestRightIndices,
      depth + 1,
      maxDepth,
      lambda,
      minChildWeight,
      colsample,
      numFeatures
    );

    return {
      isLeaf: false,
      featureIndex: bestFeature,
      featureName: featName,
      threshold: bestThreshold,
      gain: bestGain,
      left,
      right,
    };
  }

  /**
   * Traverse single tree to find leaf value
   */
  private predictTree(node: TreeNode, x: number[]): number {
    if (node.isLeaf) {
      return node.leafValue ?? 0;
    }
    const val = x[node.featureIndex!];
    if (val <= node.threshold!) {
      return this.predictTree(node.left!, x);
    } else {
      return this.predictTree(node.right!, x);
    }
  }

  /**
   * Predict raw logits for an input feature vector
   */
  predictLogits(x: number[]): number[] {
    const K = this.numClasses;
    const logits = new Array<number>(K).fill(0);
    const lr = this.hyperparameters.learningRate;

    for (const round of this.rounds) {
      for (let k = 0; k < K; k++) {
        const tree = round.trees[k];
        if (tree) {
          logits[k] += lr * this.predictTree(tree.root, x);
        }
      }
    }
    return logits;
  }

  /**
   * Predict class probabilities
   */
  predictProba(x: number[]): number[] {
    const logits = this.predictLogits(x);
    return XGBoostClassifier.softmax(logits);
  }

  /**
   * Full prediction on an ExtractedFeatures object
   */
  predict(features: ExtractedFeatures): PredictionResult {
    const vec = XGBoostClassifier.featureToVector(features);
    const proba = this.predictProba(vec);

    let maxProb = -1;
    let predClass: StorageClassId = 0;

    const probabilities: Record<StorageClassId, number> = {
      0: proba[0] ?? 0,
      1: proba[1] ?? 0,
      2: proba[2] ?? 0,
      3: proba[3] ?? 0,
      4: proba[4] ?? 0,
    };

    for (let k = 0; k < 5; k++) {
      if (proba[k] > maxProb) {
        maxProb = proba[k];
        predClass = k as StorageClassId;
      }
    }

    return {
      classId: predClass,
      className: STORAGE_CLASSES[predClass].name,
      confidence: maxProb,
      probabilities,
      features,
      isLowConfidence: maxProb < 0.6,
    };
  }

  /**
   * Evaluate model on test dataset and produce complete metrics
   */
  evaluate(XTest: number[][], yTest: StorageClassId[]): EvaluationMetrics {
    const N = XTest.length;
    const K = this.numClasses;

    // 5x5 confusion matrix: [actual][predicted]
    const cm: number[][] = Array.from({ length: K }, () => new Array(K).fill(0));
    let correct = 0;
    let totalLoss = 0;

    for (let i = 0; i < N; i++) {
      const logits = this.predictLogits(XTest[i]);
      const probs = XGBoostClassifier.softmax(logits);
      const actual = yTest[i];

      totalLoss -= Math.log(Math.max(probs[actual], 1e-15));

      let bestClass: StorageClassId = 0;
      let maxP = -1;
      for (let k = 0; k < K; k++) {
        if (probs[k] > maxP) {
          maxP = probs[k];
          bestClass = k as StorageClassId;
        }
      }

      cm[actual][bestClass]++;
      if (bestClass === actual) {
        correct++;
      }
    }

    const accuracy = correct / N;
    const testLoss = totalLoss / N;

    // Compute per-class precision, recall, F1
    const classMetrics: Record<
      StorageClassId,
      { precision: number; recall: number; f1: number; support: number }
    > = {} as any;

    let sumF1 = 0;

    for (let k = 0; k < K; k++) {
      const classId = k as StorageClassId;
      const tp = cm[k][k];
      let rowSum = 0; // actual count (support)
      let colSum = 0; // predicted count

      for (let j = 0; j < K; j++) {
        rowSum += cm[k][j];
        colSum += cm[j][k];
      }

      const precision = colSum > 0 ? tp / colSum : 0;
      const recall = rowSum > 0 ? tp / rowSum : 0;
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

      classMetrics[classId] = {
        precision: Number(precision.toFixed(4)),
        recall: Number(recall.toFixed(4)),
        f1: Number(f1.toFixed(4)),
        support: rowSum,
      };

      sumF1 += f1;
    }

    const macroF1 = Number((sumF1 / K).toFixed(4));

    // Normalize feature importance to 0..100
    let totalGain = 0;
    for (const name of this.featureNames) {
      totalGain += this.featureGainSum[name] || 0;
    }

    const featureImportance: Record<keyof ExtractedFeatures, number> = {} as any;
    for (const name of this.featureNames) {
      const gain = this.featureGainSum[name] || 0;
      featureImportance[name] = totalGain > 0 ? Number(((gain / totalGain) * 100).toFixed(2)) : 0;
    }

    return {
      accuracy: Number(accuracy.toFixed(4)),
      macroF1,
      classMetrics,
      confusionMatrix: cm,
      featureImportance,
      testLoss: Number(testLoss.toFixed(4)),
      sampleCount: {
        train: 0,
        test: N,
      },
    };
  }

  /**
   * Serialize model state
   */
  toJSON(): SerializedModel {
    return {
      hyperparameters: this.hyperparameters,
      rounds: this.rounds,
      numClasses: this.numClasses,
      featureNames: this.featureNames,
      featureGainSum: this.featureGainSum,
    };
  }

  /**
   * Load serialized model state
   */
  static fromJSON(data: SerializedModel): XGBoostClassifier {
    const model = new XGBoostClassifier(data.hyperparameters);
    model.rounds = data.rounds;
    model.numClasses = data.numClasses;
    model.featureNames = data.featureNames;
    model.featureGainSum = data.featureGainSum;
    return model;
  }
}
