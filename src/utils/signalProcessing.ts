import { ExtractedFeatures } from '../types/storage';

/**
 * Moving median filter for 1D temperature series
 */
export function movingMedian(data: number[], windowSize: number): number[] {
  const n = data.length;
  const half = Math.floor(windowSize / 2);
  const result = new Array<number>(n);

  for (let i = 0; i < n; i++) {
    const start = Math.max(0, i - half);
    const end = Math.min(n, i + half + 1);
    const window = data.slice(start, end).sort((a, b) => a - b);
    const mid = Math.floor(window.length / 2);
    if (window.length % 2 === 1) {
      result[i] = window[mid];
    } else {
      result[i] = (window[mid - 1] + window[mid]) / 2;
    }
  }

  return result;
}

/**
 * Moving average filter for 1D temperature series
 */
export function movingAverage(data: number[], windowSize: number): number[] {
  const n = data.length;
  const half = Math.floor(windowSize / 2);
  const result = new Array<number>(n);

  for (let i = 0; i < n; i++) {
    const start = Math.max(0, i - half);
    const end = Math.min(n, i + half + 1);
    let sum = 0;
    for (let j = start; j < end; j++) {
      sum += data[j];
    }
    result[i] = sum / (end - start);
  }

  return result;
}

/**
 * Preprocessing pipeline: Moving Median followed by Moving Average
 */
export function smoothTemperature(
  raw: number[],
  medianWindow: number = 5,
  maWindow: number = 7
): number[] {
  const medianFiltered = movingMedian(raw, Math.max(3, medianWindow | 1)); // ensure odd
  const smoothed = movingAverage(medianFiltered, Math.max(1, maWindow));
  return smoothed;
}

/**
 * Feature Extraction from 6-hour (360-minute) temperature window
 */
export function extractFeatures(
  raw: number[],
  smoothed: number[],
  threshold: number
): ExtractedFeatures {
  const n = raw.length;
  if (n === 0) {
    return {
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
  }

  // 1. Mean & Peak
  let sum = 0;
  let peak = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = raw[i];
    sum += v;
    if (v > peak) peak = v;
  }
  const mean_temp = sum / n;

  // 2. Std Dev
  let varianceSum = 0;
  for (let i = 0; i < n; i++) {
    const diff = raw[i] - mean_temp;
    varianceSum += diff * diff;
  }
  const std_temp = Math.sqrt(varianceSum / n);

  // 3. Excursion duration (based on smoothed signal to filter transient noise spikes)
  let duration_above_thresh = 0;
  let firstExcursionIdx = -1;
  let peakIdx = -1;
  let maxSmoothTemp = -Infinity;

  for (let i = 0; i < n; i++) {
    if (smoothed[i] > threshold) {
      duration_above_thresh++;
      if (firstExcursionIdx === -1) {
        firstExcursionIdx = i;
      }
    }
    if (smoothed[i] > maxSmoothTemp) {
      maxSmoothTemp = smoothed[i];
      peakIdx = i;
    }
  }

  // 4. Rate of rise & Max Slope
  // Computed from smoothed signal so noise doesn't corrupt physical rate of rise
  let positiveDeltasSum = 0;
  let positiveCount = 0;
  let max_slope = 0;

  for (let i = 1; i < n; i++) {
    const delta = smoothed[i] - smoothed[i - 1];
    if (delta > 0) {
      positiveDeltasSum += delta;
      positiveCount++;
      if (delta > max_slope) {
        max_slope = delta;
      }
    }
  }
  const mean_rate_rise = positiveCount > 0 ? positiveDeltasSum / positiveCount : 0;

  // 5. Recovery Time (min)
  // Measured from when peak occurs until temperature drops back below threshold
  let recovery_time = 0;
  if (duration_above_thresh > 0) {
    let recovered = false;
    const searchStart = peakIdx >= 0 ? peakIdx : firstExcursionIdx;
    for (let i = searchStart; i < n; i++) {
      if (smoothed[i] <= threshold) {
        recovery_time = i - searchStart;
        recovered = true;
        break;
      }
    }
    if (!recovered) {
      // Window length (or remaining window from excursion start)
      recovery_time = n - firstExcursionIdx;
    }
  } else {
    recovery_time = 0;
  }

  // 6. Number of Spikes (|raw[i] - raw[i-1]| > 1.0 °C)
  let num_spikes = 0;
  let flatline_count = 0;
  for (let i = 1; i < n; i++) {
    const diff = Math.abs(raw[i] - raw[i - 1]);
    if (diff > 1.0) {
      num_spikes++;
    }
    if (diff < 0.0001) {
      flatline_count++;
    }
  }

  // 7. Flatline fraction (share of consecutive identical readings)
  const flatline_fraction = n > 1 ? flatline_count / (n - 1) : 0;

  return {
    mean_temp: Number(mean_temp.toFixed(4)),
    std_temp: Number(std_temp.toFixed(4)),
    mean_rate_rise: Number(mean_rate_rise.toFixed(4)),
    duration_above_thresh,
    max_slope: Number(max_slope.toFixed(4)),
    recovery_time,
    peak_temp: Number(peak.toFixed(4)),
    num_spikes,
    flatline_fraction: Number(flatline_fraction.toFixed(4)),
  };
}
