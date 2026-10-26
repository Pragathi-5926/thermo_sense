import { ExtractedFeatures, StorageClassId, StorageSettings, StorageWindowSample } from '../types/storage';
import { extractFeatures, smoothTemperature } from './signalProcessing';

/**
 * Box-Muller Gaussian random generator
 */
export function randomGaussian(mean: number = 0, std: number = 1): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return mean + z * std;
}

/**
 * Random float between min and max
 */
export function randomBetween(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

/**
 * Generate a 360-minute raw temperature window for a specific class
 */
export function generateWindowForClass(
  classId: StorageClassId,
  settings: StorageSettings
): number[] {
  const n = 360;
  const raw = new Array<number>(n);
  const setpoint = settings.setpoint; // default 5.0 °C

  // Base compressor oscillation parameters
  const cyclePeriod = randomBetween(35, 60); // minutes per compressor duty cycle
  const cycleAmp = randomBetween(0.2, 0.4); // ±0.3 °C
  const baseNoiseSigma = randomBetween(0.05, 0.12);

  // Initialize with normal baseline cycling
  for (let t = 0; t < n; t++) {
    const cycle = cycleAmp * Math.sin((2 * Math.PI * t) / cyclePeriod);
    const noise = randomGaussian(0, baseNoiseSigma);
    raw[t] = setpoint + cycle + noise;
  }

  switch (classId) {
    case 0: {
      // Normal: small oscillation around setpoint from the compressor duty cycle (±0.3 °C)
      // Slight slow drift of ±0.2 °C across hours
      const slowDrift = randomBetween(-0.15, 0.15);
      for (let t = 0; t < n; t++) {
        raw[t] += (t / n) * slowDrift;
      }
      break;
    }

    case 1: {
      // Door left open:
      // Sharp rise of 1.5–4.0 °C within 5–20 minutes,
      // stays elevated 10–40 min, then recovers fairly fast
      const eventStart = Math.floor(randomBetween(40, 160));
      const riseDuration = Math.floor(randomBetween(6, 20)); // min
      const holdDuration = Math.floor(randomBetween(12, 40)); // min
      const peakDelta = randomBetween(1.8, 4.2); // °C rise
      const recoveryHalfLife = randomBetween(15, 35); // min pull down

      const riseEnd = eventStart + riseDuration;
      const holdEnd = riseEnd + holdDuration;

      for (let t = eventStart; t < n; t++) {
        if (t <= riseEnd) {
          // Sharp sigmoidal or linear rise
          const progress = (t - eventStart) / riseDuration;
          raw[t] += peakDelta * Math.sin((progress * Math.PI) / 2);
        } else if (t <= holdEnd) {
          // Elevated plateau with slight air turbulence noise
          const turbulence = randomGaussian(0, 0.15);
          raw[t] += peakDelta + turbulence;
        } else {
          // Pull-down recovery exponential curve
          const dt = t - holdEnd;
          const remainingHeat = peakDelta * Math.exp(-dt / recoveryHalfLife);
          raw[t] += remainingHeat;
        }
      }
      break;
    }

    case 2: {
      // Compressor / cooling failure:
      // Slow steady rise (0.3–1.2 °C/hour) that continues for hours with little or no recovery;
      // may show weakened cycling beforehand
      const failureStart = Math.floor(randomBetween(20, 120));
      const ratePerHour = randomBetween(0.45, 1.25); // °C / hr
      const ratePerMin = ratePerHour / 60;
      const ambientLimit = randomBetween(13.0, 18.0); // ambient asymptotic ceiling

      for (let t = failureStart; t < n; t++) {
        const elapsed = t - failureStart;
        // As compressor fails, cyclic oscillation dampens out
        const damping = Math.max(0, 1 - elapsed / 60);
        raw[t] -= damping * 0.2 * Math.sin((2 * Math.PI * t) / cyclePeriod);

        // Monotonic thermal rise towards ambient ceiling
        const thermalRise = (ambientLimit - setpoint) * (1 - Math.exp(-(ratePerMin * elapsed) / (ambientLimit - setpoint)));
        raw[t] += thermalRise;
      }
      break;
    }

    case 3: {
      // Ambient temperature change:
      // Gradual, smooth rise (0.1–0.5 °C/hour) following an outside-temperature ramp,
      // slow partial recovery, low short-term volatility
      const rampStart = Math.floor(randomBetween(30, 90));
      const peakTime = rampStart + Math.floor(randomBetween(120, 180));
      const maxDelta = randomBetween(1.2, 2.6); // smooth excursion 6.2 - 7.6 °C

      for (let t = 0; t < n; t++) {
        if (t >= rampStart && t <= peakTime) {
          // Smooth half-cosine rise
          const frac = (t - rampStart) / (peakTime - rampStart);
          raw[t] += maxDelta * 0.5 * (1 - Math.cos(frac * Math.PI));
        } else if (t > peakTime) {
          // Very slow partial recovery
          const frac = (t - peakTime) / (n - peakTime);
          const partialRecover = maxDelta * 0.35 * Math.sin((frac * Math.PI) / 2);
          raw[t] += Math.max(0.3, maxDelta - partialRecover);
        }
      }
      break;
    }

    case 4: {
      // Sensor malfunction:
      // Physically implausible behavior: random spikes, sudden jumps that revert instantly,
      // dropouts/flatline (stuck values), or excessive noise
      const failureType = Math.floor(Math.random() * 4);

      if (failureType === 0) {
        // Random extreme spikes (> 1 °C to 8 °C in 1 minute, reverting immediately)
        const spikeCount = Math.floor(randomBetween(5, 15));
        for (let s = 0; s < spikeCount; s++) {
          const pos = Math.floor(randomBetween(10, n - 10));
          const spikeMag = randomBetween(2.0, 7.5) * (Math.random() > 0.3 ? 1 : -1);
          raw[pos] += spikeMag;
          if (Math.random() > 0.5 && pos + 1 < n) {
            raw[pos + 1] += spikeMag * 0.7; // 2-min glitch
          }
        }
      } else if (failureType === 1) {
        // Sudden step jumps that revert abruptly or float
        const jumpStart = Math.floor(randomBetween(50, 150));
        const jumpDuration = Math.floor(randomBetween(15, 60));
        const jumpDelta = randomBetween(3.0, 8.0) * (Math.random() > 0.3 ? 1 : -1);
        for (let t = jumpStart; t < Math.min(n, jumpStart + jumpDuration); t++) {
          raw[t] += jumpDelta;
        }
      } else if (failureType === 2) {
        // Flatline / stuck telemetry (e.g. sensor ADC freezes at exact float value)
        const flatStart = Math.floor(randomBetween(60, 180));
        const flatDuration = Math.floor(randomBetween(90, 200));
        const stuckValue = Number((setpoint + randomBetween(-1.5, 3.5)).toFixed(2));
        for (let t = flatStart; t < Math.min(n, flatStart + flatDuration); t++) {
          raw[t] = stuckValue;
        }
      } else {
        // Severe telemetry noise & ADC dropout jitter (sigma = 1.5 - 3.0 °C)
        const noisyStart = Math.floor(randomBetween(40, 120));
        for (let t = noisyStart; t < n; t++) {
          raw[t] += randomGaussian(0, randomBetween(1.2, 2.8));
          if (Math.random() < 0.05) {
            raw[t] = 0; // sensor communication drop / zero-wire fault
          }
        }
      }
      break;
    }
  }

  // Format points to 2 decimal places to mimic sensor telemetry
  return raw.map((v) => Number(v.toFixed(2)));
}

/**
 * Generate full synthetic dataset with balanced classes
 */
export function generateSyntheticDataset(
  totalSamples: number = 2000,
  settings: StorageSettings
): StorageWindowSample[] {
  const samples: StorageWindowSample[] = [];
  const samplesPerClass = Math.floor(totalSamples / 5);

  let idCounter = 1;
  const classes: StorageClassId[] = [0, 1, 2, 3, 4];

  for (const c of classes) {
    const count = c === 4 ? totalSamples - samplesPerClass * 4 : samplesPerClass;
    for (let i = 0; i < count; i++) {
      const rawPoints = generateWindowForClass(c, settings);
      const smoothed = smoothTemperature(rawPoints, settings.medianWindow, settings.maWindow);
      const features = extractFeatures(rawPoints, smoothed, settings.threshold);

      samples.push({
        id: `SAMPLE-${String(idCounter++).padStart(5, '0')}`,
        classId: c,
        points: rawPoints,
        features,
      });
    }
  }

  // Shuffle dataset (Fisher-Yates)
  for (let i = samples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [samples[i], samples[j]] = [samples[j], samples[i]];
  }

  return samples;
}

/**
 * Convert feature dataset to CSV string
 */
export function exportFeaturesToCSV(samples: StorageWindowSample[]): string {
  const headers = ['id', 'class_id', 'class_name', ...Object.keys(samples[0]?.features || {})];
  const rows: string[] = [headers.join(',')];

  const classNames: Record<StorageClassId, string> = {
    0: 'Normal',
    1: 'Door Left Open',
    2: 'Compressor Failure',
    3: 'Ambient Change',
    4: 'Sensor Malfunction',
  };

  for (const s of samples) {
    const f = s.features;
    const row = [
      s.id,
      s.classId,
      `"${classNames[s.classId]}"`,
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
    rows.push(row.join(','));
  }

  return rows.join('\n');
}
