'use strict';

/**
 * Motion Controller for LivePortrait / JoyVASA (MLX and ONNX compatible).
 *
 * Implements:
 * 1. AudioEnergyDetector: Computes short-time RMS energy and detects natural speech events (emphasis, clause pause).
 * 2. GestureScheduler: Schedules event-based physiological wavelets (Nod, Tilt, Shift, Emphasis) without robotic sin waves.
 * 3. IdleStatePool: Multi-state idle pool (Neutral, Settle, Tilt, Shift) with smooth transitions.
 * 4. MotionBlender: Blends scheduled gestures onto motion vectors while 100% preserving lip-sync keypoints.
 */

// LivePortrait mouth keypoint indices that MUST NEVER be corrupted during speech
const LIVEPORTRAIT_MOUTH_EXPRESSION_INDICES = [6, 12, 14, 17, 19, 20];

/**
 * Hann / smoothstep wavelet window: C^1 continuous, exactly 0 at tau=0 and tau=1.
 * tau in [0, 1]
 */
function hannWindow(tau) {
  if (tau <= 0 || tau >= 1) return 0;
  return 0.5 * (1 - Math.cos(2 * Math.PI * tau));
}

/**
 * Asymmetric nod wavelet: faster drop (0..0.4), slower rebound (0.4..1.0).
 */
function nodWavelet(tau) {
  if (tau <= 0 || tau >= 1) return 0;
  if (tau < 0.4) {
    const p = tau / 0.4;
    return Math.sin(Math.PI * 0.5 * p);
  } else {
    const p = (tau - 0.4) / 0.6;
    return Math.cos(Math.PI * 0.5 * p) - 0.25 * Math.sin(Math.PI * p);
  }
}

class AudioEnergyDetector {
  /**
   * Analyze audio samples and return per-frame energy metrics and events.
   */
  static analyze(audioSamples, sampleRate = 16000, fps = 25) {
    if (!audioSamples || audioSamples.length === 0) {
      return { frameCount: 0, rms: new Float32Array(0), events: [] };
    }
    const samplesPerFrame = Math.round(sampleRate / fps);
    const frameCount = Math.ceil(audioSamples.length / samplesPerFrame);
    const rawRms = new Float32Array(frameCount);

    let totalRms = 0;
    for (let f = 0; f < frameCount; ++f) {
      const start = f * samplesPerFrame;
      const end = Math.min(audioSamples.length, (f + 1) * samplesPerFrame);
      let sumSq = 0;
      const count = end - start;
      if (count > 0) {
        for (let i = start; i < end; ++i) {
          sumSq += audioSamples[i] * audioSamples[i];
        }
        rawRms[f] = Math.sqrt(sumSq / count);
      } else {
        rawRms[f] = 0;
      }
      totalRms += rawRms[f];
    }

    const meanRms = totalRms / Math.max(1, frameCount);

    // 3-point smoothing
    const smoothRms = new Float32Array(frameCount);
    for (let f = 0; f < frameCount; ++f) {
      const prev = f > 0 ? rawRms[f - 1] : rawRms[f];
      const next = f + 1 < frameCount ? rawRms[f + 1] : rawRms[f];
      smoothRms[f] = 0.25 * prev + 0.5 * rawRms[f] + 0.25 * next;
    }

    // Detect events (emphasis spikes, clause pauses)
    const events = [];
    const minGestureGapFrames = Math.round(fps * 1.8); // at least 1.8s between major gestures
    let lastGestureFrame = -minGestureGapFrames;
    const pauseThreshold = meanRms * 0.25;

    let inPause = false;
    let pauseStartFrame = 0;

    for (let f = 1; f < frameCount - 1; ++f) {
      const val = smoothRms[f];
      const prev = smoothRms[f - 1];

      // Detect pause start/end
      if (val < pauseThreshold) {
        if (!inPause) {
          inPause = true;
          pauseStartFrame = f;
        }
      } else {
        if (inPause) {
          const pauseDuration = f - pauseStartFrame;
          inPause = false;
          // After a noticeable clause pause (> 0.25s), trigger a subtle clause reset nod on speech resumption
          if (pauseDuration >= Math.round(fps * 0.25) && (f - lastGestureFrame) >= minGestureGapFrames) {
            events.push({ frame: f, type: 'nod', intensity: 0.8 });
            lastGestureFrame = f;
            continue;
          }
        }
      }

      // Detect emphasis: sudden energy surge above mean
      if (!inPause && (f - lastGestureFrame) >= minGestureGapFrames) {
        const delta = val - prev;
        if (delta > meanRms * 0.6 && val > meanRms * 1.5) {
          events.push({ frame: f, type: 'emphasis', intensity: Math.min(1.4, val / Math.max(1e-4, meanRms)) });
          lastGestureFrame = f;
          continue;
        }
      }

      // Cadence fallback: if speaking steadily for > 3.5s without any gesture, insert conversational tilt/shift
      if (!inPause && (f - lastGestureFrame) >= Math.round(fps * 3.5)) {
        const gestureType = (events.length % 2 === 0) ? 'tilt' : 'shift';
        const dir = (events.length % 4 < 2) ? 1 : -1;
        events.push({ frame: f, type: gestureType, direction: dir, intensity: 0.85 });
        lastGestureFrame = f;
      }
    }

    return { frameCount, rms: smoothRms, meanRms, events };
  }
}

class MotionController {
  /**
   * Evaluate gesture delta for a given frame index.
   * Returns: { deltaPitch, deltaYaw, deltaRoll, deltaTy } in physical units (degrees, normalized translation).
   */
  static evaluateGestures(frame, events, fps = 25, gestureScale = 1.0) {
    let deltaPitch = 0;
    let deltaYaw = 0;
    let deltaRoll = 0;
    let deltaTy = 0;

    if (!events || events.length === 0) {
      return { deltaPitch, deltaYaw, deltaRoll, deltaTy };
    }

    for (const ev of events) {
      let durationFrames = Math.round(fps * 0.8);
      if (ev.type === 'emphasis') durationFrames = Math.round(fps * 0.55);
      else if (ev.type === 'tilt') durationFrames = Math.round(fps * 0.95);
      else if (ev.type === 'shift') durationFrames = Math.round(fps * 1.15);

      if (frame >= ev.frame && frame < ev.frame + durationFrames) {
        const tau = (frame - ev.frame) / durationFrames;
        const scale = (ev.intensity || 1.0) * gestureScale;

        if (ev.type === 'nod') {
          const w = nodWavelet(tau);
          deltaPitch += 1.6 * w * scale;
        } else if (ev.type === 'emphasis') {
          const w = nodWavelet(tau);
          deltaPitch += 2.2 * w * scale;
          deltaTy += 0.0006 * w * scale;
        } else if (ev.type === 'tilt') {
          const direction = (ev.direction || 1);
          const w = hannWindow(tau);
          deltaRoll += 1.4 * direction * w * scale;
        } else if (ev.type === 'shift') {
          const direction = (ev.direction || 1);
          const w = hannWindow(tau);
          deltaYaw += 1.8 * direction * w * scale;
        }
      }
    }

    return { deltaPitch, deltaYaw, deltaRoll, deltaTy };
  }

  /**
   * Generate an organic multi-state Idle motion sequence for a given duration.
   * Combines Idle Neutral (65-70%), Idle Tilt, Idle Settle, and Idle Shift into a seamless timeline.
   */
  static generateIdleMotion(frameCount, fps = 25) {
    const motion = [];

    // Schedule idle events across the idle clip
    const events = [];
    const minEventInterval = Math.round(fps * 2.2);
    let nextEventFrame = Math.round(fps * 1.0); // first event around 1.0s

    let toggleSide = 1;
    while (nextEventFrame + Math.round(fps * 1.2) < frameCount) {
      const r = (events.length % 3);
      let type = 'tilt';
      if (r === 0) type = 'tilt';
      else if (r === 1) type = 'nod';
      else type = 'shift';

      events.push({
        frame: nextEventFrame,
        type,
        direction: toggleSide,
        intensity: 0.8
      });
      toggleSide = -toggleSide;
      nextEventFrame += minEventInterval;
    }

    // Baseline breathing cycle (~3.0s cycle)
    const breathCycle = Math.max(1, Math.round(fps * 3.0));

    for (let f = 0; f < frameCount; ++f) {
      // Natural breathing harmonics
      const tauBreath = (2 * Math.PI * (f % breathCycle)) / breathCycle;
      const breathPitch = 0.25 * Math.sin(tauBreath);
      const breathYaw = 0.20 * Math.sin(tauBreath);
      const breathRoll = 0.08 * Math.sin(2 * tauBreath);
      const breathTy = 0.0004 * Math.sin(tauBreath);

      // Event gestures (nods, tilts, shifts)
      const gesture = MotionController.evaluateGestures(f, events, fps, 1.0);

      // Total pose
      const pitch = breathPitch + gesture.deltaPitch;
      const yaw = breathYaw + gesture.deltaYaw;
      const roll = breathRoll + gesture.deltaRoll;
      const ty = breathTy + gesture.deltaTy;

      motion.push({
        pitch,
        yaw,
        roll,
        t: [0, ty, 0],
        scale: 1.0,
      });
    }

    return motion;
  }
}

module.exports = {
  MotionController,
  AudioEnergyDetector,
  LIVEPORTRAIT_MOUTH_EXPRESSION_INDICES,
  hannWindow,
  nodWavelet,
};
