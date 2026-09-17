'use strict';

/** @typedef {import('./types').JoyVASAConfig} JoyVASAConfig */

const fs = require('fs');
const path = require('path');
const {FasterLivePortrait} = require('./faster-live-portrait.js');

function isTypedArray(value) {
  return value && ArrayBuffer.isView(value) && !(value instanceof DataView);
}

function flatten(value, name) {
  if (typeof value === 'number') return [value];
  if (isTypedArray(value)) return Array.from(value, Number);
  if (Array.isArray(value)) {
    const result = [];
    for (const item of value) result.push(...flatten(item, name));
    return result;
  }
  throw new TypeError(`${name} must be an Array or TypedArray`);
}

function finiteFloat32(value, name, expectedLength) {
  const flat = flatten(value, name);
  if (expectedLength !== undefined && flat.length !== expectedLength) {
    throw new RangeError(`${name} must contain ${expectedLength} values`);
  }
  const result = new Float32Array(flat.length);
  for (let i = 0; i < flat.length; ++i) {
    if (!Number.isFinite(flat[i])) throw new TypeError(`${name} must contain finite numbers`);
    result[i] = flat[i];
  }
  return result;
}

function readJson(value) {
  if (!value) return {};
  if (typeof value === 'object') return value;
  if (typeof value !== 'string') throw new TypeError('metadata must be a path or object');
  return JSON.parse(fs.readFileSync(value, 'utf8'));
}

function resolveModels(config, metadata, metadataPath) {
  if (config.models) return config.models;
  if (!metadata.models) throw new TypeError('JoyVASA requires models or metadata.models');
  const base = metadataPath ? path.dirname(path.resolve(metadataPath)) : process.cwd();
  const result = {};
  for (const [name, modelPath] of Object.entries(metadata.models)) {
    result[name] = path.isAbsolute(modelPath) ? modelPath : path.join(base, modelPath);
  }
  return result;
}

function getModel(info, name) {
  const model = info.models.find(item => item.name === name);
  if (!model) throw new Error(`JoyVASA model '${name}' is not present in the model set`);
  return model;
}

function getInput(model, aliases) {
  const names = new Set(aliases);
  const input = model.inputs.find(item => names.has(item.name));
  if (!input) throw new Error(`JoyVASA model '${model.name}' is missing input ${aliases.join('/')}`);
  return input;
}

function getOutput(result, name) {
  if (!Array.isArray(result) || result.length === 0) throw new Error('JoyVASA ONNX graph returned no outputs');
  return name ? result.find(item => item.name === name) || (() => {
    throw new Error(`JoyVASA output '${name}' was not found`);
  })() : result[0];
}

function normalRandom(length) {
  const result = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const u1 = Math.max(Number.MIN_VALUE, Math.random());
    const u2 = Math.random();
    const radius = Math.sqrt(-2 * Math.log(u1));
    const angle = 2 * Math.PI * u2;
    result[i] = radius * Math.cos(angle);
    if (i + 1 < length) result[i + 1] = radius * Math.sin(angle);
  }
  return result;
}

function cosineSchedule(numSteps, mode) {
  if (mode !== 'cosine') throw new Error(`JoyVASA currently supports only cosine schedule, got '${mode}'`);
  const betas = new Float64Array(numSteps + 1);
  const alphaBars = new Float64Array(numSteps + 1);
  alphaBars[0] = 1;
  const s = 0.008;
  for (let i = 1; i <= numSteps; ++i) {
    const x0 = ((i - 1) / numSteps + s) / (1 + s) * Math.PI * 0.5;
    const x1 = (i / numSteps + s) / (1 + s) * Math.PI * 0.5;
    const a0 = Math.cos(x0) ** 2;
    const a1 = Math.cos(x1) ** 2;
    betas[i] = Math.min(0.999, Math.max(0.0001, 1 - a1 / a0));
    alphaBars[i] = alphaBars[i - 1] * (1 - betas[i]);
  }
  const alphas = new Float64Array(numSteps + 1);
  const sigmasFlex = new Float64Array(numSteps + 1);
  const sigmasInflex = new Float64Array(numSteps + 1);
  for (let i = 1; i <= numSteps; ++i) {
    alphas[i] = 1 - betas[i];
    sigmasFlex[i] = Math.sqrt(betas[i]);
    sigmasInflex[i] = Math.sqrt(((1 - alphaBars[i - 1]) / (1 - alphaBars[i])) * betas[i]);
  }
  return {alphas, alphaBars, sigmasFlex, sigmasInflex};
}

function resampleLinear(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples;
  const length = Math.max(1, Math.round(samples.length * toRate / fromRate));
  const result = new Float32Array(length);
  const scale = samples.length / length;
  for (let i = 0; i < length; ++i) {
    const position = i * scale;
    const left = Math.floor(position);
    const right = Math.min(samples.length - 1, left + 1);
    const fraction = position - left;
    result[i] = samples[left] * (1 - fraction) + samples[right] * fraction;
  }
  return result;
}

function tail(data, count, width) {
  return new Float32Array(data.slice(data.length - count * width));
}

/**
 * JoyVASA audio-to-motion inference over exported ONNX subgraphs.
 *
 * The exporter emits two graphs. `sample()` performs the checkpoint-configured
 * DDPM sampler in JavaScript and calls the motion-generator graph once per
 * step. This class produces motion coefficients; rendering those coefficients
 * with FasterLivePortrait remains a separate stage.
 */
class JoyVASA {
  /** @param {JoyVASAConfig|Object} config */
  constructor(config) {
    if (!config || typeof config !== 'object') throw new TypeError('JoyVASA requires a config object');
    const metadataPath = typeof config.metadata === 'string' ? config.metadata : undefined;
    const metadata = readJson(config.metadata);
    const motion = metadata.motion || {};
    const audioInput = metadata.audioInput || {};
    const initial = metadata.initialState || {};
    const pick = (key, fallback) => config[key] === undefined ? (motion[key] === undefined ? fallback : motion[key]) : config[key];

    const models = config.runtime ? undefined : resolveModels(config, metadata, metadataPath);
    this.runtime = config.runtime || new FasterLivePortrait({
      models,
      provider: config.provider || 'cpu',
      numThreads: config.numThreads || 1,
      debug: config.debug === true,
    });
    this.audioEncoderModel = config.audioEncoderModel || 'audioEncoder';
    this.motionGeneratorModel = config.motionGeneratorModel || 'motionGenerator';
    this.fps = Number(pick('fps', audioInput.fps || 25));
    this.nMotions = Number(pick('nMotions', 100));
    this.nPrevMotions = Number(pick('nPrevMotions', 10));
    this.motionFeatDim = Number(pick('motionFeatDim', 76));
    this.featureDim = Number(pick('featureDim', 512));
    this.nDiffSteps = Number(pick('nDiffSteps', 500));
    this.diffSchedule = pick('diffSchedule', 'cosine');
    this.target = pick('target', 'sample');
    this.cfgMode = pick('cfgMode', 'incremental');
    this.guidingConditions = config.guidingConditions === undefined
      ? String(motion.guidingConditions === undefined ? 'audio,' : motion.guidingConditions)
          .split(',').filter(item => item === 'audio')
      : (Array.isArray(config.guidingConditions) ? config.guidingConditions : [config.guidingConditions]);
    this.cfgScale = Number(config.cfgScale === undefined ? 2.8 : config.cfgScale);
    this.sampleRate = Number(audioInput.sampleRate || 16000);
    if (this.sampleRate !== 16000) throw new RangeError('JoyVASA ONNX audio input must use 16000 Hz');
    this.samplesPerWindow = Math.round(this.sampleRate * this.nMotions / this.fps);
    this.useIndicator = config.useIndicator === undefined
      ? Boolean(motion.useIndicator)
      : Boolean(config.useIndicator);

    if (!Number.isInteger(this.fps) || this.fps <= 0 || !Number.isInteger(this.nMotions) || this.nMotions <= 0 ||
        !Number.isInteger(this.nPrevMotions) || this.nPrevMotions < 0 || !Number.isInteger(this.motionFeatDim) ||
        this.motionFeatDim <= 0 || !Number.isInteger(this.featureDim) || this.featureDim <= 0 ||
        !Number.isInteger(this.nDiffSteps) || this.nDiffSteps <= 0) {
      throw new RangeError('Invalid JoyVASA model dimensions or diffusion settings');
    }
    this.schedule = cosineSchedule(this.nDiffSteps, this.diffSchedule);
    const info = this.runtime.getModelInfo();
    this.audioInfo = getModel(info, this.audioEncoderModel);
    this.motionInfo = getModel(info, this.motionGeneratorModel);
    this.audioOutputName = config.audioOutputName;
    this.motionOutputName = config.motionOutputName;
    this.motionInputs = {
      motion: getInput(this.motionInfo, ['motion_feat', 'motion_input']),
      audio: getInput(this.motionInfo, ['audio_feat', 'audio_input']),
      prevMotion: getInput(this.motionInfo, ['prev_motion_feat', 'prev_motion_input']),
      prevAudio: getInput(this.motionInfo, ['prev_audio_feat', 'prev_audio_input']),
      step: getInput(this.motionInfo, ['step', 'time_step', 't']),
      indicator: this.motionInfo.inputs.find(item => ['indicator', 'motion_indicator'].includes(item.name)),
    };
    if (this.useIndicator && !this.motionInputs.indicator) {
      throw new Error('JoyVASA config.useIndicator is true but motion_generator.onnx has no indicator input');
    }
    this.initial = {
      startMotionFeat: finiteFloat32(config.startMotionFeat === undefined ? initial.startMotionFeat || new Float32Array(this.nPrevMotions * this.motionFeatDim) : config.startMotionFeat,
          'startMotionFeat', this.nPrevMotions * this.motionFeatDim),
      startAudioFeat: finiteFloat32(config.startAudioFeat === undefined ? initial.startAudioFeat || new Float32Array(this.nPrevMotions * this.featureDim) : config.startAudioFeat,
          'startAudioFeat', this.nPrevMotions * this.featureDim),
      nullAudioFeat: finiteFloat32(config.nullAudioFeat === undefined ? initial.nullAudioFeat || new Float32Array(this.featureDim) : config.nullAudioFeat,
          'nullAudioFeat', this.featureDim),
    };
  }

  /** @returns {{models:Object[]}} */
  getModelInfo() { return this.runtime.getModelInfo(); }

  _audioWindow(samples, sampleRate) {
    let result = finiteFloat32(samples, 'audio');
    if (sampleRate !== this.sampleRate) result = resampleLinear(result, sampleRate, this.sampleRate);
    if (result.length === 0) throw new RangeError('audio must not be empty');
    if (result.length > this.samplesPerWindow) return result.slice(0, this.samplesPerWindow);
    if (result.length < this.samplesPerWindow) {
      const padded = new Float32Array(this.samplesPerWindow);
      padded.set(result);
      const value = result[result.length - 1];
      for (let i = result.length; i < padded.length; ++i) padded[i] = value;
      return padded;
    }
    return result;
  }

  _floatInput(spec, data, shape) {
    if (spec.type !== 'float32') throw new Error(`JoyVASA expects float32 input '${spec.name}', got ${spec.type}`);
    return {name: spec.name, type: 'float32', shape, data};
  }

  _encodeAudioWindow(samples) {
    const result = this.runtime.run(this.audioEncoderModel, [
      this._floatInput(getInput(this.audioInfo, ['audio', 'input_values', 'input']), samples, [1, samples.length]),
    ]);
    const output = getOutput(result, this.audioOutputName);
    if (output.type !== 'float32') throw new Error(`JoyVASA audio output must be float32, got ${output.type}`);
    const expected = this.nMotions * this.featureDim;
    if (output.data.length !== expected) throw new Error(`JoyVASA audio output must contain ${expected} values`);
    return new Float32Array(output.data);
  }

  async _encodeAudioWindowAsync(samples) {
    const run = typeof this.runtime.runAsync === 'function'
      ? this.runtime.runAsync.bind(this.runtime)
      : (name, inputs) => Promise.resolve(this.runtime.run(name, inputs));
    const result = await run(this.audioEncoderModel, [
      this._floatInput(getInput(this.audioInfo, ['audio', 'input_values', 'input']), samples, [1, samples.length]),
    ]);
    const output = getOutput(result, this.audioOutputName);
    if (output.type !== 'float32') throw new Error(`JoyVASA audio output must be float32, got ${output.type}`);
    const expected = this.nMotions * this.featureDim;
    if (output.data.length !== expected) throw new Error(`JoyVASA audio output must contain ${expected} values`);
    return new Float32Array(output.data);
  }

  /**
   * Encode one 16 kHz audio window. The returned tensor is flattened
   * ``[1, nMotions, featureDim]``.
   */
  encodeAudio(samples, options = {}) {
    return this._encodeAudioWindow(this._audioWindow(samples, options.sampleRate || this.sampleRate));
  }

  /** Async counterpart used by Electron so ONNX inference runs off the JS thread. */
  async encodeAudioAsync(samples, options = {}) {
    return this._encodeAudioWindowAsync(this._audioWindow(samples, options.sampleRate || this.sampleRate));
  }

  _stepTensor(step, batchSize) {
    if (!Number.isInteger(step) || step < 1 || step > this.nDiffSteps) throw new RangeError('Invalid diffusion step');
    if (this.motionInputs.step.type === 'int32') {
      return {name: this.motionInputs.step.name, type: 'int32', shape: [batchSize], data: new Int32Array(batchSize).fill(step)};
    }
    if (this.motionInputs.step.type === 'int64') {
      return {name: this.motionInputs.step.name, type: 'int64', shape: [batchSize], data: new BigInt64Array(batchSize).fill(BigInt(step))};
    }
    throw new Error(`JoyVASA step input must be int32 or int64, got ${this.motionInputs.step.type}`);
  }

  _denoise(motion, audio, prevMotion, prevAudio, step, batchSize = 1, indicator) {
    const inputs = [
      this._floatInput(this.motionInputs.motion, motion, [batchSize, this.nMotions, this.motionFeatDim]),
      this._floatInput(this.motionInputs.audio, audio, [batchSize, this.nMotions, this.featureDim]),
      this._floatInput(this.motionInputs.prevMotion, prevMotion, [batchSize, this.nPrevMotions, this.motionFeatDim]),
      this._floatInput(this.motionInputs.prevAudio, prevAudio, [batchSize, this.nPrevMotions, this.featureDim]),
      this._stepTensor(step, batchSize),
    ];
    if (this.motionInputs.indicator) {
      if (this.motionInputs.indicator.type !== 'float32') throw new Error('JoyVASA indicator must be float32');
      inputs.push(this._floatInput(this.motionInputs.indicator, indicator || new Float32Array(batchSize * this.nMotions),
          [batchSize, this.nMotions]));
    }
    const output = getOutput(this.runtime.run(this.motionGeneratorModel, inputs), this.motionOutputName);
    if (output.type !== 'float32') throw new Error(`JoyVASA motion output must be float32, got ${output.type}`);
    return new Float32Array(output.data);
  }

  async _denoiseAsync(motion, audio, prevMotion, prevAudio, step, batchSize = 1, indicator) {
    const inputs = [
      this._floatInput(this.motionInputs.motion, motion, [batchSize, this.nMotions, this.motionFeatDim]),
      this._floatInput(this.motionInputs.audio, audio, [batchSize, this.nMotions, this.featureDim]),
      this._floatInput(this.motionInputs.prevMotion, prevMotion, [batchSize, this.nPrevMotions, this.motionFeatDim]),
      this._floatInput(this.motionInputs.prevAudio, prevAudio, [batchSize, this.nPrevMotions, this.featureDim]),
      this._stepTensor(step, batchSize),
    ];
    if (this.motionInputs.indicator) {
      if (this.motionInputs.indicator.type !== 'float32') throw new Error('JoyVASA indicator must be float32');
      inputs.push(this._floatInput(this.motionInputs.indicator, indicator || new Float32Array(batchSize * this.nMotions),
        [batchSize, this.nMotions]));
    }
    const run = typeof this.runtime.runAsync === 'function'
      ? this.runtime.runAsync.bind(this.runtime)
      : (name, values) => Promise.resolve(this.runtime.run(name, values));
    const result = getOutput(await run(this.motionGeneratorModel, inputs), this.motionOutputName);
    if (result.type !== 'float32') throw new Error(`JoyVASA motion output must be float32, got ${result.type}`);
    return new Float32Array(result.data);
  }

  _initialState(options) {
    const prevMotion = options.prevMotionFeat === undefined
      ? new Float32Array(this.initial.startMotionFeat)
      : finiteFloat32(options.prevMotionFeat, 'prevMotionFeat', this.nPrevMotions * this.motionFeatDim);
    const prevAudio = options.prevAudioFeat === undefined
      ? new Float32Array(this.initial.startAudioFeat)
      : finiteFloat32(options.prevAudioFeat, 'prevAudioFeat', this.nPrevMotions * this.featureDim);
    return {prevMotion, prevAudio};
  }

  /**
   * Run the official diffusion sampler for one window. `audio` may be raw
   * samples or `{features: Float32Array}` produced by `encodeAudio()`.
   */
  sample(audio, options = {}) {
    let audioFeatures;
    if (audio && typeof audio === 'object' && audio.features !== undefined) {
      audioFeatures = finiteFloat32(audio.features, 'audio.features', this.nMotions * this.featureDim);
    } else if (options.audioFeatures !== undefined) {
      audioFeatures = finiteFloat32(options.audioFeatures, 'audioFeatures', this.nMotions * this.featureDim);
    } else {
      audioFeatures = this.encodeAudio(audio, {sampleRate: options.sampleRate || this.sampleRate});
    }
    const state = this._initialState(options);
    const indicator = this.motionInputs.indicator
      ? (options.indicator === undefined
        ? new Float32Array(this.nMotions).fill(1)
        : finiteFloat32(options.indicator, 'indicator', this.nMotions))
      : undefined;
    const useCfg = options.cfg === undefined
      ? this.guidingConditions.includes('audio')
      : Boolean(options.cfg);
    const cfgScale = Number(options.cfgScale === undefined ? this.cfgScale : options.cfgScale);
    if (!Number.isFinite(cfgScale)) throw new TypeError('cfgScale must be finite');
    let motion = options.noise === undefined
      ? normalRandom(this.nMotions * this.motionFeatDim)
      : finiteFloat32(options.noise, 'noise', this.nMotions * this.motionFeatDim);
    const initialNoise = new Float32Array(motion);
    const nullAudio = new Float32Array(this.nMotions * this.featureDim);
    for (let i = 0; i < this.nMotions; ++i) nullAudio.set(this.initial.nullAudioFeat, i * this.featureDim);
    for (let step = this.nDiffSteps; step >= 1; --step) {
      let denoiseMotion;
      let denoiseAudio;
      let denoisePrevMotion;
      let denoisePrevAudio;
      let denoiseIndicator;
      let batchSize;
      if (useCfg) {
        batchSize = 2;
        denoiseMotion = new Float32Array(motion.length * 2);
        denoiseMotion.set(motion);
        denoiseMotion.set(motion, motion.length);
        denoiseAudio = new Float32Array(audioFeatures.length * 2);
        denoiseAudio.set(nullAudio);
        denoiseAudio.set(audioFeatures, audioFeatures.length);
        denoisePrevMotion = new Float32Array(state.prevMotion.length * 2);
        denoisePrevMotion.set(state.prevMotion);
        denoisePrevMotion.set(state.prevMotion, state.prevMotion.length);
        denoisePrevAudio = new Float32Array(state.prevAudio.length * 2);
        denoisePrevAudio.set(state.prevAudio);
        denoisePrevAudio.set(state.prevAudio, state.prevAudio.length);
        if (indicator) {
          denoiseIndicator = new Float32Array(indicator.length * 2);
          denoiseIndicator.set(indicator);
          denoiseIndicator.set(indicator, indicator.length);
        }
      } else {
        batchSize = 1;
        denoiseMotion = motion;
        denoiseAudio = audioFeatures;
        denoisePrevMotion = state.prevMotion;
        denoisePrevAudio = state.prevAudio;
        denoiseIndicator = indicator;
      }
      const result = this._denoise(denoiseMotion, denoiseAudio, denoisePrevMotion,
          denoisePrevAudio, step, batchSize, denoiseIndicator);
      const rowLength = (this.nPrevMotions + this.nMotions) * this.motionFeatDim;
      const currentOffset = this.nPrevMotions * this.motionFeatDim;
      if (result.length !== rowLength * batchSize) throw new Error('Unexpected JoyVASA motion output shape');
      const target = new Float32Array(this.nMotions * this.motionFeatDim);
      for (let i = 0; i < target.length; ++i) {
        const conditional = result[(useCfg ? rowLength : 0) + currentOffset + i];
        const unconditional = result[currentOffset + i];
        target[i] = useCfg ? unconditional + cfgScale * (conditional - unconditional) : conditional;
      }
      const alpha = this.schedule.alphas[step];
      const alphaBar = this.schedule.alphaBars[step];
      const alphaBarPrev = this.schedule.alphaBars[step - 1];
      const sigma = this.schedule.sigmasInflex[step];
      const z = step > 1 ? normalRandom(target.length) : new Float32Array(target.length);
      const next = new Float32Array(target.length);
      if (this.target === 'noise') {
        const c0 = 1 / Math.sqrt(alpha);
        const c1 = (1 - alpha) / Math.sqrt(1 - alphaBar);
        for (let i = 0; i < next.length; ++i) next[i] = c0 * (motion[i] - c1 * target[i]) + sigma * z[i];
      } else if (this.target === 'sample') {
        const c0 = (1 - alphaBarPrev) * Math.sqrt(alpha) / (1 - alphaBar);
        const c1 = (1 - alpha) * Math.sqrt(alphaBarPrev) / (1 - alphaBar);
        for (let i = 0; i < next.length; ++i) next[i] = c0 * motion[i] + c1 * target[i] + sigma * z[i];
      } else {
        throw new Error(`Unsupported JoyVASA target '${this.target}'`);
      }
      motion = next;
    }
    return {
      motion,
      motionShape: [1, this.nMotions, this.motionFeatDim],
      audioFeatures,
      audioFeaturesShape: [1, this.nMotions, this.featureDim],
      initialNoise,
      prevMotionFeat: tail(motion, this.nPrevMotions, this.motionFeatDim),
      prevAudioFeat: tail(audioFeatures, this.nPrevMotions, this.featureDim),
      fps: this.fps,
    };
  }

  /**
   * Async diffusion sampler. Every ONNX graph invocation is awaited; this is
   * intentionally separate from sample() so existing synchronous callers keep
   * their behavior while Electron can remain responsive.
   */
  async sampleAsync(audio, options = {}) {
    let audioFeatures;
    if (audio && typeof audio === 'object' && audio.features !== undefined) {
      audioFeatures = finiteFloat32(audio.features, 'audio.features', this.nMotions * this.featureDim);
    } else if (options.audioFeatures !== undefined) {
      audioFeatures = finiteFloat32(options.audioFeatures, 'audioFeatures', this.nMotions * this.featureDim);
    } else {
      audioFeatures = await this.encodeAudioAsync(audio, {sampleRate: options.sampleRate || this.sampleRate});
    }
    const state = this._initialState(options);
    const indicator = this.motionInputs.indicator
      ? (options.indicator === undefined
        ? new Float32Array(this.nMotions).fill(1)
        : finiteFloat32(options.indicator, 'indicator', this.nMotions))
      : undefined;
    const useCfg = options.cfg === undefined
      ? this.guidingConditions.includes('audio')
      : Boolean(options.cfg);
    const cfgScale = Number(options.cfgScale === undefined ? this.cfgScale : options.cfgScale);
    if (!Number.isFinite(cfgScale)) throw new TypeError('cfgScale must be finite');
    let motion = options.noise === undefined
      ? normalRandom(this.nMotions * this.motionFeatDim)
      : finiteFloat32(options.noise, 'noise', this.nMotions * this.motionFeatDim);
    const initialNoise = new Float32Array(motion);
    const nullAudio = new Float32Array(this.nMotions * this.featureDim);
    for (let i = 0; i < this.nMotions; ++i) nullAudio.set(this.initial.nullAudioFeat, i * this.featureDim);
    for (let step = this.nDiffSteps; step >= 1; --step) {
      let denoiseMotion;
      let denoiseAudio;
      let denoisePrevMotion;
      let denoisePrevAudio;
      let denoiseIndicator;
      let batchSize;
      if (useCfg) {
        batchSize = 2;
        denoiseMotion = new Float32Array(motion.length * 2);
        denoiseMotion.set(motion);
        denoiseMotion.set(motion, motion.length);
        denoiseAudio = new Float32Array(audioFeatures.length * 2);
        denoiseAudio.set(nullAudio);
        denoiseAudio.set(audioFeatures, audioFeatures.length);
        denoisePrevMotion = new Float32Array(state.prevMotion.length * 2);
        denoisePrevMotion.set(state.prevMotion);
        denoisePrevMotion.set(state.prevMotion, state.prevMotion.length);
        denoisePrevAudio = new Float32Array(state.prevAudio.length * 2);
        denoisePrevAudio.set(state.prevAudio);
        denoisePrevAudio.set(state.prevAudio, state.prevAudio.length);
        if (indicator) {
          denoiseIndicator = new Float32Array(indicator.length * 2);
          denoiseIndicator.set(indicator);
          denoiseIndicator.set(indicator, indicator.length);
        }
      } else {
        batchSize = 1;
        denoiseMotion = motion;
        denoiseAudio = audioFeatures;
        denoisePrevMotion = state.prevMotion;
        denoisePrevAudio = state.prevAudio;
        denoiseIndicator = indicator;
      }
      const result = await this._denoiseAsync(denoiseMotion, denoiseAudio, denoisePrevMotion,
        denoisePrevAudio, step, batchSize, denoiseIndicator);
      const rowLength = (this.nPrevMotions + this.nMotions) * this.motionFeatDim;
      const currentOffset = this.nPrevMotions * this.motionFeatDim;
      if (result.length !== rowLength * batchSize) throw new Error('Unexpected JoyVASA motion output shape');
      const target = new Float32Array(this.nMotions * this.motionFeatDim);
      for (let i = 0; i < target.length; ++i) {
        const conditional = result[(useCfg ? rowLength : 0) + currentOffset + i];
        const unconditional = result[currentOffset + i];
        target[i] = useCfg ? unconditional + cfgScale * (conditional - unconditional) : conditional;
      }
      const alpha = this.schedule.alphas[step];
      const alphaBar = this.schedule.alphaBars[step];
      const alphaBarPrev = this.schedule.alphaBars[step - 1];
      const sigma = this.schedule.sigmasInflex[step];
      const z = step > 1 ? normalRandom(target.length) : new Float32Array(target.length);
      const next = new Float32Array(target.length);
      if (this.target === 'noise') {
        const c0 = 1 / Math.sqrt(alpha);
        const c1 = (1 - alpha) / Math.sqrt(1 - alphaBar);
        for (let i = 0; i < next.length; ++i) next[i] = c0 * (motion[i] - c1 * target[i]) + sigma * z[i];
      } else if (this.target === 'sample') {
        const c0 = (1 - alphaBarPrev) * Math.sqrt(alpha) / (1 - alphaBar);
        const c1 = (1 - alpha) * Math.sqrt(alphaBarPrev) / (1 - alphaBar);
        for (let i = 0; i < next.length; ++i) next[i] = c0 * motion[i] + c1 * target[i] + sigma * z[i];
      } else {
        throw new Error(`Unsupported JoyVASA target '${this.target}'`);
      }
      motion = next;
      // Yield even when a caller supplied a synchronous fallback runtime.
      if (step % 8 === 0) await Promise.resolve();
    }
    return {
      motion,
      motionShape: [1, this.nMotions, this.motionFeatDim],
      audioFeatures,
      audioFeaturesShape: [1, this.nMotions, this.featureDim],
      initialNoise,
      prevMotionFeat: tail(motion, this.nPrevMotions, this.motionFeatDim),
      prevAudioFeat: tail(audioFeatures, this.nPrevMotions, this.featureDim),
      fps: this.fps,
    };
  }

  /** Generate motion coefficients for an arbitrarily long 16 kHz clip. */
  generateMotionSequence(audio, options = {}) {
    const samples = finiteFloat32(audio, 'audio');
    const sampleRate = options.sampleRate || this.sampleRate;
    const resampled = sampleRate === this.sampleRate ? samples : resampleLinear(samples, sampleRate, this.sampleRate);
    const windows = Math.max(1, Math.ceil(resampled.length / this.samplesPerWindow));
    const padded = new Float32Array(windows * this.samplesPerWindow);
    padded.set(resampled);
    if (resampled.length > 0) {
      const value = resampled[resampled.length - 1];
      for (let i = resampled.length; i < padded.length; ++i) padded[i] = value;
    }
    let prevMotionFeat;
    let prevAudioFeat;
    let noise;
    const chunks = [];
    for (let i = 0; i < windows; ++i) {
      const result = this.sample(padded.slice(i * this.samplesPerWindow, (i + 1) * this.samplesPerWindow), {
        ...options,
        prevMotionFeat,
        prevAudioFeat,
        noise,
      });
      chunks.push(result.motion);
      prevMotionFeat = result.prevMotionFeat;
      prevAudioFeat = result.prevAudioFeat;
      // FasterLivePortrait's Python pipeline reuses the starting noise for
      // the following overlapping window. Keep that behavior when supplied.
      noise = result.initialNoise;
    }
    const frameCount = Math.max(1, Math.ceil(resampled.length / this.sampleRate * this.fps));
    const output = new Float32Array(frameCount * this.motionFeatDim);
    let offset = 0;
    for (const chunk of chunks) {
      const count = Math.min(chunk.length, output.length - offset);
      output.set(chunk.slice(0, count), offset);
      offset += count;
      if (offset === output.length) break;
    }
    return {motion: output, motionShape: [1, frameCount, this.motionFeatDim], fps: this.fps, frameCount};
  }

  /** Generate motion for an arbitrary clip without blocking the JS thread. */
  async generateMotionSequenceAsync(audio, options = {}) {
    const samples = finiteFloat32(audio, 'audio');
    const sampleRate = options.sampleRate || this.sampleRate;
    const resampled = sampleRate === this.sampleRate ? samples : resampleLinear(samples, sampleRate, this.sampleRate);
    const windows = Math.max(1, Math.ceil(resampled.length / this.samplesPerWindow));
    const padded = new Float32Array(windows * this.samplesPerWindow);
    padded.set(resampled);
    if (resampled.length > 0) {
      const value = resampled[resampled.length - 1];
      for (let i = resampled.length; i < padded.length; ++i) padded[i] = value;
    }
    let prevMotionFeat;
    let prevAudioFeat;
    let noise;
    const chunks = [];
    for (let i = 0; i < windows; ++i) {
      const result = await this.sampleAsync(padded.slice(i * this.samplesPerWindow, (i + 1) * this.samplesPerWindow), {
        ...options,
        prevMotionFeat,
        prevAudioFeat,
        noise,
      });
      chunks.push(result.motion);
      prevMotionFeat = result.prevMotionFeat;
      prevAudioFeat = result.prevAudioFeat;
      noise = result.initialNoise;
    }
    const frameCount = Math.max(1, Math.ceil(resampled.length / this.sampleRate * this.fps));
    const output = new Float32Array(frameCount * this.motionFeatDim);
    let offset = 0;
    for (const chunk of chunks) {
      const count = Math.min(chunk.length, output.length - offset);
      output.set(chunk.slice(0, count), offset);
      offset += count;
      if (offset === output.length) break;
    }
    return {motion: output, motionShape: [1, frameCount, this.motionFeatDim], fps: this.fps, frameCount};
  }
}

module.exports = {JoyVASA};
