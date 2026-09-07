/** @typedef {import('./types').OfflineTtsConfig} OfflineTtsConfig */
/** @typedef {import('./types').OfflineTtsHandle} OfflineTtsHandle */
/** @typedef {import('./types').TtsRequest} TtsRequest */
/** @typedef {import('./types').GeneratedAudio} GeneratedAudio */

const addon = require('./addon.js');

const kFromAsyncFactory = Symbol('OfflineTts.fromAsync');

class GenerationConfig {
  constructor(opts = {}) {
    Object.assign(this, opts);
  }
}

class OfflineTts {
  constructor(configOrInternal) {
    if (configOrInternal && typeof configOrInternal === 'object' &&
        configOrInternal[kFromAsyncFactory]) {
      this.handle = configOrInternal.handle;
      this.config = configOrInternal.config;
    } else if (configOrInternal && typeof configOrInternal === 'object' &&
               (configOrInternal.model !== undefined || configOrInternal.ruleFsts !== undefined)) {
      this.config = configOrInternal;
      this.handle = addon.createOfflineTts(this.config);
    } else {
      this.handle = configOrInternal;
    }
    this.numSpeakers = addon.getOfflineTtsNumSpeakers(this.handle);
    this.sampleRate = addon.getOfflineTtsSampleRate(this.handle);
  }

  static async createAsync(config) {
    const handle = await addon.createOfflineTtsAsync(config);
    return new OfflineTts({
      [kFromAsyncFactory]: true,
      handle,
      config,
    });
  }

  /**
   * Generate audio synchronously.
   * @param {TtsRequest} obj
   * @returns {GeneratedAudio}
   */
  generate(obj) {
    if (!obj || typeof obj !== 'object') {
      throw new TypeError('generate() expects an object');
    }

    // If generationConfig is present, use new API
    if (obj.generationConfig !== undefined) {
      return addon.offlineTtsGenerateWithConfig(this.handle, obj);
    }

    // Fallback to legacy path
    return addon.offlineTtsGenerate(this.handle, obj);
  }
  generateAsync(obj) {
    const {onProgress, ...rest} = obj;
    const hasConfig = obj.generationConfig !== undefined;
    const fn = hasConfig && addon.offlineTtsGenerateAsyncWithConfig ?
        addon.offlineTtsGenerateAsyncWithConfig :
        addon.offlineTtsGenerateAsync;

    return fn(this.handle, {
      ...rest,
      callback: typeof onProgress === 'function' ?
          (info) => {
            const ret = onProgress(info);
            return ret === 0 || ret === false ? 0 : 1;
          } :
          undefined,
    });
  }

  extractMiocodecEmbeddings(audioDir) {
    return addon.offlineTtsExtractMiocodecEmbeddings(this.handle, audioDir);
  }

  extractMiocodecEmbeddingsAsync(audioDir) {
    return addon.offlineTtsExtractMiocodecEmbeddingsAsync(this.handle, audioDir);
  }

  generateWithMiocodecEmbeddings(obj) {
    return addon.offlineTtsGenerateWithMiocodecEmbeddings(this.handle, obj);
  }

  generateWithMiocodecEmbeddingsAsync(obj) {
    return addon.offlineTtsGenerateWithMiocodecEmbeddingsAsync(this.handle, obj);
  }

  convertVoiceWithMiocodecEmbeddings(obj) {
    return addon.offlineTtsConvertVoiceWithMiocodecEmbeddings(this.handle, obj);
  }

  convertVoiceWithMiocodecEmbeddingsAsync(obj) {
    return addon.offlineTtsConvertVoiceWithMiocodecEmbeddingsAsync(this.handle, obj);
  }
  }
}


module.exports = {
  OfflineTts,
  GenerationConfig,
}
