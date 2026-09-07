/** @typedef {import('./types').OfflineSpeechDenoiserConfig} OfflineSpeechDenoiserConfig */
/** @typedef {import('./types').OfflineSpeechDenoiserHandle} OfflineSpeechDenoiserHandle */
/** @typedef {import('./types').GeneratedAudio} GeneratedAudio */
/** @typedef {import('./types').AudioProcessRequest} AudioProcessRequest */

const addon = require('./addon.js');

class OfflineSpeechDenoiser {
  constructor(configOrHandle) {
    if (configOrHandle && typeof configOrHandle === 'object' &&
        (configOrHandle.model !== undefined || configOrHandle.dpdfnet !== undefined || configOrHandle.gtcrn !== undefined)) {
      this.config = configOrHandle;
      this.handle = addon.createOfflineSpeechDenoiser(configOrHandle);
    } else {
      this.handle = configOrHandle;
    }
    this.sampleRate = addon.offlineSpeechDenoiserGetSampleRateWrapper(this.handle);
  }

  static async createAsync(config) {
    const handle = await addon.createOfflineSpeechDenoiserAsync(config);
    return new OfflineSpeechDenoiser(handle);
  }
  static create(config) {
    const handle = addon.createOfflineSpeechDenoiser(config);
    return new OfflineSpeechDenoiser(handle);
  }

  /**
   * Run denoiser synchronously.
   * @param {AudioProcessRequest} obj - { samples: Float32Array, sampleRate: number, enableExternalBuffer?: boolean }
   * @returns {GeneratedAudio}
   */
  run(obj) {
    return addon.offlineSpeechDenoiserRunWrapper(this.handle, obj);
  }
}

module.exports = {
  OfflineSpeechDenoiser,
}
