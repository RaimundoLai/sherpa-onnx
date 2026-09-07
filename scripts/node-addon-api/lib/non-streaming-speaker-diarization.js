/** @typedef {import('./types').OfflineSpeakerDiarizationConfig} OfflineSpeakerDiarizationConfig */
/** @typedef {import('./types').OfflineSpeakerDiarizationHandle} OfflineSpeakerDiarizationHandle */
/** @typedef {import('./types').SpeakerDiarizationSegment} SpeakerDiarizationSegment */

const addon = require('./addon.js');

class OfflineSpeakerDiarization {
  constructor(configOrHandle) {
    if (configOrHandle && typeof configOrHandle === 'object' && configOrHandle.clustering !== undefined) {
      this.config = configOrHandle;
      this.handle = addon.createOfflineSpeakerDiarization(configOrHandle);
    } else {
      this.handle = configOrHandle;
    }
    this.sampleRate = addon.getOfflineSpeakerDiarizationSampleRate(this.handle);
  }

  static async createAsync(config) {
    const handle = await addon.createOfflineSpeakerDiarizationAsync(config);
    const sd = new OfflineSpeakerDiarization(handle);
    sd.config = config;
    return sd;
  }
  static create(config) {
    const handle = addon.createOfflineSpeakerDiarization(config);
    const sd = new OfflineSpeakerDiarization(handle);
    sd.config = config;
    return sd;
  }
  /**
   * @param {Float32Array} samples - 1-D float32 array in [-1, 1]
   * @returns {SpeakerDiarizationSegment[]}
   */
  process(samples) {
    return addon.offlineSpeakerDiarizationProcess(this.handle, samples);
  }

  processAsync(samples, callable) {
    if (typeof callable !== 'function') {
      callable = () => {};
    }
    return addon.offlineSpeakerDiarizationProcessAsync(this.handle, samples, callable);
  }
  setConfig(config) {
    addon.offlineSpeakerDiarizationSetConfig(this.handle, config);
    this.config.clustering = config.clustering;
  }
}

module.exports = {
  OfflineSpeakerDiarization,
}
