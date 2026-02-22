const addon = require('./addon.js');

class OfflineSpeechDenoiser {
  constructor(handle) {
    this.handle = handle;
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

  /*
    obj is
    {samples: samples, sampleRate: sampleRate, enableExternalBuffer: true}

    samples is a float32 array containing samples in the range [-1, 1]
    sampleRate is a number

   return an object {samples: Float32Array, sampleRate: <a number>}
   */
  run(obj) {
    return addon.offlineSpeechDenoiserRunWrapper(this.handle, obj);
  }
}

module.exports = {
  OfflineSpeechDenoiser,
}
