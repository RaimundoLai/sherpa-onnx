const addon = require('./addon.js');

class OfflineSpeechDenoiser {
  constructor(config) {
    if (typeof config === 'object' && config !== null && config._handle) {
      this.handle = config._handle;
      this.config = config._config || {};
    } else {
      this.handle = addon.createOfflineSpeechDenoiser(config);
      this.config = config;
    }

    this.sampleRate =
        addon.offlineSpeechDenoiserGetSampleRateWrapper(this.handle);
  }

  static async createAsync(config) {
    const handle = await addon.createOfflineSpeechDenoiserAsync(config);
    return new OfflineSpeechDenoiser({_handle: handle, _config: config});
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
