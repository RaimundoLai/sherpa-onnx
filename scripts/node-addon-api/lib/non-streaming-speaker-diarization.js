const addon = require('./addon.js');

class OfflineSpeakerDiarization {
  constructor(handle) {
    this.handle = handle;
    this.sampleRate = addon.getOfflineSpeakerDiarizationSampleRate(this.handle);
  }

  static async createAsync(config) {
    const handle = await addon.createOfflineSpeakerDiarizationAsync(config);
    const sd = new OfflineSpeakerDiarization(handle);
    sd.config = config;
    return sd;
  }

  /**
   * samples is a 1-d float32 array. Each element of the array should be
   * in the range [-1, 1].
   *
   * We assume its sample rate equals to this.sampleRate.
   *
   * Returns an array of object, where an object is
   *
   *  {
   *    "start": start_time_in_seconds,
   *    "end": end_time_in_seconds,
   *    "speaker": an_integer,
   *  }
   */
  process(samples) {
    return addon.offlineSpeakerDiarizationProcess(this.handle, samples);
  }

  processAsync(samples, callable) {
    return new Promise((resolve, reject) => {
      const result = addon.offlineSpeakerDiarizationProcessAsync(this.handle, samples, callable);
      resolve(result);
    });
  }
  setConfig(config) {
    addon.offlineSpeakerDiarizationSetConfig(this.handle, config);
    this.config.clustering = config.clustering;
  }
}

module.exports = {
  OfflineSpeakerDiarization,
}
