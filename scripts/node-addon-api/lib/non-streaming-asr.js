const addon = require('./addon.js');

class OfflineStream {
  constructor(handle) {
    this.handle = handle;
  }

  // obj is {samples: samples, sampleRate: sampleRate}
  // samples is a float32 array containing samples in the range [-1, 1]
  // sampleRate is a number
  acceptWaveform(obj) {
    addon.acceptWaveformOffline(this.handle, obj)
  }

  acceptWaveformAsync(obj) {
    return new Promise((resolve, reject) => {
      const result = addon.acceptWaveformOfflineAsync(this.handle, obj)
      resolve(result);
    });
  }
}

class OfflineRecognizer {
  constructor(handle) {
    this.handle = handle;
  }

  static async createAsync(config) {
    const handle = await addon.createOfflineRecognizerAsync(config);
    return new OfflineRecognizer(handle);
  }
  static create(config) {
    const handle = addon.createOfflineRecognizer(config);
    return new OfflineRecognizer(handle);
  }
  createStream() {
    const handle = addon.createOfflineStream(this.handle);
    return new OfflineStream(handle);
  }

  setConfig(config) {
    addon.offlineRecognizerSetConfig(this.handle, config);
  }

  decode(stream) {
    addon.decodeOfflineStream(this.handle, stream.handle);
  }

  decodeAsync(stream) {
    return new Promise((resolve, reject) => {
      const result = addon.decodeOfflineStreamAsync(this.handle, stream.handle);
      resolve(result);
    });
  }

  getResult(stream) {
    const jsonStr = addon.getOfflineStreamResultAsJson(stream.handle);

    return JSON.parse(jsonStr);
  }
}

module.exports = {
  OfflineRecognizer,
  OfflineStream,
}
