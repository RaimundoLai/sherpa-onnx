/** @typedef {import('./types').OfflineStreamObject} OfflineStreamObject */
/** @typedef {import('./types').OfflineStreamHandle} OfflineStreamHandle */
/** @typedef {import('./types').OfflineRecognizerHandle} OfflineRecognizerHandle */
/** @typedef {import('./types').Waveform} Waveform */
/**
 * @typedef {import('./types').OfflineRecognizerConfig} OfflineRecognizerConfig
 */
/**
 * @typedef {import('./types').OfflineRecognizerResult} OfflineRecognizerResult
 */

const addon = require('./addon.js');

/**
 * Internal symbol to mark async-created recognizers.
 * Not accessible unless someone has a reference to this Symbol.
 */
const kFromAsyncFactory = Symbol('OfflineRecognizer.fromAsync');

/**
 * OfflineStream represents a synchronous offline audio stream.
 */
class OfflineStream {
  /**
   * @param {OfflineStreamObject|Object} handle
   */
  constructor(handle) {
    this.handle = handle;
  }

  /**
   * Accept a chunk of waveform samples.
   * @param {Waveform} obj - { samples: Float32Array, sampleRate: number }
   */
  acceptWaveform(obj) {
    addon.acceptWaveformOffline(this.handle, obj);
  }

  /**
   * Set a string option on the underlying offline stream.
   * @param {string} key
   * @param {string} value
   */
  setOption(key, value) {
    addon.offlineStreamSetOption(this.handle, key, value);
  }

  acceptWaveformAsync(obj) {
    return addon.acceptWaveformOfflineAsync(this.handle, obj);
  }

  free() {
    if (this.handle) {
      addon.freeOfflineStream(this.handle);
      this.handle = null;
    }
  }
}

/**
 * OfflineRecognizer wraps the native offline recognizer.
 */
class OfflineRecognizer {
  constructor(configOrInternal) {
    if (configOrInternal && typeof configOrInternal === 'object' &&
        configOrInternal[kFromAsyncFactory]) {
      this.handle = configOrInternal.handle;
      this.config = configOrInternal.config;
      return;
    }
    if (configOrInternal && typeof configOrInternal === 'object' &&
        (configOrInternal.modelConfig !== undefined || configOrInternal.featConfig !== undefined)) {
      this.config = configOrInternal;
      this.handle = addon.createOfflineRecognizer(this.config);
      return;
    }
    this.handle = configOrInternal;
  }

  static async createAsync(config) {
    const handle = await addon.createOfflineRecognizerAsync(config);
    return new OfflineRecognizer({
      [kFromAsyncFactory]: true,
      handle,
      config,
    });
  }

  static create(config) {
    const handle = addon.createOfflineRecognizer(config);
    const r = new OfflineRecognizer(handle);
    r.config = config;
    return r;
  }

  createStream(hotwords) {
    const handle = hotwords === undefined ?
        addon.createOfflineStream(this.handle) :
        addon.createOfflineStream(this.handle, hotwords);
    return new OfflineStream(handle);
  }

  /**
   * Replace the recognizer config at runtime.
   * @param {OfflineRecognizerConfig} config
   */
  setConfig(config) {
    this.config = config;
    addon.offlineRecognizerSetConfig(this.handle, config);
  }

  /**
   * Decode an offline stream (synchronous).
   * @param {OfflineStream} stream
   */
  decode(stream) {
    addon.decodeOfflineStream(this.handle, stream.handle);
  }

  async decodeAsync(stream) {
    const jsonStr =
        await addon.decodeOfflineStreamAsync(this.handle, stream.handle);
    if (typeof jsonStr === 'string') {
      return JSON.parse(jsonStr);
    }
    return jsonStr;
  }
  getResult(stream) {
    const jsonStr = addon.getOfflineStreamResultAsJson(stream.handle);
    return JSON.parse(jsonStr);
  }

  free() {
    if (this.handle) {
      addon.freeOfflineRecognizer(this.handle);
      this.handle = null;
    }
  }
}

module.exports = {
  OfflineRecognizer,
  OfflineStream,
};
