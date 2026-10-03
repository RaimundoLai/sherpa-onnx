/** @typedef {import('./types').OnlineSpeechDenoiserConfig} OnlineSpeechDenoiserConfig */
/** @typedef {import('./types').OnlineSpeechDenoiserHandle} OnlineSpeechDenoiserHandle */
/** @typedef {import('./types').GeneratedAudio} GeneratedAudio */
/** @typedef {import('./types').AudioProcessRequest} AudioProcessRequest */

const addon = require('./addon.js');

class OnlineSpeechDenoiser {
  /**
   * @param {OnlineSpeechDenoiserConfig|OnlineSpeechDenoiserHandle} configOrHandle
   */
  constructor(configOrHandle) {
    if (configOrHandle && typeof configOrHandle === 'object' &&
        (configOrHandle.model !== undefined || configOrHandle.dpdfnet !== undefined || configOrHandle.gtcrn !== undefined)) {
      this.config = configOrHandle;
      this.handle = addon.createOnlineSpeechDenoiser(configOrHandle);
    } else {
      this.handle = configOrHandle;
    }

    this.sampleRate = addon.onlineSpeechDenoiserGetSampleRateWrapper(this.handle);
    this.frameShiftInSamples =
        addon.onlineSpeechDenoiserGetFrameShiftInSamplesWrapper(this.handle);
    this.operationQueue = Promise.resolve();
  }

  /**
   * Create a denoiser without blocking Node.js while ONNX Runtime loads the model.
   * @param {OnlineSpeechDenoiserConfig} config
   * @returns {Promise<OnlineSpeechDenoiser>}
   */
  static async createAsync(config) {
    const handle = await addon.createOnlineSpeechDenoiserAsync(config);
    return new OnlineSpeechDenoiser(handle);
  }

  /**
   * Serialize operations because the underlying streaming denoiser is stateful.
   * @template T
   * @param {() => Promise<T>} operation
   * @returns {Promise<T>}
   */
  enqueue(operation) {
    const result = this.operationQueue.then(operation, operation);
    this.operationQueue = result.then(() => undefined, () => undefined);
    return result;
  }

  /**
   * Process one chunk without blocking Node.js.
   * @param {AudioProcessRequest} obj
   * @returns {Promise<GeneratedAudio>}
   */
  run(obj) {
    return this.enqueue(() =>
      addon.onlineSpeechDenoiserRunAsyncWrapper(this.handle, obj));
  }

  /**
   * Flush buffered output and reset the streaming state without blocking Node.js.
   * @param {boolean} [enableExternalBuffer=true]
   * @returns {Promise<GeneratedAudio>}
   */
  flush(enableExternalBuffer = true) {
    return this.enqueue(() =>
      addon.onlineSpeechDenoiserFlushAsyncWrapper(
          this.handle, enableExternalBuffer));
  }

  /**
   * Reset the streaming state without blocking Node.js.
   * @returns {Promise<void>}
   */
  reset() {
    return this.enqueue(() =>
      addon.onlineSpeechDenoiserResetAsyncWrapper(this.handle));
  }
}

module.exports = {
  OnlineSpeechDenoiser,
};
