const addon = require('./addon.js');

class OfflineTts {
  constructor(handle) {
    this.handle = handle;
    this.numSpeakers = addon.getOfflineTtsNumSpeakers(this.handle);
    this.sampleRate = addon.getOfflineTtsSampleRate(this.handle);
  }

  static async createAsync(config) {
    const handle = await addon.createOfflineTtsAsync(config);
    return new OfflineTts(handle);
  }

  /*
   input obj: {text: "xxxx", sid: 0, speed: 1.0}
   where text is a string, sid is a int32, speed is a float

   return an object {samples: Float32Array, sampleRate: <a number>}
   */
  generate(obj) {
    return addon.offlineTtsGenerate(this.handle, obj);
  }

  generateAsync(obj) {
    return addon.offlineTtsGenerateAsync(this.handle, obj);
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

module.exports = {
  OfflineTts,
}
