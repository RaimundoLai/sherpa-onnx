/** @typedef {import('./types').FaceImage} FaceImage */
/** @typedef {import('./types').FaceDetection} FaceDetection */
/** @typedef {import('./types').FaceDetectorConfig} FaceDetectorConfig */
/** @typedef {import('./types').AuraFaceConfig} AuraFaceConfig */

const addon = require('./addon.js');

const FACE_DETECTOR_DEFAULTS = Object.freeze({
  numThreads: 2,
  debug: false,
  provider: 'cpu',
  inputWidth: 640,
  inputHeight: 640,
  scoreThreshold: 0.5,
  nmsThreshold: 0.4,
  maxFaces: 0,
});

/**
 * Create a face detector configuration without coupling callers to the
 * native addon defaults. The model path remains application-specific.
 * @param {Partial<FaceDetectorConfig>|Object} overrides
 * @returns {FaceDetectorConfig|Object}
 */
function createFaceDetectorConfig(overrides = {}) {
  if (!overrides || typeof overrides !== 'object' || Array.isArray(overrides)) {
    throw new TypeError('Face detector options must be an object');
  }
  return {...FACE_DETECTOR_DEFAULTS, ...overrides};
}

/**
 * Generic ONNX face detector.
 *
 * The default model is MediaPipe Face Detection. The native symbol keeps its
 * historical name for ABI compatibility; legacy SCRFD/RetinaFace graphs are
 * still accepted when an application explicitly supplies one.
 */
class FaceDetector {
  /**
   * @param {FaceDetectorConfig|Object} configOrHandle
   */
  constructor(configOrHandle) {
    if (configOrHandle && typeof configOrHandle === 'object' &&
        configOrHandle.model !== undefined) {
      const create = addon.createFaceDetector || addon.createRetinaFaceDetector;
      this.handle = create(configOrHandle);
      this.config = configOrHandle;
    } else {
      this.handle = configOrHandle;
    }
  }

  /**
   * @param {FaceImage} image
   * @returns {FaceDetection[]}
   */
  detect(image) {
    const detect = addon.faceDetectorDetect || addon.retinaFaceDetectorDetect;
    return detect(this.handle, image);
  }

  /**
   * Async version of detect(). ONNX Runtime runs on an N-API worker thread,
   * so callers can await face detection without blocking Electron.
   * @param {FaceImage} image
   * @returns {Promise<FaceDetection[]>}
   */
  detectAsync(image) {
    const detect = addon.faceDetectorDetectAsync || addon.retinaFaceDetectorDetectAsync;
    if (typeof detect !== 'function') {
      return Promise.resolve().then(() => this.detect(image));
    }
    return detect(this.handle, image);
  }
}

// Backward-compatible JavaScript name for applications that used the first
// experimental API. New code should use FaceDetector.
const RetinaFaceDetector = FaceDetector;

/**
 * AuraFace ArcFace-compatible embedding extractor.
 */
class AuraFaceRecognizer {
  /**
   * @param {AuraFaceConfig|Object} configOrHandle
   */
  constructor(configOrHandle) {
    if (configOrHandle && typeof configOrHandle === 'object' &&
        configOrHandle.model !== undefined) {
      this.handle = addon.createAuraFaceRecognizer(configOrHandle);
      this.config = configOrHandle;
    } else {
      this.handle = configOrHandle;
    }
    this.dim = addon.auraFaceRecognizerDim(this.handle);
  }

  /**
   * Compute a normalized embedding for a detected face.
   * @param {FaceImage} image
   * @param {FaceDetection} [face]
   * @returns {Float32Array}
   */
  compute(image, face) {
    if (face === undefined) {
      return addon.auraFaceRecognizerComputeEmbedding(this.handle, image);
    }
    return addon.auraFaceRecognizerComputeEmbedding(this.handle, image, face);
  }

  /**
   * Async version of compute(). The ONNX embedding graph runs on an N-API
   * worker thread and resolves with an owned Float32Array.
   * @param {FaceImage} image
   * @param {FaceDetection} [face]
   * @returns {Promise<Float32Array>}
   */
  computeAsync(image, face) {
    const compute = addon.auraFaceRecognizerComputeEmbeddingAsync;
    if (typeof compute !== 'function') {
      return Promise.resolve().then(() => this.compute(image, face));
    }
    return face === undefined
      ? compute(this.handle, image)
      : compute(this.handle, image, face);
  }
}

/**
 * Compute cosine similarity between two face embeddings.
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} [dim]
 * @returns {number}
 */
function faceCosineSimilarity(a, b, dim) {
  if (dim === undefined) return addon.faceCosineSimilarity(a, b);
  return addon.faceCosineSimilarity(a, b, dim);
}

/**
 * Compare two normalized face embeddings with a configurable threshold.
 * The default is a starting point; applications should calibrate it with
 * their camera, lighting, and target false-match rate.
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} [threshold=0.45]
 * @returns {boolean}
 */
function faceSamePerson(a, b, threshold = 0.45) {
  if (!Number.isFinite(threshold)) {
    throw new TypeError('threshold must be a finite number');
  }
  return faceCosineSimilarity(a, b) >= threshold;
}

/**
 * Small identity tracker for grouping detections across frames. The tracker
 * only stores embeddings and IDs; the application remains responsible for
 * deciding which IDs should receive a box or other UI treatment.
 */
class FaceIdentityTracker {
  /**
   * @param {{threshold?:number, maxIdentities?:number}} [options]
   */
  constructor(options = {}) {
    this.threshold = options.threshold === undefined ? 0.45 : options.threshold;
    this.maxIdentities = options.maxIdentities === undefined ? 0 : options.maxIdentities;
    this.identities = [];
  }

  /**
   * Assign an embedding to the closest identity, or create a new identity.
   * @param {Float32Array} embedding
   * @returns {number}
   */
  match(embedding) {
    if (!(embedding instanceof Float32Array) || embedding.length === 0) {
      throw new TypeError('embedding must be a non-empty Float32Array');
    }
    let bestIndex = -1;
    let bestScore = -1;
    for (let i = 0; i < this.identities.length; ++i) {
      if (this.identities[i].embedding.length !== embedding.length) continue;
      const score = faceCosineSimilarity(embedding, this.identities[i].embedding);
      if (score > bestScore) {
        bestScore = score;
        bestIndex = i;
      }
    }
    if (bestIndex >= 0 && bestScore >= this.threshold) {
      this._updateCentroid(this.identities[bestIndex], embedding);
      return this.identities[bestIndex].id;
    }
    if (this.maxIdentities > 0 && this.identities.length >= this.maxIdentities) {
      return bestIndex >= 0 ? this.identities[bestIndex].id : -1;
    }
    const identity = {
      id: this.identities.length,
      embedding: new Float32Array(embedding),
      samples: 1,
    };
    this.identities.push(identity);
    return identity.id;
  }

  /** @returns {{id:number, samples:number}[]} */
  getIdentities() {
    return this.identities.map((identity) => ({
      id: identity.id,
      samples: identity.samples,
    }));
  }

  reset() {
    this.identities = [];
  }

  _updateCentroid(identity, embedding) {
    const weight = identity.samples;
    for (let i = 0; i < embedding.length; ++i) {
      identity.embedding[i] =
          (identity.embedding[i] * weight + embedding[i]) / (weight + 1);
    }
    let norm = 0;
    for (const value of identity.embedding) norm += value * value;
    norm = Math.sqrt(norm);
    if (norm > 1e-12) {
      for (let i = 0; i < identity.embedding.length; ++i) {
        identity.embedding[i] /= norm;
      }
    }
    identity.samples += 1;
  }
}

module.exports = {
  FaceDetector,
  RetinaFaceDetector,
  AuraFaceRecognizer,
  FaceIdentityTracker,
  faceCosineSimilarity,
  faceSamePerson,
  FACE_DETECTOR_DEFAULTS,
  createFaceDetectorConfig,
};
