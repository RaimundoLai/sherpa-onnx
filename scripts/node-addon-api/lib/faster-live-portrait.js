/** @typedef {import('./types').FaceImage} FaceImage */
/** @typedef {import('./types').FasterLivePortraitConfig} FasterLivePortraitConfig */
/** @typedef {import('./types').OnnxTensor} OnnxTensor */
/** @typedef {import('./types').OnnxTensorOutput} OnnxTensorOutput */

const addon = require('./addon.js');

function assertImage(image) {
  if (!image || typeof image !== 'object' ||
      (!Buffer.isBuffer(image.data) && !(image.data instanceof Uint8Array))) {
    throw new TypeError('image.data must be a Buffer or Uint8Array');
  }
  if (!Number.isInteger(image.width) || image.width <= 0 ||
      !Number.isInteger(image.height) || image.height <= 0 ||
      (image.channels !== 3 && image.channels !== 4)) {
    throw new TypeError('image.width, image.height and image.channels are invalid');
  }
  const stride = image.stride === undefined ? image.width * image.channels : image.stride;
  if (!Number.isInteger(stride) || stride < image.width * image.channels ||
      stride * image.height > image.data.byteLength) {
    throw new TypeError('image.stride is smaller than the image');
  }
}

function channelOrder(image) {
  const format = image.format === undefined ? 'rgb' : image.format;
  if (format === 'rgb' || format === 0) {
    if (image.channels !== 3) throw new TypeError('rgb format requires 3 channels');
    return 'rgb';
  }
  if (format === 'bgr' || format === 1) {
    if (image.channels !== 3) throw new TypeError('bgr format requires 3 channels');
    return 'bgr';
  }
  if (format === 'rgba' || format === 2) {
    if (image.channels !== 4) throw new TypeError('rgba format requires 4 channels');
    return 'rgba';
  }
  if (format === 'bgra' || format === 3) {
    if (image.channels !== 4) throw new TypeError('bgra format requires 4 channels');
    return 'bgra';
  }
  throw new TypeError('image.format must be rgb, bgr, rgba, bgra, or 0..3');
}

/**
 * Convert packed RGB/BGR pixels into the [1, 3, H, W] float tensor used by
 * FasterLivePortrait's appearance and motion ONNX models.
 *
 * @param {FaceImage} image
 * @param {{name?: string, normalize?: boolean}} [options]
 * @returns {OnnxTensor}
 */
function imageToTensor(image, options = {}) {
  assertImage(image);
  const order = channelOrder(image);
  const normalize = options.normalize === undefined ? true : options.normalize;
  const width = image.width;
  const height = image.height;
  const channels = image.channels;
  const stride = image.stride === undefined ? width * channels : image.stride;
  const bytes = image.data;
  const plane = width * height;
  const data = new Float32Array(plane * 3);
  for (let y = 0; y < height; ++y) {
    const row = y * stride;
    for (let x = 0; x < width; ++x) {
      const source = row + x * channels;
      let r;
      let g;
      let b;
      if (order === 'rgb' || order === 'rgba') {
        r = bytes[source];
        g = bytes[source + 1];
        b = bytes[source + 2];
      } else {
        b = bytes[source];
        g = bytes[source + 1];
        r = bytes[source + 2];
      }
      const index = y * width + x;
      const scale = normalize ? 1 / 255 : 1;
      data[index] = r * scale;
      data[plane + index] = g * scale;
      data[2 * plane + index] = b * scale;
    }
  }
  return {
    ...(options.name === undefined ? {} : {name: options.name}),
    type: 'float32',
    shape: [1, 3, height, width],
    data,
  };
}

/**
 * @typedef {Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array} TypedArray
 */

/**
 * @param {any} value
 * @returns {value is TypedArray}
 */
function isTypedArray(value) {
  return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

function readPoint(point) {
  let result;
  if (Array.isArray(point) || isTypedArray(point)) {
    if (point.length < 2) throw new TypeError('Each landmark needs x and y');
    result = {x: Number(point[0]), y: Number(point[1]), z: point.length > 2 ? Number(point[2]) : 0};
  } else if (point && typeof point === 'object') {
    result = {x: Number(point.x), y: Number(point.y), z: point.z === undefined ? 0 : Number(point.z)};
  } else {
    throw new TypeError('Landmarks must contain [x,y] or {x,y,z} points');
  }
  if (!Number.isFinite(result.x) || !Number.isFinite(result.y) ||
      !Number.isFinite(result.z)) {
    throw new TypeError('Landmark coordinates must be finite numbers');
  }
  return result;
}

/**
 * @param {any} value
 * @returns {Array<any> | TypedArray}
 */
function unwrapLandmarks(value) {
  if (value && typeof value === 'object' && !Array.isArray(value) &&
      !isTypedArray(value)) {
    const candidate = value.faceLandmarks || value.multiFaceLandmarks || value.landmarks;
    if (Array.isArray(candidate)) {
      // MediaPipe Face Landmarker returns one array per face, while some
      // integrations expose one face directly.
      value = Array.isArray(candidate[0]) || isTypedArray(candidate[0])
        ? candidate[0]
        : candidate;
    }
  }
  if (!Array.isArray(value) && !isTypedArray(value)) {
    throw new TypeError('MediaPipe landmarks must be an array of points');
  }
  if (value.length === 0) throw new TypeError('MediaPipe landmarks must not be empty');
  return value;
}

/**
 * MediaPipe Face Mesh landmarks represented in pixel coordinates and ready to
 * be passed to later crop/landmark stages. The class does not load MediaPipe;
 * it accepts the result produced by MediaPipe in Node, a browser, or another
 * host process.
 */
class MediaPipeLandmarks {
  /**
   * @param {Array|Float32Array|Object} landmarks
   * @param {{width:number, height:number, normalized?:boolean, includeZ?:boolean, dimension?:2|3}} options
   */
  constructor(landmarks, options) {
    if (!options || !Number.isFinite(options.width) || options.width <= 0 ||
        !Number.isFinite(options.height) || options.height <= 0) {
      throw new TypeError('MediaPipeLandmarks requires positive width and height');
    }
    const normalized = options.normalized === undefined ? true : options.normalized;
    const includeZ = options.includeZ === true;
    const dimension = options.dimension === undefined ? 2 : options.dimension;
    if (dimension !== 2 && dimension !== 3) {
      throw new TypeError('MediaPipeLandmarks dimension must be 2 or 3');
    }
    const source = unwrapLandmarks(landmarks);
    const points = [];
    if (typeof source[0] === 'number') {
      if (source.length % dimension !== 0) {
        throw new TypeError('Flat landmark array length is not divisible by dimension');
      }
      for (let i = 0; i < source.length; i += dimension) {
        points.push(readPoint(source.slice(i, i + dimension)));
      }
    } else {
      for (const point of source) points.push(readPoint(point));
    }
    const outputDimension = includeZ ? 3 : 2;
    this.width = options.width;
    this.height = options.height;
    this.count = points.length;
    this.dimension = outputDimension;
    this.data = new Float32Array(points.length * outputDimension);
    for (let i = 0; i < points.length; ++i) {
      const point = points[i];
      const x = normalized ? point.x * options.width : point.x;
      const y = normalized ? point.y * options.height : point.y;
      this.data[i * outputDimension] = x;
      this.data[i * outputDimension + 1] = y;
      if (includeZ) this.data[i * outputDimension + 2] = normalized ? point.z * options.width : point.z;
    }
  }

  /** @returns {OnnxTensor} A [1, landmarkCount, dimension] float tensor. */
  toTensor(options = {}) {
    const name = options.name;
    return {
      ...(name === undefined ? {} : {name}),
      type: 'float32',
      shape: options.batch === false ? [this.count, this.dimension] : [1, this.count, this.dimension],
      data: new Float32Array(this.data),
    };
  }

  /** @returns {number[]} [x1, y1, x2, y2] in source-image pixels. */
  bbox() {
    let x1 = Infinity;
    let y1 = Infinity;
    let x2 = -Infinity;
    let y2 = -Infinity;
    for (let i = 0; i < this.count; ++i) {
      const x = this.data[i * this.dimension];
      const y = this.data[i * this.dimension + 1];
      x1 = Math.min(x1, x);
      y1 = Math.min(y1, y);
      x2 = Math.max(x2, x);
      y2 = Math.max(y2, y);
    }
    return [x1, y1, x2, y2];
  }

  /** @returns {number[][]} Pixel-coordinate landmark points. */
  toArray() {
    const result = [];
    for (let i = 0; i < this.count; ++i) {
      const point = [this.data[i * this.dimension], this.data[i * this.dimension + 1]];
      if (this.dimension === 3) point.push(this.data[i * this.dimension + 2]);
      result.push(point);
    }
    return result;
  }
}

/**
 * @param {Array|Float32Array|Object} landmarks
 * @param {{width:number, height:number, normalized?:boolean, includeZ?:boolean, dimension?:2|3, name?:string}} options
 * @returns {OnnxTensor}
 */
function mediaPipeLandmarksToTensor(landmarks, options) {
  return new MediaPipeLandmarks(landmarks, options).toTensor(options);
}

/**
 * Multi-model ONNX Runtime interface matching FasterLivePortrait's exported
 * model set. The `models` object maps logical names to ONNX paths.
 */
class FasterLivePortrait {
  /** @param {FasterLivePortraitConfig|Object} configOrHandle */
  /**
   * Asynchronously create a model set on an N-API worker thread.
   * @param {FasterLivePortraitConfig|Object} config
   * @returns {Promise<FasterLivePortrait>}
   */
  static async create(config) {
    if (typeof addon.createFasterLivePortraitModelSetAsync === 'function') {
      const handle = await addon.createFasterLivePortraitModelSetAsync(config);
      const instance = new FasterLivePortrait(handle);
      instance.config = config;
      return instance;
    }
    return new FasterLivePortrait(config);
  }

  constructor(configOrHandle) {
    if (configOrHandle && typeof configOrHandle === 'object' &&
        configOrHandle.models !== undefined) {
      this.handle = addon.createFasterLivePortraitModelSet(configOrHandle);
      this.config = configOrHandle;
    } else {
      this.handle = configOrHandle;
    }
  }

  /** @returns {{models: Object[]}} */
  getModelInfo() {
    return JSON.parse(addon.fasterLivePortraitGetModelInfo(this.handle));
  }

  /**
   * @param {string} modelName
   * @param {OnnxTensor[]} inputs
   * @returns {OnnxTensorOutput[]}
   */
  run(modelName, inputs) {
    if (typeof modelName !== 'string' || modelName.length === 0) {
      throw new TypeError('modelName must be a non-empty string');
    }
    if (!Array.isArray(inputs)) throw new TypeError('inputs must be an array');
    return addon.fasterLivePortraitRun(this.handle, modelName, inputs);
  }

  /**
   * Async version of run().  Inference is queued on an N-API worker thread so
   * callers can await it without blocking Electron's renderer/main event loop.
   * @param {string} modelName
   * @param {OnnxTensor[]} inputs
   * @returns {Promise<OnnxTensorOutput[]>}
   */
  runAsync(modelName, inputs) {
    if (typeof modelName !== 'string' || modelName.length === 0) {
      throw new TypeError('modelName must be a non-empty string');
    }
    if (!Array.isArray(inputs)) throw new TypeError('inputs must be an array');
    if (typeof addon.fasterLivePortraitRunAsync !== 'function') {
      return Promise.resolve().then(() => this.run(modelName, inputs));
    }
    return addon.fasterLivePortraitRunAsync(this.handle, modelName, inputs);
  }

  /**
   * Run a single-input image graph such as appearanceFeatureExtractor,
   * motionExtractor, or landmark.
   * @param {string} modelName
   * @param {FaceImage} image
   * @param {{inputName?:string, normalize?:boolean}} [options]
   * @returns {OnnxTensorOutput[]}
   */
  runImage(modelName, image, options = {}) {
    return this.run(modelName, [imageToTensor(image, {
      name: options.inputName,
      normalize: options.normalize,
    })]);
  }

  /** @returns {Promise<OnnxTensorOutput[]>} */
  runImageAsync(modelName, image, options = {}) {
    return this.runAsync(modelName, [imageToTensor(image, {
      name: options.inputName,
      normalize: options.normalize,
    })]);
  }
}

module.exports = {
  FasterLivePortrait,
  MediaPipeLandmarks,
  imageToTensor,
  mediaPipeLandmarksToTensor,
};
