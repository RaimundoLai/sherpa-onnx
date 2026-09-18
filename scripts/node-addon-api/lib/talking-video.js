'use strict';

const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawnSync, spawn} = require('child_process');
const {FasterLivePortrait} = require('./faster-live-portrait.js');
const {JoyVASA} = require('./joyvasa.js');
const {FaceDetector, createFaceDetectorConfig} = require('./face.js');

const TALKING_VIDEO_DEFAULTS = Object.freeze({
  poseSmoothing: 0.12,
  eyeSmoothing: 0.35,
  mouthSmoothing: 0.50,
  eyeOpeningScale: 0.45,
  nativeFrameBatchSize: 4,
  faceDetectorOptions: Object.freeze({
    scoreThreshold: 0.5,
    nmsThreshold: 0.4,
    maxFaces: 1,
  }),
  joyvasaOptions: Object.freeze({}),
});

/**
 * Normalize public talking-video options. Applications can keep this object
 * in their own configuration and change rendering behavior without editing
 * this module or the generated patch.
 * @param {Object} overrides
 * @returns {Object}
 */
function createTalkingVideoOptions(overrides = {}) {
  if (!overrides || typeof overrides !== 'object' || Array.isArray(overrides)) {
    throw new TypeError('Talking-video options must be an object');
  }
  return {
    ...TALKING_VIDEO_DEFAULTS,
    ...overrides,
    faceDetectorOptions: {
      ...TALKING_VIDEO_DEFAULTS.faceDetectorOptions,
      ...(overrides.faceDetectorOptions || {}),
    },
    joyvasaOptions: {
      ...TALKING_VIDEO_DEFAULTS.joyvasaOptions,
      ...(overrides.joyvasaOptions || {}),
    },
  };
}

function resolveFaceDetectorOptions(options, model, landmarkModel) {
  const overrides = options.faceDetectorOptions || {};
  return createFaceDetectorConfig({
    ...overrides,
    model: overrides.model || model,
    landmarkModel: overrides.landmarkModel || landmarkModel,
    provider: overrides.provider || options.provider || 'cpu',
    numThreads: overrides.numThreads || options.numThreads || 2,
  });
}

function resolveJoyVasaOptions(options, metadata, provider, numThreads) {
  const overrides = options.joyvasaOptions || {};
  const config = {
    ...overrides,
    metadata,
    provider: overrides.provider || provider,
    numThreads: overrides.numThreads || numThreads,
  };
  if (options.nDiffSteps !== undefined && config.nDiffSteps === undefined) {
    config.nDiffSteps = Number(options.nDiffSteps);
  }
  return config;
}

async function detectFacesAsync(detector, image) {
  if (typeof detector.detectAsync === 'function') {
    return await detector.detectAsync(image);
  }
  return detector.detect(image);
}

function spawnAsync(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      ...options,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.once('error', reject);
    child.once('close', (status, signal) => resolve({status, signal, stdout, stderr}));
  });
}

function modelOutput(outputs, name) {
  const output = outputs.find((item) => item.name === name);
  if (!output) throw new Error(`Missing ONNX output '${name}'`);
  return output;
}

function finiteOutputs(outputs, label) {
  for (const output of outputs) {
    for (const value of output.data) {
      if (!Number.isFinite(value)) throw new Error(`${label} produced a non-finite output`);
    }
  }
  return outputs;
}

function flatten(value) {
  if (Array.isArray(value)) return value.flat(Infinity).map(Number);
  return Array.from(value, Number);
}

function matMul(a, b, aRows, aCols, bCols) {
  const result = new Float32Array(aRows * bCols);
  for (let row = 0; row < aRows; ++row) {
    for (let col = 0; col < bCols; ++col) {
      let value = 0;
      for (let k = 0; k < aCols; ++k) value += a[row * aCols + k] * b[k * bCols + col];
      result[row * bCols + col] = value;
    }
  }
  return result;
}

function transpose3(a) {
  return new Float32Array([a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]);
}

function rotationMatrix(pitch, yaw, roll) {
  const radians = (value) => value * Math.PI / 180;
  const x = radians(pitch);
  const y = radians(yaw);
  const z = radians(roll);
  const rx = new Float32Array([1, 0, 0, 0, Math.cos(x), -Math.sin(x), 0, Math.sin(x), Math.cos(x)]);
  const ry = new Float32Array([Math.cos(y), 0, Math.sin(y), 0, 1, 0, -Math.sin(y), 0, Math.cos(y)]);
  const rz = new Float32Array([Math.cos(z), -Math.sin(z), 0, Math.sin(z), Math.cos(z), 0, 0, 0, 1]);
  return transpose3(matMul(rz, matMul(ry, rx, 3, 3, 3), 3, 3, 3));
}

function headposeDegree(output) {
  const logits = output.data;
  let maxValue = -Infinity;
  for (const value of logits) maxValue = Math.max(maxValue, value);
  let denominator = 0;
  let weighted = 0;
  for (let i = 0; i < logits.length; ++i) {
    const probability = Math.exp(logits[i] - maxValue);
    denominator += probability;
    weighted += probability * i;
  }
  return weighted / denominator * 3 - 97.5;
}

const LIVEPORTRAIT_EYE_EXPRESSION_INDICES = [11, 13, 15, 16, 18];
// LivePortrait's 21-point expression layout uses these points for the mouth
// and jaw/lip contour.  Smooth these independently so phoneme motion remains
// responsive without the small frame-to-frame jumps from diffusion noise.
const LIVEPORTRAIT_MOUTH_EXPRESSION_INDICES = [6, 12, 14, 17, 19, 20];
// The remaining points describe cheeks, brows, and other non-lip facial
// regions. They should follow the head-pose smoothing instead of inheriting
// high-frequency diffusion noise.
const LIVEPORTRAIT_STABLE_EXPRESSION_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10];

// JoyVASA's expression channels (especially jaw/lips) must remain responsive,
// but the diffusion output can contain small high-frequency frame noise. Smooth
// the rigid pose, eyelids, and mouth groups independently so lip-sync remains
// responsive without visible micro-jumps.
function smoothRigidPose(previous, current, alpha) {
  if (!previous || alpha >= 1) return current;
  const mix = (oldValue, newValue) => oldValue + (newValue - oldValue) * alpha;
  return {
    ...current,
    pitch: mix(previous.pitch, current.pitch),
    yaw: mix(previous.yaw, current.yaw),
    roll: mix(previous.roll, current.roll),
    scale: mix(previous.scale, current.scale),
    t: new Float32Array([
      mix(previous.t[0], current.t[0]),
      mix(previous.t[1], current.t[1]),
      current.t[2],
    ]),
  };
}

function smoothExpressionGroups(previous, current, alpha, indices) {
  if (!previous || alpha >= 1) return current;
  const exp = new Float32Array(current.exp);
  for (const point of indices) {
    for (let axis = 0; axis < 3; ++axis) {
      const index = point * 3 + axis;
      exp[index] = previous.exp[index] + (current.exp[index] - previous.exp[index]) * alpha;
    }
  }
  return {...current, exp};
}

function smoothMotion(previous, current, poseAlpha, eyeAlpha, mouthAlpha) {
  const pose = smoothRigidPose(previous, current, poseAlpha);
  const stable = smoothExpressionGroups(
    previous,
    pose,
    poseAlpha,
    LIVEPORTRAIT_STABLE_EXPRESSION_INDICES,
  );
  const eyes = smoothExpressionGroups(
    previous,
    stable,
    eyeAlpha,
    LIVEPORTRAIT_EYE_EXPRESSION_INDICES,
  );
  return smoothExpressionGroups(
    previous,
    eyes,
    mouthAlpha,
    LIVEPORTRAIT_MOUTH_EXPRESSION_INDICES,
  );
}

// These are EMA alphas: lower values remove more frame-to-frame jitter at the
// cost of a little more motion lag. Head pose is intentionally conservative
// because pitch/yaw/roll noise is much more visible than lip noise.
const DEFAULT_POSE_SMOOTHING = 0.12;

function smoothingAlpha(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(0, Math.min(1, parsed));
}

function buildExpressionDelta(sourceExp, firstExp, currentExp, eyeOpeningScale) {
  const deltaExp = new Float32Array(63);
  for (let i = 0; i < 63; ++i) deltaExp[i] = sourceExp[i] + currentExp[i] - firstExp[i];
  if (eyeOpeningScale >= 1) return deltaExp;

  // In LivePortrait's 21x3 expression layout, the eye group uses these
  // points. Positive Y is the opening direction for this exported model;
  // reduce only that outward motion. Negative Y (closing/blinking) and all
  // mouth channels remain untouched.
  for (const point of LIVEPORTRAIT_EYE_EXPRESSION_INDICES) {
    const index = point * 3 + 1;
    const motionDelta = currentExp[index] - firstExp[index];
    if (motionDelta > 0) {
      deltaExp[index] = sourceExp[index] + motionDelta * eyeOpeningScale;
    }
  }
  return deltaExp;
}

function transformKeypoints(pitch, yaw, roll, t, exp, scale, kp) {
  const rotation = rotationMatrix(pitch, yaw, roll);
  const result = new Float32Array(63);
  for (let point = 0; point < 21; ++point) {
    for (let axis = 0; axis < 3; ++axis) {
      let value = 0;
      for (let sourceAxis = 0; sourceAxis < 3; ++sourceAxis) {
        value += kp[point * 3 + sourceAxis] * rotation[sourceAxis * 3 + axis];
      }
      result[point * 3 + axis] = scale * (value + exp[point * 3 + axis]);
    }
    result[point * 3] += t[0];
    result[point * 3 + 1] += t[1];
  }
  return result;
}

function sampleRgb(data, width, height, x, y) {
  if (x < 0 || y < 0 || x > width - 1 || y > height - 1) return [0, 0, 0];
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const dx = x - x0;
  const dy = y - y0;
  const pixel = (px, py, channel) => data[(py * width + px) * 3 + channel];
  const result = [];
  for (let channel = 0; channel < 3; ++channel) {
    const top = pixel(x0, y0, channel) * (1 - dx) + pixel(x1, y0, channel) * dx;
    const bottom = pixel(x0, y1, channel) * (1 - dx) + pixel(x1, y1, channel) * dx;
    result.push(top * (1 - dy) + bottom * dy);
  }
  return result;
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function cropRgb(source, width, height, centerX, centerY, side, outputSize) {
  const output = new Uint8Array(outputSize * outputSize * 3);
  for (let y = 0; y < outputSize; ++y) {
    const sourceY = centerY + ((y + 0.5) / outputSize - 0.5) * side;
    for (let x = 0; x < outputSize; ++x) {
      const sourceX = centerX + ((x + 0.5) / outputSize - 0.5) * side;
      const rgb = sampleRgb(source, width, height, sourceX, sourceY);
      const offset = (y * outputSize + x) * 3;
      output[offset] = clampByte(rgb[0]);
      output[offset + 1] = clampByte(rgb[1]);
      output[offset + 2] = clampByte(rgb[2]);
    }
  }
  return output;
}

function resizeRgb(source, width, height, outputSize) {
  const output = new Uint8Array(outputSize * outputSize * 3);
  for (let y = 0; y < outputSize; ++y) {
    const sourceY = (y + 0.5) * height / outputSize - 0.5;
    for (let x = 0; x < outputSize; ++x) {
      const sourceX = (x + 0.5) * width / outputSize - 0.5;
      const rgb = sampleRgb(source, width, height, sourceX, sourceY);
      const offset = (y * outputSize + x) * 3;
      output[offset] = clampByte(rgb[0]);
      output[offset + 1] = clampByte(rgb[1]);
      output[offset + 2] = clampByte(rgb[2]);
    }
  }
  return output;
}

function resizePackedRgb(source, width, height, maxDimension) {
  if (!Number.isFinite(maxDimension) || maxDimension <= 0 || Math.max(width, height) <= maxDimension) {
    return {data: source, width, height};
  }
  const scale = maxDimension / Math.max(width, height);
  const targetWidth = Math.max(2, Math.round(width * scale));
  const targetHeight = Math.max(2, Math.round(height * scale));
  const output = new Uint8Array(targetWidth * targetHeight * 3);
  for (let y = 0; y < targetHeight; ++y) {
    const sourceY = (y + 0.5) * height / targetHeight - 0.5;
    for (let x = 0; x < targetWidth; ++x) {
      const sourceX = (x + 0.5) * width / targetWidth - 0.5;
      const rgb = sampleRgb(source, width, height, sourceX, sourceY);
      const offset = (y * targetWidth + x) * 3;
      output[offset] = clampByte(rgb[0]);
      output[offset + 1] = clampByte(rgb[1]);
      output[offset + 2] = clampByte(rgb[2]);
    }
  }
  return {data: output, width: targetWidth, height: targetHeight};
}

function adaptFeature3d(feature, targetShape) {
  if (!feature || feature.shape.length !== 5 || targetShape.length !== 5) {
    throw new RangeError('warping feature_3d must be a rank-5 tensor');
  }
  const sourceShape = feature.shape;
  if (sourceShape[0] !== targetShape[0] || sourceShape[1] !== targetShape[1] ||
      sourceShape[2] !== targetShape[2]) {
    throw new RangeError(`warping feature_3d shape mismatch: ${sourceShape} vs ${targetShape}`);
  }
  const sourceHeight = sourceShape[3];
  const sourceWidth = sourceShape[4];
  const targetHeight = targetShape[3];
  const targetWidth = targetShape[4];
  if (sourceHeight === targetHeight && sourceWidth === targetWidth) return feature;
  if (!Number.isInteger(targetHeight) || !Number.isInteger(targetWidth) ||
      targetHeight <= 0 || targetWidth <= 0 ||
      sourceHeight % targetHeight !== 0 || sourceWidth % targetWidth !== 0) {
    throw new RangeError(`warping feature_3d only supports integer downsampling: ${sourceShape} vs ${targetShape}`);
  }
  if (targetHeight > sourceHeight || targetWidth > sourceWidth) {
    throw new RangeError(`warping feature_3d cannot upsample: ${sourceShape} vs ${targetShape}`);
  }
  const scaleY = sourceHeight / targetHeight;
  const scaleX = sourceWidth / targetWidth;
  const batch = sourceShape[0];
  const channels = sourceShape[1];
  const depth = sourceShape[2];
  const sourcePlane = sourceHeight * sourceWidth;
  const targetPlane = targetHeight * targetWidth;
  const output = new Float32Array(batch * channels * depth * targetPlane);
  for (let b = 0; b < batch; ++b) {
    for (let c = 0; c < channels; ++c) {
      for (let z = 0; z < depth; ++z) {
        const sourceOffset = ((b * channels + c) * depth + z) * sourcePlane;
        const targetOffset = ((b * channels + c) * depth + z) * targetPlane;
        for (let y = 0; y < targetHeight; ++y) {
          for (let x = 0; x < targetWidth; ++x) {
            let sum = 0;
            for (let dy = 0; dy < scaleY; ++dy) {
              const sourceRow = (y * scaleY + dy) * sourceWidth;
              for (let dx = 0; dx < scaleX; ++dx) {
                sum += feature.data[sourceOffset + sourceRow + x * scaleX + dx];
              }
            }
            output[targetOffset + y * targetWidth + x] =
                sum / (scaleY * scaleX);
          }
        }
      }
    }
  }
  return {
    name: feature.name,
    type: feature.type,
    shape: targetShape,
    data: output,
  };
}

function portraitRun(portrait, modelName, inputs) {
  if (typeof portrait.runAsync === 'function') {
    return portrait.runAsync(modelName, inputs);
  }
  return Promise.resolve(portrait.run(modelName, inputs));
}

function portraitRunImage(portrait, modelName, image) {
  if (typeof portrait.runImageAsync === 'function') {
    return portrait.runImageAsync(modelName, image);
  }
  return Promise.resolve(portrait.runImage(modelName, image));
}

async function addStitchingDelta(portrait, sourceKp, drivingKp) {
  const inputData = new Float32Array(126);
  inputData.set(sourceKp);
  inputData.set(drivingKp, 63);
  const outputs = finiteOutputs(await portraitRun(portrait, 'stitching', [{
    name: 'input', type: 'float32', shape: [1, 126], data: inputData,
  }]), 'stitching');
  const delta = modelOutput(outputs, 'output').data;
  const result = new Float32Array(drivingKp);
  for (let i = 0; i < 63; ++i) result[i] += delta[i];
  for (let point = 0; point < 21; ++point) {
    result[point * 3] += delta[63];
    result[point * 3 + 1] += delta[64];
  }
  return result;
}

function unpackWarpedRgb(output) {
  if (output.shape.length !== 4 || output.shape[0] !== 1 || output.shape[1] !== 3) {
    throw new Error(`Unexpected warping output shape: ${output.shape}`);
  }
  const height = output.shape[2];
  const width = output.shape[3];
  const plane = width * height;
  const rgb = new Uint8Array(plane * 3);
  for (let i = 0; i < plane; ++i) {
    rgb[i * 3] = clampByte(output.data[i] * 255);
    rgb[i * 3 + 1] = clampByte(output.data[plane + i] * 255);
    rgb[i * 3 + 2] = clampByte(output.data[2 * plane + i] * 255);
  }
  return {rgb, width, height};
}

function unpackNativeRgb(output) {
  const bytes = output instanceof Uint8Array ? output : new Uint8Array(output);
  if (bytes.length === 0 || bytes.length % 3 !== 0) {
    throw new Error(`Unexpected native MLX RGB output length: ${bytes.length}`);
  }
  const side = Math.sqrt(bytes.length / 3);
  if (!Number.isInteger(side) || side <= 0) {
    throw new Error(`Native MLX RGB output is not square: ${bytes.length}`);
  }
  return {rgb: new Uint8Array(bytes), width: side, height: side};
}

function createPasteMap(width, height, centerX, centerY, side, generatedWidth, generatedHeight) {
  const count = width * height;
  const x0 = new Uint16Array(count);
  const y0 = new Uint16Array(count);
  const dx = new Float32Array(count);
  const dy = new Float32Array(count);
  const alpha = new Float32Array(count);
  let offset = 0;
  for (let y = 0; y < height; ++y) {
    for (let x = 0; x < width; ++x, ++offset) {
      const xFromCenter = x - centerX;
      const yFromCenter = y - centerY;
      const currentX = xFromCenter * generatedWidth / side + generatedWidth / 2;
      const currentY = yFromCenter * generatedHeight / side + generatedHeight / 2;
      if (currentX < 0 || currentY < 0 || currentX >= generatedWidth || currentY >= generatedHeight) continue;
      const nx = xFromCenter / (side * 0.5);
      const ny = yFromCenter / (side * 0.5);
      const radius = Math.sqrt(nx * nx + ny * ny);
      const currentAlpha = radius <= 0.55 ? 1 : radius >= 0.92 ? 0 : (0.92 - radius) / 0.37;
      if (currentAlpha <= 0) continue;
      const floorX = Math.floor(currentX);
      const floorY = Math.floor(currentY);
      x0[offset] = floorX;
      y0[offset] = floorY;
      dx[offset] = currentX - floorX;
      dy[offset] = currentY - floorY;
      alpha[offset] = currentAlpha;
    }
  }
  return {x0, y0, dx, dy, alpha};
}

function pasteBackWithMap(source, generated, width, height, map) {
  const output = new Uint8Array(source);
  const generatedData = generated.rgb;
  const generatedWidth = generated.width;
  const generatedHeight = generated.height;
  for (let pixel = 0; pixel < width * height; ++pixel) {
    const alpha = map.alpha[pixel];
    if (alpha <= 0) continue;
    const x = map.x0[pixel];
    const y = map.y0[pixel];
    const x1 = Math.min(generatedWidth - 1, x + 1);
    const y1 = Math.min(generatedHeight - 1, y + 1);
    const fractionX = map.dx[pixel];
    const fractionY = map.dy[pixel];
    const top = (y * generatedWidth + x) * 3;
    const topRight = (y * generatedWidth + x1) * 3;
    const bottom = (y1 * generatedWidth + x) * 3;
    const bottomRight = (y1 * generatedWidth + x1) * 3;
    const offset = pixel * 3;
    for (let channel = 0; channel < 3; ++channel) {
      const topValue = generatedData[top + channel] * (1 - fractionX) +
          generatedData[topRight + channel] * fractionX;
      const bottomValue = generatedData[bottom + channel] * (1 - fractionX) +
          generatedData[bottomRight + channel] * fractionX;
      const animated = topValue * (1 - fractionY) + bottomValue * fractionY;
      output[offset + channel] = clampByte(
          alpha * animated + (1 - alpha) * source[offset + channel]);
    }
  }
  return output;
}

function resolveWarpingModel(options) {
  const profile = options.profile || process.env.SHERPA_ONNX_TALKING_PROFILE || 'original';
  if (profile !== 'original' && profile !== 'metal') {
    throw new RangeError("profile must be 'original' or 'metal'");
  }
  if (options.warpingSpadeModel) return {profile, model: options.warpingSpadeModel};
  if (profile === 'metal') {
    return {profile, model: options.metalWarpingSpadeModel ||
      process.env.SHERPA_ONNX_FLP_METAL_WARPING_MODEL ||
      process.env.SHERPA_ONNX_FLP_WARPING_MODEL || 'warping_spade_fp16.onnx'};
  }
  return {profile, model: options.originalWarpingSpadeModel ||
    process.env.SHERPA_ONNX_FLP_ORIGINAL_WARPING_MODEL || 'warping_spade.onnx'};
}

function getFlpModels(modelDir, warpingSpadeModel) {
  const names = {
    appearance: 'appearance_feature_extractor.onnx',
    motion: 'motion_extractor.onnx',
    stitching: 'stitching.onnx',
    warpingSpade: warpingSpadeModel ||
        process.env.SHERPA_ONNX_FLP_WARPING_MODEL || 'warping_spade.onnx',
  };
  const models = {};
  for (const [name, filename] of Object.entries(names)) {
    models[name] = path.isAbsolute(filename) ? filename : path.join(modelDir, filename);
  }
  return models;
}

function resolveFaceDetectorModel(modelDir, explicitModel) {
  const candidates = [
    explicitModel,
    process.env.SHERPA_ONNX_FACE_DETECTOR_MODEL,
    path.join(modelDir, 'mediapipe', 'float', 'face_detector.onnx'),
    path.join(modelDir, 'mediapipe', 'face_detector.onnx'),
    path.join(modelDir, 'face_detector.onnx'),
  ].filter(Boolean).map((candidate) =>
    path.isAbsolute(candidate) ? candidate : path.join(modelDir, candidate));
  const model = candidates.find((candidate) => fs.existsSync(candidate));
  if (!model) {
    throw new Error(
      `No MediaPipe ONNX face detector found under ${modelDir}; ` +
      'set faceDetectorModel or SHERPA_ONNX_FACE_DETECTOR_MODEL');
  }
  return model;
}

function resolveFaceLandmarkModel(modelDir, explicitModel) {
  const candidates = [
    explicitModel,
    process.env.SHERPA_ONNX_FACE_LANDMARK_MODEL,
    path.join(modelDir, 'mediapipe', 'float', 'face_landmark_detector.onnx'),
    path.join(modelDir, 'mediapipe', 'face_landmark_detector.onnx'),
  ].filter(Boolean).map((candidate) =>
    path.isAbsolute(candidate) ? candidate : path.join(modelDir, candidate));
  return candidates.find((candidate) => fs.existsSync(candidate));
}

function auraFaceSeedLandmarks(face) {
  if (!face || !Array.isArray(face.landmarks) || face.landmarks.length !== 10) {
    throw new Error('MediaPipe face detector did not return five-point landmarks');
  }
  const points = [];
  for (let i = 0; i < 5; ++i) {
    points.push([Number(face.landmarks[i * 2]), Number(face.landmarks[i * 2 + 1])]);
  }
  for (const point of points) {
    if (!Number.isFinite(point[0]) || !Number.isFinite(point[1])) {
      throw new Error('MediaPipe face detector returned non-finite landmarks');
    }
  }
  // The MLX crop helper expects image-left eye, image-right eye, nose, then
  // the two mouth corners. Sorting makes this independent of the detector's
  // semantic left/right naming convention.
  const eyes = points.slice(0, 2).sort((a, b) => a[0] - b[0]);
  const mouth = points.slice(3, 5).sort((a, b) => a[0] - b[0]);
  return [eyes[0], eyes[1], points[2], mouth[0], mouth[1]];
}

function auraFaceSeedBbox(face) {
  if (!face || !Array.isArray(face.bbox) || face.bbox.length !== 4) {
    throw new Error('MediaPipe face detector did not return a four-value bbox');
  }
  const bbox = face.bbox.map(Number);
  if (!bbox.every(Number.isFinite) || bbox[2] <= bbox[0] || bbox[3] <= bbox[1]) {
    throw new Error('MediaPipe face detector returned an invalid bbox');
  }
  return bbox;
}

async function renderTalkingVideoMlx(options, source, width, height, maxSeconds, outputFps) {
  if (typeof options.mlxNativeRenderFrame === 'function') {
    return await renderTalkingVideoNativeMlx(options, source, width, height, maxSeconds, outputFps);
  }
  if (!options.sourceImagePath || !options.audioPath || !options.outputRaw) {
    throw new TypeError(
      'MLX talking-video backend requires sourceImagePath, audioPath and outputRaw');
  }
  const modelDir = options.modelDir || process.env.SHERPA_ONNX_FLP_MODEL_DIR;
  if (!modelDir) throw new Error('modelDir is required for MLX MediaPipe detection');
  const detectorModel = resolveFaceDetectorModel(modelDir, options.faceDetectorModel);
  const landmarkModel = resolveFaceLandmarkModel(modelDir, options.faceLandmarkModel);
  const detector = new FaceDetector(resolveFaceDetectorOptions(
    {...options, provider: options.detectorProvider || options.provider || 'cpu'},
    detectorModel,
    landmarkModel,
  ));
  const faces = await detectFacesAsync(detector, {
    data: source,
    width,
    height,
    channels: 3,
    format: 'rgb',
  });
  if (!faces.length) throw new Error('No face found in source image for MLX backend');

  const referenceDir = options.mlxReferenceDir ||
    process.env.SHERPA_ONNX_MLX_REFERENCE_DIR;
  const weightsDir = options.mlxWeightsDir ||
    process.env.SHERPA_ONNX_MLX_WEIGHTS_DIR;
  if (!referenceDir || !weightsDir) {
    throw new Error(
      'MLX backend requires SHERPA_ONNX_MLX_REFERENCE_DIR and ' +
      'SHERPA_ONNX_MLX_WEIGHTS_DIR');
  }
  const referenceRoot = path.resolve(referenceDir);
  if (!fs.existsSync(path.join(referenceRoot, 'configs', 'mlx_infer.yaml')) ||
      !fs.existsSync(path.join(referenceRoot, 'src'))) {
    throw new Error(
      `MLX reference checkout is incomplete: ${referenceRoot}. ` +
      'Set SHERPA_ONNX_MLX_REFERENCE_DIR to the FasterLivePortrait-MLX ' +
      'source checkout; SHERPA_ONNX_MLX_WEIGHTS_DIR should point to the ' +
      'canonical avatar/mlx weights directory.');
  }
  const bridgePath = options.mlxBridgePath || path.resolve(
    __dirname, '../../fasterliveportrait/mlx_bridge.py');
  if (!fs.existsSync(bridgePath)) {
    throw new Error(`MLX bridge does not exist: ${bridgePath}`);
  }
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sherpa-onnx-mlx-'));
  const sourceRaw = path.join(tempDir, 'source.rgb');
  fs.writeFileSync(sourceRaw, source);

  // The Python bridge remains a compatibility path for development/reference
  // checkouts. It keeps the explicit ONNX-vs-MLX motion switch so older
  // environments can compare the two backends without changing the native
  // Electron path below.
  const motionBackend = options.mlxMotionBackend ||
    process.env.SHERPA_ONNX_MLX_MOTION_BACKEND || 'onnx';
  if (motionBackend !== 'onnx' && motionBackend !== 'mlx') {
    throw new RangeError("mlxMotionBackend must be 'onnx' or 'mlx'");
  }
  let motionPath;
  let motionFrames;
  let motionDim;
  let motionFps;
  let motionDiffusionSteps = 0;
  if (motionBackend === 'onnx') {
    const joyMetadata = options.joyvasaMetadata || process.env.SHERPA_ONNX_JOYVASA_METADATA;
    if (!joyMetadata) throw new Error('joyvasaMetadata is required for ONNX MLX motion');
    const joy = new JoyVASA(resolveJoyVasaOptions(
      {...options, provider: options.motionProvider || options.provider || defaultProvider()},
      joyMetadata,
      options.motionProvider || options.provider || defaultProvider(),
      options.numThreads || 2,
    ));
    const sampleRate = options.audioSampleRate || 16000;
    const maxSamples = maxSeconds === 0
      ? options.audioSamples.length
      : Math.max(1, Math.floor(maxSeconds * sampleRate));
    const audioSamples = options.audioSamples.length > maxSamples
      ? options.audioSamples.slice(0, maxSamples)
      : options.audioSamples;
    const motion = typeof joy.generateMotionSequenceAsync === 'function'
      ? await joy.generateMotionSequenceAsync(audioSamples, {
        sampleRate,
        cfg: options.cfg === true,
      })
      : joy.generateMotionSequence(audioSamples, {
        sampleRate,
        cfg: options.cfg === true,
      });
    motionPath = path.join(tempDir, 'motion.f32');
    fs.writeFileSync(
      motionPath,
      Buffer.from(motion.motion.buffer, motion.motion.byteOffset, motion.motion.byteLength),
    );
    motionFrames = motion.frameCount;
    motionDim = joy.motionFeatDim;
    motionFps = joy.fps;
    motionDiffusionSteps = joy.nDiffSteps;
  }

  const defaultPython = path.join(referenceRoot, '.venv', 'bin', 'python');
  const configuredPython = options.mlxPython || process.env.SHERPA_ONNX_MLX_PYTHON;
  // Prefer uv for a real reference project so its pyproject.toml (including
  // opencv-python, numpy, soundfile and MLX) is synchronized automatically.
  // An explicitly configured interpreter still wins for offline/dev setups.
  const python = configuredPython ||
    (fs.existsSync(path.join(referenceRoot, 'pyproject.toml'))
      ? 'uv'
      : (fs.existsSync(defaultPython) ? defaultPython : 'uv'));
  const bridgeArgs = [
    '--reference-root', path.resolve(referenceDir),
    '--weights-root', path.resolve(weightsDir),
    '--source-rgb', sourceRaw,
    '--width', String(width),
    '--height', String(height),
    '--face-landmarks', JSON.stringify(auraFaceSeedLandmarks(faces[0])),
    '--face-bbox', JSON.stringify(auraFaceSeedBbox(faces[0])),
    '--audio', path.resolve(options.audioPath),
    '--output-raw', path.resolve(options.outputRaw),
    '--max-seconds', String(maxSeconds),
    '--output-fps', String(outputFps),
    '--diffusion-steps', String(options.nDiffSteps === undefined ? 0 : options.nDiffSteps),
    '--profile', options.mlxProfile || process.env.SHERPA_ONNX_MLX_PROFILE || 'quality',
    '--cfg-scale', String(options.cfgScale === undefined ? 2.8 : options.cfgScale),
    '--motion-fps', String(motionFps || 25),
    '--motion-diffusion-steps', String(motionDiffusionSteps),
  ];
  if (motionPath) {
    bridgeArgs.push(
      '--motion-f32', path.resolve(motionPath),
      '--motion-frames', String(motionFrames),
      '--motion-dim', String(motionDim),
    );
  }
  if (options.cfg === true) bridgeArgs.push('--cfg');
  let command = python;
  let args = [bridgePath, ...bridgeArgs];
  if (path.basename(python) === 'uv') {
    command = python;
    args = ['run', '--project', referenceRoot,
      'python', bridgePath, ...bridgeArgs];
  }
  const child = await spawnAsync(command, args, {
    cwd: referenceRoot,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      ...(path.basename(python) === 'uv' && !process.env.UV_CACHE_DIR
        ? {UV_CACHE_DIR: path.join(os.tmpdir(), 'sherpa-onnx-uv-cache')}
        : {}),
    },
  });
  if (child.stderr) process.stderr.write(child.stderr);
  if (child.status !== 0) {
    throw new Error(
      `MLX bridge exited with status ${child.status}: ${child.stderr || child.stdout}`);
  }
  const lines = String(child.stdout || '').trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) throw new Error('MLX bridge returned no result');
  let result;
  try {
    result = JSON.parse(lines[lines.length - 1]);
  } catch (error) {
    throw new Error(`Could not parse MLX bridge result: ${error.message}`);
  }
  return {
    ...result,
    backend: 'mlx',
    provider: 'mlx',
    profile: result.profile || 'mlx-quality',
    motionBackend: result.motionBackend || motionBackend,
    diffusionSteps: motionBackend === 'onnx'
      ? motionDiffusionSteps
      : result.diffusionSteps,
  };
}

/**
 * Native Apple-Silicon MLX path. FasterLivePortrait and the complete JoyVASA
 * audio-to-motion path run in the Swift+Cmlx binding. The ONNX motion and
 * stitching graphs remain only for the FLP head-pose/stitching compatibility
 * stage; the 361MB HuBERT checkpoint is not loaded by ONNX anymore.
 */
async function renderTalkingVideoNativeMlx(options, source, width, height, maxSeconds, outputFps) {
  if (!options.outputRaw) throw new TypeError('Native MLX talking-video requires outputRaw');
  const nativeRoot = options.mlxWeightsDir || options.modelDir;
  if (!nativeRoot) throw new Error('Native MLX weights directory is required');
  const motionDir = options.motionModelDir || options.modelDir;
  const motionPath = options.motionModelPath || path.join(motionDir, 'motion_extractor.onnx');
  const stitchingPath = options.stitchingModelPath || path.join(motionDir, 'stitching.onnx');
  if (!fs.existsSync(motionPath) || !fs.existsSync(stitchingPath)) {
    throw new Error(`Native MLX motion baseline requires motion_extractor.onnx and stitching.onnx under ${motionDir}`);
  }
  const detectorModel = resolveFaceDetectorModel(options.modelDir || nativeRoot, options.faceDetectorModel);
  const landmarkModel = resolveFaceLandmarkModel(options.modelDir || nativeRoot, options.faceLandmarkModel);
  const provider = options.provider || defaultProvider();
  const portrait = new FasterLivePortrait({
    provider,
    numThreads: options.numThreads || 2,
    models: {motion: motionPath, stitching: stitchingPath},
  });
  const detector = new FaceDetector(resolveFaceDetectorOptions(options, detectorModel, landmarkModel));
  const faces = await detectFacesAsync(detector, {data: source, width, height, channels: 3, format: 'rgb'});
  if (!faces.length) throw new Error('No face found in source image');
  const face = faces[0];
  const faceWidth = face.bbox[2] - face.bbox[0];
  const faceHeight = face.bbox[3] - face.bbox[1];
  const cropSide = Math.max(faceWidth, faceHeight) * 2.3;
  const cropCenterX = (face.bbox[0] + face.bbox[2]) * 0.5;
  const cropCenterY = (face.bbox[1] + face.bbox[3]) * 0.5 - cropSide * 0.125;
  const sourceCrop512 = cropRgb(source, width, height, cropCenterX, cropCenterY, cropSide, 512);
  const sourceCrop256 = resizeRgb(sourceCrop512, 512, 512, 256);
  const sourceCropImage = {data: sourceCrop256, width: 256, height: 256, channels: 3, format: 'rgb'};
  const sourceMotion = finiteOutputs(await portraitRunImage(portrait, 'motion', sourceCropImage), 'motion');
  const sourcePitch = headposeDegree(modelOutput(sourceMotion, 'pitch'));
  const sourceYaw = headposeDegree(modelOutput(sourceMotion, 'yaw'));
  const sourceRoll = headposeDegree(modelOutput(sourceMotion, 'roll'));
  const sourceT = flatten(modelOutput(sourceMotion, 't').data);
  const sourceExp = flatten(modelOutput(sourceMotion, 'exp').data);
  const sourceScale = flatten(modelOutput(sourceMotion, 'scale').data)[0];
  const sourceKp = flatten(modelOutput(sourceMotion, 'kp').data);
  const sourceR = rotationMatrix(sourcePitch, sourceYaw, sourceRoll);
  const sourceCanonicalKp = transformKeypoints(sourcePitch, sourceYaw, sourceRoll,
    sourceT, sourceExp, sourceScale, sourceKp);

  const joyMetadata = options.joyvasaMetadata;
  const joyTemplate = options.joyvasaTemplate;
  if (!joyMetadata || !joyTemplate) throw new Error('Native MLX path requires JoyVASA metadata and template');
  const sampleRate = options.audioSampleRate || 16000;
  const maxSamples = maxSeconds === 0 ? options.audioSamples.length :
    Math.max(1, Math.floor(maxSeconds * sampleRate));
  const audioSamples = options.audioSamples.length > maxSamples
    ? options.audioSamples.slice(0, maxSamples) : options.audioSamples;
  const joyConfig = JSON.parse(fs.readFileSync(joyMetadata, 'utf8'));
  const joyMotionConfig = joyConfig.motion || {};
  const motionFps = Number(joyMotionConfig.fps || joyConfig.audioInput?.fps || 25);
  const motionDim = Number(joyMotionConfig.motionFeatDim || 73);
  const requestedSteps = options.nDiffSteps === undefined || Number(options.nDiffSteps) === 0
    ? Number(joyMotionConfig.nDiffSteps || 50)
    : Number(options.nDiffSteps);
  const nativeJoyVasa = typeof options.mlxNativeGenerateJoyVasaMotion === 'function' &&
    options.mlxJoyvasaAudioModel && options.mlxJoyvasaMotionModel &&
    fs.existsSync(options.mlxJoyvasaAudioModel) && fs.existsSync(options.mlxJoyvasaMotionModel);
  let motion;
  let motionFrameCount;
  let motionDiffusionSteps = requestedSteps;
  let motionBackend = 'onnx';
  if (nativeJoyVasa) {
    // The exported JoyVASA MLX checkpoint advertises audio guidance.  The
    // reference MLX pipeline enables CFG when no explicit override is given;
    // preserve that default for native MLX so audio does not collapse into
    // weak, mostly-closed-mouth motion.  The ONNX compatibility path below
    // retains its existing explicit `cfg === true` behavior.
    const useCfg = options.cfg === undefined ? true : options.cfg === true;
    motion = await options.mlxNativeGenerateJoyVasaMotion(
      options.mlxJoyvasaAudioModel,
      options.mlxJoyvasaMotionModel,
      audioSamples,
      sampleRate,
      requestedSteps,
      options.cfgScale === undefined ? 2.8 : Number(options.cfgScale),
      useCfg,
    );
    motionFrameCount = Math.floor(motion.length / motionDim);
    motionBackend = 'mlx';
  } else {
    const joy = new JoyVASA(resolveJoyVasaOptions(options, joyMetadata, provider, options.numThreads || 2));
    const joyResult = typeof joy.generateMotionSequenceAsync === 'function'
      ? await joy.generateMotionSequenceAsync(audioSamples, {sampleRate, cfg: options.cfg === true})
      : joy.generateMotionSequence(audioSamples, {sampleRate, cfg: options.cfg === true});
    motion = joyResult.motion;
    motionFrameCount = joyResult.frameCount;
    motionDiffusionSteps = joy.nDiffSteps;
    motionBackend = 'onnx';
  }
  const template = JSON.parse(fs.readFileSync(joyTemplate, 'utf8'));
  const values = (name) => flatten(template[name]);
  const meanExp = values('mean_exp');
  const stdExp = values('std_exp');
  const minScale = values('min_scale')[0];
  const maxScale = values('max_scale')[0];
  const minT = values('min_t');
  const maxT = values('max_t');
  const minPitch = values('min_pitch')[0];
  const maxPitch = values('max_pitch')[0];
  const minYaw = values('min_yaw')[0];
  const maxYaw = values('max_yaw')[0];
  const minRoll = values('min_roll')[0];
  const maxRoll = values('max_roll')[0];
  const decodeMotion = (row) => {
    const exp = new Float32Array(63);
    for (let i = 0; i < 63; ++i) exp[i] = row[i] * stdExp[i] + meanExp[i];
    const t = new Float32Array(3);
    for (let i = 0; i < 3; ++i) t[i] = row[64 + i] * (maxT[i] - minT[i]) + minT[i];
    return {
      exp,
      scale: row[63] * (maxScale - minScale) + minScale,
      t,
      pitch: row[67] * (maxPitch - minPitch) + minPitch,
      yaw: row[68] * (maxYaw - minYaw) + minYaw,
      roll: row[69] * (maxRoll - minRoll) + minRoll,
    };
  };
  if (outputFps > motionFps) throw new RangeError(`outputFps must be no greater than ${motionFps}`);
  const frameCount = Math.max(1, Math.ceil(motionFrameCount / motionFps * outputFps));
  const firstMotion = decodeMotion(motion.slice(0, motionDim));
  const firstR = rotationMatrix(firstMotion.pitch, firstMotion.yaw, firstMotion.roll);
  const outputFd = fs.openSync(options.outputRaw, 'w');
  const sourceCropBuffer = Buffer.from(sourceCrop256);
  // A small bounded batch removes most JS↔Rust↔Swift call overhead while
  // keeping Metal's transient allocations bounded. ONNX remains on its
  // original one-frame path; this is native MLX renderer only.
  const nativeBatchRender = typeof options.mlxNativeRenderFrames === 'function';
  const nativeBatchSize = nativeBatchRender
    ? Math.max(1, Math.min(4, Number(options.nativeFrameBatchSize || 4)))
    : 1;
  const poseAlpha = smoothingAlpha(options.poseSmoothing, DEFAULT_POSE_SMOOTHING);
  const eyeAlpha = smoothingAlpha(options.eyeSmoothing, 0.35);
  const mouthAlpha = smoothingAlpha(options.mouthSmoothing, 0.50);
  const eyeOpeningScale = smoothingAlpha(options.eyeOpeningScale, 0.45);
  let previousRigidPose;
  let pasteMap;
  try {
    for (let frame = 0; frame < frameCount;) {
      const batchEnd = Math.min(frameCount, frame + nativeBatchSize);
      const batchCount = batchEnd - frame;
      const batchDriving = new Float32Array(batchCount * 63);
      for (let batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        const outputFrame = frame + batchIndex;
        const selectedMotionFrame = Math.min(
          motionFrameCount - 1,
          Math.floor(outputFrame * motionFps / outputFps),
        );
        const rawCurrent = decodeMotion(motion.slice(selectedMotionFrame * motionDim,
          (selectedMotionFrame + 1) * motionDim));
        const current = smoothMotion(
          previousRigidPose,
          rawCurrent,
          poseAlpha,
          eyeAlpha,
          mouthAlpha,
        );
        previousRigidPose = current;
        const currentR = rotationMatrix(current.pitch, current.yaw, current.roll);
        const relativeR = matMul(matMul(currentR, transpose3(firstR), 3, 3, 3), sourceR, 3, 3, 3);
        const deltaExp = buildExpressionDelta(
          sourceExp,
          firstMotion.exp,
          current.exp,
          eyeOpeningScale,
        );
        const scale = sourceScale * (current.scale / firstMotion.scale);
        const t = new Float32Array(3);
        for (let i = 0; i < 3; ++i) t[i] = sourceT[i] + current.t[i] - firstMotion.t[i];
        t[2] = 0;
        const drivingKp = new Float32Array(63);
        for (let point = 0; point < 21; ++point) {
          for (let axis = 0; axis < 3; ++axis) {
            let value = 0;
            for (let sourceAxis = 0; sourceAxis < 3; ++sourceAxis) {
              value += sourceKp[point * 3 + sourceAxis] * relativeR[sourceAxis * 3 + axis];
            }
            drivingKp[point * 3 + axis] = scale * (value + deltaExp[point * 3 + axis]);
          }
          drivingKp[point * 3] += t[0];
          drivingKp[point * 3 + 1] += t[1];
        }
        const stitchedKp = await addStitchingDelta(portrait, sourceCanonicalKp, drivingKp);
        if (stitchedKp.length !== 63) throw new Error('FLP stitching output must contain 63 floats');
        batchDriving.set(stitchedKp, batchIndex * 63);
      }

      if (nativeBatchRender) {
        const rendered = await options.mlxNativeRenderFrames(
          nativeRoot,
          sourceCropBuffer,
          256,
          256,
          batchDriving,
          sourceCanonicalKp,
        );
        const bytes = rendered.data instanceof Uint8Array
          ? rendered.data
          : new Uint8Array(rendered.data);
        const generatedWidth = Number(rendered.width);
        const generatedHeight = Number(rendered.height);
        const frameBytes = generatedWidth * generatedHeight * 3;
        if (!Number.isInteger(generatedWidth) || !Number.isInteger(generatedHeight) ||
            generatedWidth <= 0 || generatedHeight <= 0 || bytes.length !== frameBytes * batchCount) {
          throw new Error(`Unexpected native MLX frame batch size: ${bytes.length}`);
        }
        for (let batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
          const generated = {
            rgb: bytes.subarray(batchIndex * frameBytes, (batchIndex + 1) * frameBytes),
            width: generatedWidth,
            height: generatedHeight,
          };
          if (!pasteMap) pasteMap = createPasteMap(width, height, cropCenterX, cropCenterY,
            cropSide, generated.width, generated.height);
          fs.writeSync(outputFd, pasteBackWithMap(source, generated, width, height, pasteMap));
          if (typeof options.onFrame === 'function') options.onFrame(frame + batchIndex + 1, frameCount);
        }
      } else {
        // Kept as a compatibility fallback for an older native binding.
        for (let batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
          const oneDriving = batchDriving.subarray(batchIndex * 63, (batchIndex + 1) * 63);
          const generatedBytes = await options.mlxNativeRenderFrame(
            nativeRoot,
            sourceCropBuffer,
            256,
            256,
            oneDriving,
            sourceCanonicalKp,
          );
          const generated = unpackNativeRgb(generatedBytes);
          if (!pasteMap) pasteMap = createPasteMap(width, height, cropCenterX, cropCenterY,
            cropSide, generated.width, generated.height);
          fs.writeSync(outputFd, pasteBackWithMap(source, generated, width, height, pasteMap));
          if (typeof options.onFrame === 'function') options.onFrame(frame + batchIndex + 1, frameCount);
        }
      }
      frame = batchEnd;
    }
  } finally {
    fs.closeSync(outputFd);
  }
  return {
    width,
    height,
    fps: outputFps,
    frames: frameCount,
    duration: frameCount / outputFps,
    provider: 'mlx',
    profile: 'native-mlx',
    diffusionSteps: motionDiffusionSteps,
    motionBackend,
    face: {score: face.score, bbox: face.bbox},
    audioSamples: audioSamples.length,
  };
}

function defaultProvider() {
  if (process.env.SHERPA_ONNX_TALKING_PROVIDER) {
    return process.env.SHERPA_ONNX_TALKING_PROVIDER;
  }
  return process.platform === 'darwin' ? 'coreml' : 'cpu';
}

function createTalkingVideoPipeline(options = {}) {
  const modelDir = options.modelDir || process.env.SHERPA_ONNX_FLP_MODEL_DIR;
  const joyMetadata = options.joyvasaMetadata || process.env.SHERPA_ONNX_JOYVASA_METADATA;
  if (!modelDir || !joyMetadata) {
    throw new Error('modelDir and joyvasaMetadata are required');
  }
  const provider = options.provider || defaultProvider();
  const numThreads = options.numThreads || 2;
  const warping = resolveWarpingModel(options);
  const warpingModelPath = path.isAbsolute(warping.model) ? warping.model :
    path.join(modelDir, warping.model);
  if (!fs.existsSync(warpingModelPath)) {
    throw new Error(
      `Warping model for profile '${warping.profile}' was not found: ${warpingModelPath}. ` +
      `Set the profile-specific model path or unset SHERPA_ONNX_FLP_WARPING_MODEL.`);
  }
  const faceDetectorModel = resolveFaceDetectorModel(modelDir, options.faceDetectorModel);
  const faceLandmarkModel = resolveFaceLandmarkModel(modelDir, options.faceLandmarkModel);
  const joyConfig = resolveJoyVasaOptions(options, joyMetadata, provider, numThreads);
  const warpingProvider = options.warpingProvider ||
    process.env.SHERPA_ONNX_FLP_WARPING_PROVIDER || 'cpu';
  const portraitModels = getFlpModels(modelDir, warpingModelPath);
  if (provider !== warpingProvider) delete portraitModels.warpingSpade;
  const portrait = new FasterLivePortrait({
    provider,
    numThreads,
    models: portraitModels,
  });
  // ORT 1.28.2 CUDA supports the 5-D GridSample used by the converted
  // opset-20 warping model; callers may still select CPU for compatibility.
  const warpingPortrait = provider === warpingProvider ? portrait :
    new FasterLivePortrait({
      provider: warpingProvider,
      numThreads,
      models: {warpingSpade: warpingModelPath},
    });
  return {
    modelDir,
    profile: warping.profile,
    provider,
    portrait,
    warpingPortrait,
    detector: new FaceDetector(resolveFaceDetectorOptions(options, faceDetectorModel, faceLandmarkModel)),
    joy: new JoyVASA(joyConfig),
    sourceCache: new Map(),
  };
}

/**
 * Render a talking-head video using either the native Node addon/ONNX graphs
 * or the optional Apple-Silicon MLX sidecar. Image codecs and video encoding
 * remain host responsibilities; sourceRgb is packed RGB and audioSamples are
 * mono Float32 samples.
 */
async function renderTalkingVideo(options) {
  if (!options || !options.sourceRgb || !Number.isInteger(options.width) ||
      !Number.isInteger(options.height) || !options.audioSamples) {
    throw new TypeError('renderTalkingVideo requires sourceRgb, width, height and audioSamples');
  }
  options = createTalkingVideoOptions(options);
  const inputSource = new Uint8Array(options.sourceRgb);
  const inputWidth = options.width;
  const inputHeight = options.height;
  if (inputSource.length !== inputWidth * inputHeight * 3) throw new RangeError('sourceRgb size does not match width and height');
  const resizedSource = resizePackedRgb(inputSource, inputWidth, inputHeight, options.maxDimension);
  const source = resizedSource.data;
  const width = resizedSource.width;
  const height = resizedSource.height;
  const backend = options.backend || process.env.SHERPA_ONNX_TALKING_BACKEND || 'onnx';
  const maxSeconds = options.maxSeconds === undefined ? 0 : Number(options.maxSeconds);
  if (!Number.isFinite(maxSeconds) || maxSeconds < 0) throw new RangeError('maxSeconds must be non-negative');
  const outputFps = options.outputFps === undefined ? 25 : Number(options.outputFps);
  if (!Number.isFinite(outputFps) || outputFps <= 0 || outputFps > 25) {
    throw new RangeError('outputFps must be greater than 0 and no greater than 25');
  }
  if (backend === 'mlx') {
    return await renderTalkingVideoMlx(options, source, width, height, maxSeconds, outputFps);
  }
  if (backend !== 'onnx') throw new RangeError("backend must be 'onnx' or 'mlx'");
  const modelDir = options.modelDir || process.env.SHERPA_ONNX_FLP_MODEL_DIR;
  const joyMetadata = options.joyvasaMetadata || process.env.SHERPA_ONNX_JOYVASA_METADATA;
  const joyTemplate = options.joyvasaTemplate || process.env.SHERPA_ONNX_JOYVASA_TEMPLATE;
  if (!modelDir || !joyMetadata || !joyTemplate) {
    throw new Error('modelDir, joyvasaMetadata and joyvasaTemplate are required');
  }

  const pipeline = options.pipeline || createTalkingVideoPipeline({
    modelDir,
    joyvasaMetadata: joyMetadata,
    provider: options.provider,
    numThreads: options.numThreads,
    nDiffSteps: options.nDiffSteps,
    profile: options.profile,
    originalWarpingSpadeModel: options.originalWarpingSpadeModel,
    metalWarpingSpadeModel: options.metalWarpingSpadeModel,
    warpingSpadeModel: options.warpingSpadeModel,
    warpingProvider: options.warpingProvider,
    faceDetectorModel: options.faceDetectorModel,
    faceLandmarkModel: options.faceLandmarkModel,
  });
  const {provider, portrait, warpingPortrait = portrait, detector, joy} = pipeline;
  const sourceImage = {data: source, width, height, channels: 3, format: 'rgb'};
  const sourceKey = `${width}x${height}:${crypto.createHash('sha1').update(source).digest('hex')}`;
  let sourceState = pipeline.sourceCache.get(sourceKey);
  if (!sourceState) {
    const faces = await detectFacesAsync(detector, sourceImage);
    if (faces.length === 0) throw new Error('No face found in source image');
    const face = faces[0];
    const faceWidth = face.bbox[2] - face.bbox[0];
    const faceHeight = face.bbox[3] - face.bbox[1];
    const cropSide = Math.max(faceWidth, faceHeight) * 2.3;
    const cropCenterX = (face.bbox[0] + face.bbox[2]) * 0.5;
    const cropCenterY = (face.bbox[1] + face.bbox[3]) * 0.5 - cropSide * 0.125;
    const sourceCrop512 = cropRgb(source, width, height, cropCenterX, cropCenterY, cropSide, 512);
    const sourceCrop256 = resizeRgb(sourceCrop512, 512, 512, 256);
    const sourceCropImage = {data: sourceCrop256, width: 256, height: 256, channels: 3, format: 'rgb'};
    const appearance = finiteOutputs(await portraitRunImage(portrait, 'appearance', sourceCropImage), 'appearance');
    const sourceMotion = finiteOutputs(await portraitRunImage(portrait, 'motion', sourceCropImage), 'motion');
    const sourcePitch = headposeDegree(modelOutput(sourceMotion, 'pitch'));
    const sourceYaw = headposeDegree(modelOutput(sourceMotion, 'yaw'));
    const sourceRoll = headposeDegree(modelOutput(sourceMotion, 'roll'));
    const sourceT = flatten(modelOutput(sourceMotion, 't').data);
    const sourceExp = flatten(modelOutput(sourceMotion, 'exp').data);
    const sourceScale = flatten(modelOutput(sourceMotion, 'scale').data)[0];
    const sourceKp = flatten(modelOutput(sourceMotion, 'kp').data);
    const sourceR = rotationMatrix(sourcePitch, sourceYaw, sourceRoll);
    const sourceCanonicalKp = transformKeypoints(sourcePitch, sourceYaw, sourceRoll, sourceT, sourceExp, sourceScale, sourceKp);
    sourceState = {face, cropSide, cropCenterX, cropCenterY, appearance, sourceT, sourceExp, sourceScale, sourceKp, sourceR, sourceCanonicalKp};
    if (pipeline.sourceCache.size >= 2) pipeline.sourceCache.delete(pipeline.sourceCache.keys().next().value);
    pipeline.sourceCache.set(sourceKey, sourceState);
  }
  const {face, cropSide, cropCenterX, cropCenterY, appearance, sourceT, sourceExp, sourceScale, sourceKp, sourceR, sourceCanonicalKp} = sourceState;

  const sampleRate = options.audioSampleRate || 16000;
  const maxSamples = maxSeconds === 0 ? options.audioSamples.length : Math.max(1, Math.floor(maxSeconds * sampleRate));
  const audioSamples = options.audioSamples.length > maxSamples ? options.audioSamples.slice(0, maxSamples) : options.audioSamples;
  // generateMotionSequence stitches multiple JoyVASA windows and returns the
  // actual frame count. maxSeconds=0 means no artificial duration cap.
  const joyResult = typeof joy.generateMotionSequenceAsync === 'function'
    ? await joy.generateMotionSequenceAsync(audioSamples, {sampleRate, cfg: options.cfg === true})
    : joy.generateMotionSequence(audioSamples, {sampleRate, cfg: options.cfg === true});
  const template = JSON.parse(fs.readFileSync(joyTemplate, 'utf8'));
  const values = (name) => flatten(template[name]);
  const meanExp = values('mean_exp');
  const stdExp = values('std_exp');
  const minScale = values('min_scale')[0];
  const maxScale = values('max_scale')[0];
  const minT = values('min_t');
  const maxT = values('max_t');
  const minPitch = values('min_pitch')[0];
  const maxPitch = values('max_pitch')[0];
  const minYaw = values('min_yaw')[0];
  const maxYaw = values('max_yaw')[0];
  const minRoll = values('min_roll')[0];
  const maxRoll = values('max_roll')[0];
  const decodeMotion = (row) => {
    const exp = new Float32Array(63);
    for (let i = 0; i < 63; ++i) exp[i] = row[i] * stdExp[i] + meanExp[i];
    const t = new Float32Array(3);
    for (let i = 0; i < 3; ++i) t[i] = row[64 + i] * (maxT[i] - minT[i]) + minT[i];
    return {
      exp,
      scale: row[63] * (maxScale - minScale) + minScale,
      t,
      pitch: row[67] * (maxPitch - minPitch) + minPitch,
      yaw: row[68] * (maxYaw - minYaw) + minYaw,
      roll: row[69] * (maxRoll - minRoll) + minRoll,
    };
  };
  const motion = joyResult.motion;
  const motionFrameCount = joyResult.motionShape[1];
  const fps = outputFps;
  if (!Number.isFinite(fps) || fps <= 0 || fps > joy.fps) {
    throw new RangeError(`outputFps must be greater than 0 and no greater than ${joy.fps}`);
  }
  const frameCount = Math.max(1, Math.ceil(motionFrameCount / joy.fps * fps));
  const firstMotion = decodeMotion(motion.slice(0, joy.motionFeatDim));
  const firstR = rotationMatrix(firstMotion.pitch, firstMotion.yaw, firstMotion.roll);
  const poseAlpha = smoothingAlpha(options.poseSmoothing, DEFAULT_POSE_SMOOTHING);
  const eyeAlpha = smoothingAlpha(options.eyeSmoothing, 0.35);
  const mouthAlpha = smoothingAlpha(options.mouthSmoothing, 0.50);
  let previousRigidPose;
  const outputRaw = options.outputRaw;
  if (!outputRaw) throw new TypeError('outputRaw is required');
  const outputFd = fs.openSync(outputRaw, 'w');
  const sourceFeature = modelOutput(appearance, 'output');
  const warpingModel = warpingPortrait.getModelInfo().models.find(
      (model) => model.name === 'warpingSpade');
  if (!warpingModel) throw new Error("Missing 'warpingSpade' model metadata");
  const warpingFeatureInput = warpingModel.inputs.find(
      (input) => input.name === 'feature_3d');
  if (!warpingFeatureInput) throw new Error("Missing 'feature_3d' warping input metadata");
  const warpingFeature = adaptFeature3d(sourceFeature, warpingFeatureInput.shape);
  let pasteMap;
  try {
    for (let frame = 0; frame < frameCount; ++frame) {
      const motionFrame = Math.min(motionFrameCount - 1, Math.floor(frame * joy.fps / fps));
      const rawCurrent = decodeMotion(motion.slice(
        motionFrame * joy.motionFeatDim,
        (motionFrame + 1) * joy.motionFeatDim,
      ));
      const current = smoothMotion(
        previousRigidPose,
        rawCurrent,
        poseAlpha,
        eyeAlpha,
        mouthAlpha,
      );
      previousRigidPose = current;
      const currentR = rotationMatrix(current.pitch, current.yaw, current.roll);
      const relativeR = matMul(matMul(currentR, transpose3(firstR), 3, 3, 3), sourceR, 3, 3, 3);
      const deltaExp = new Float32Array(63);
      for (let i = 0; i < 63; ++i) deltaExp[i] = sourceExp[i] + current.exp[i] - firstMotion.exp[i];
      const scale = sourceScale * (current.scale / firstMotion.scale);
      const t = new Float32Array(3);
      for (let i = 0; i < 3; ++i) t[i] = sourceT[i] + current.t[i] - firstMotion.t[i];
      t[2] = 0;
      const drivingKp = new Float32Array(63);
      for (let point = 0; point < 21; ++point) {
        for (let axis = 0; axis < 3; ++axis) {
          let value = 0;
          for (let sourceAxis = 0; sourceAxis < 3; ++sourceAxis) value += sourceKp[point * 3 + sourceAxis] * relativeR[sourceAxis * 3 + axis];
          drivingKp[point * 3 + axis] = scale * (value + deltaExp[point * 3 + axis]);
        }
        drivingKp[point * 3] += t[0];
        drivingKp[point * 3 + 1] += t[1];
      }
      const stitchedKp = await addStitchingDelta(portrait, sourceCanonicalKp, drivingKp);
      const warped = finiteOutputs(await portraitRun(warpingPortrait, 'warpingSpade', [
        {name: 'feature_3d', type: 'float32', shape: warpingFeature.shape, data: warpingFeature.data},
        {name: 'kp_driving', type: 'float32', shape: [1, 21, 3], data: stitchedKp},
        {name: 'kp_source', type: 'float32', shape: [1, 21, 3], data: sourceCanonicalKp},
      ]), 'warpingSpade');
      const generated = unpackWarpedRgb(modelOutput(warped, 'out'));
      if (!pasteMap) pasteMap = createPasteMap(width, height, cropCenterX, cropCenterY, cropSide, generated.width, generated.height);
      fs.writeSync(outputFd, pasteBackWithMap(source, generated, width, height, pasteMap));
      if (typeof options.onFrame === 'function') options.onFrame(frame + 1, frameCount);
    }
  } finally {
    fs.closeSync(outputFd);
  }
  return {
    width,
    height,
    fps,
    frames: frameCount,
    duration: frameCount / fps,
    provider,
    profile: pipeline.profile || options.profile || 'original',
    diffusionSteps: joy.nDiffSteps,
    face: {score: face.score, bbox: face.bbox},
    audioSamples: audioSamples.length,
  };
}

module.exports = {
  TALKING_VIDEO_DEFAULTS,
  createTalkingVideoOptions,
  createTalkingVideoPipeline,
  renderTalkingVideo,
};
