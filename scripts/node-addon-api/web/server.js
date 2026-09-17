'use strict';

const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');
const {execFileSync} = require('child_process');
const {readWave} = require('../lib/addon.js');
const {createTalkingVideoPipeline, renderTalkingVideo} = require('../lib/talking-video.js');

const root = __dirname;
const port = Number(process.env.SHERPA_ONNX_TALKING_PORT || 8787);
const maxBodyBytes = Number(process.env.SHERPA_ONNX_TALKING_MAX_BODY || 80 * 1024 * 1024);
const scribisModels = process.env.SHERPA_ONNX_MODELS_DIR ||
  path.join(os.homedir(), 'Library', 'Application Support', 'scribis', '_models');
const avatarOnnxDir = path.join(scribisModels, 'avatar', 'onnx');
const modelDir = process.env.SHERPA_ONNX_FLP_MODEL_DIR ||
  path.join(avatarOnnxDir, 'liveportrait_onnx');
const profile = process.env.SHERPA_ONNX_TALKING_PROFILE || 'original';
const backend = process.env.SHERPA_ONNX_TALKING_BACKEND || 'onnx';
const faceDetectorModel = process.env.SHERPA_ONNX_FACE_DETECTOR_MODEL ||
  path.join(avatarOnnxDir, 'mediapipe', 'float', 'face_detector.onnx');
const faceLandmarkModel = process.env.SHERPA_ONNX_FACE_LANDMARK_MODEL ||
  path.join(avatarOnnxDir, 'mediapipe', 'float', 'face_landmark_detector.onnx');
const warpingSpadeModel = process.env.SHERPA_ONNX_FLP_WARPING_MODEL;
const originalWarpingSpadeModel = process.env.SHERPA_ONNX_FLP_ORIGINAL_WARPING_MODEL;
const metalWarpingSpadeModel = process.env.SHERPA_ONNX_FLP_METAL_WARPING_MODEL;
const joyMetadata = process.env.SHERPA_ONNX_JOYVASA_METADATA || path.join(avatarOnnxDir, 'joyvasa.json');
const joyTemplate = process.env.SHERPA_ONNX_JOYVASA_TEMPLATE || path.join(avatarOnnxDir, 'joyvasa-template.json');
const provider = process.env.SHERPA_ONNX_TALKING_PROVIDER || (process.platform === 'darwin' ? 'coreml' : 'cpu');
const maxDimension = Number(process.env.SHERPA_ONNX_TALKING_MAX_DIM || 512);
const jobs = new Map();
const pipelines = new Map();

function getPipeline(body) {
  const requestedProvider = body.provider || provider;
  const requestedProfile = body.profile || profile;
  const requestedSteps = body.diffusionSteps === undefined ? 'metadata' : Number(body.diffusionSteps);
  const key = `${requestedProvider}:${requestedProfile}:${requestedSteps}`;
  let pipeline = pipelines.get(key);
  if (!pipeline) {
    pipeline = createTalkingVideoPipeline({
      modelDir,
      joyvasaMetadata: joyMetadata,
      provider: requestedProvider,
      profile: requestedProfile,
      numThreads: 2,
      nDiffSteps: requestedSteps === 'metadata' ? undefined : requestedSteps,
      originalWarpingSpadeModel,
      metalWarpingSpadeModel,
      // The legacy generic override is only allowed to select the Metal
      // profile. It must not leak a stale FP16 path into original/CoreML.
      warpingSpadeModel: requestedProfile === 'metal' ? warpingSpadeModel : undefined,
      faceDetectorModel,
      faceLandmarkModel,
    });
    pipelines.set(key, pipeline);
  }
  return pipeline;
}

function jsonResponse(response, status, value) {
  const body = JSON.stringify(value);
  response.writeHead(status, {
    'Access-Control-Allow-Origin': '*',
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
  });
  response.end(body);
}

function readBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    request.on('data', (chunk) => {
      size += chunk.length;
      if (size > maxBodyBytes) {
        reject(new Error(`request body exceeds ${maxBodyBytes} bytes`));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    request.on('error', reject);
  });
}

function run(command, args) {
  return execFileSync(command, args, {stdio: 'pipe'});
}

function writeUpload(directory, upload, fallbackName) {
  if (!upload || typeof upload.data !== 'string') throw new Error(`${fallbackName} upload is missing data`);
  const extension = path.extname(String(upload.name || '')).toLowerCase();
  const allowed = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac']);
  const filename = `${fallbackName}${allowed.has(extension) ? extension : '.bin'}`;
  const output = path.join(directory, filename);
  fs.writeFileSync(output, Buffer.from(upload.data, 'base64'));
  return output;
}

function probeImage(file) {
  const output = run('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=x', file]).toString().trim();
  const match = /^(\d+)x(\d+)$/.exec(output);
  if (!match) throw new Error('could not read image dimensions');
  return {width: Number(match[1]), height: Number(match[2])};
}

async function generateJob(body) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'sherpa-onnx-talking-'));
  const imageInput = writeUpload(directory, body.image, 'image');
  const audioInput = writeUpload(directory, body.audio, 'audio');
  const dimensions = probeImage(imageInput);
  const sourceRgb = path.join(directory, 'source.rgb');
  const audioWav = path.join(directory, 'audio-16k.wav');
  const outputRaw = path.join(directory, 'output.rgb');
  const outputMp4 = path.join(directory, 'output.mp4');
  const requestedBackend = body.backend || backend;
  run('ffmpeg', ['-y', '-loglevel', 'error', '-i', imageInput, '-f', 'rawvideo', '-pix_fmt', 'rgb24', sourceRgb]);
  run('ffmpeg', ['-y', '-loglevel', 'error', '-i', audioInput, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', audioWav]);
  // Ask the addon for an owned copy; external ArrayBuffers are rejected by
  // Electron's V8 context and are not safe to carry across the async render.
  const wave = readWave(audioWav, false);
  const result = await renderTalkingVideo({
    sourceRgb: fs.readFileSync(sourceRgb),
    width: dimensions.width,
    height: dimensions.height,
    audioSamples: wave.samples,
    audioSampleRate: wave.sampleRate,
    modelDir,
    joyvasaMetadata: joyMetadata,
    joyvasaTemplate: joyTemplate,
    provider: body.provider || provider,
    backend: requestedBackend,
    profile: body.profile || profile,
    originalWarpingSpadeModel,
    metalWarpingSpadeModel,
    pipeline: requestedBackend === 'mlx' ? undefined : getPipeline(body),
    faceDetectorModel,
    faceLandmarkModel,
    nDiffSteps: body.diffusionSteps === undefined ? undefined : Number(body.diffusionSteps),
    maxDimension: body.maxDimension === undefined ? maxDimension : Number(body.maxDimension),
    outputFps: body.outputFps === undefined ? 25 : Number(body.outputFps),
    outputRaw,
    maxSeconds: body.maxSeconds === undefined ? 0 : Number(body.maxSeconds),
    sourceImagePath: imageInput,
    audioPath: audioWav,
    mlxReferenceDir: body.mlxReferenceDir,
    mlxWeightsDir: body.mlxWeightsDir,
    mlxPython: body.mlxPython,
    mlxProfile: body.mlxProfile,
    mlxMotionBackend: body.mlxMotionBackend,
    cfgScale: body.cfgScale === undefined ? undefined : Number(body.cfgScale),
    cfg: body.cfg === true,
    onFrame: (frame, frames) => {
      if (frame === 1 || frame === frames || frame % 25 === 0) console.log(`talking-video ${frame}/${frames}`);
    },
  });
  run('ffmpeg', [
    '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
    '-video_size', `${result.width}x${result.height}`, '-framerate', String(result.fps), '-i', outputRaw,
    '-i', audioWav, '-map', '0:v:0', '-map', '1:a:0', '-frames:v', String(result.frames),
    '-t', result.duration.toFixed(3), '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-shortest', outputMp4,
  ]);
  const id = crypto.randomUUID();
  jobs.set(id, {file: outputMp4, directory});
  return {...result, url: `/api/video/${id}`};
}

function serveVideo(request, response, id) {
  const job = jobs.get(id);
  if (!job || !fs.existsSync(job.file)) return jsonResponse(response, 404, {error: 'video not found'});
  const stat = fs.statSync(job.file);
  const range = request.headers.range;
  if (!range) {
    response.writeHead(200, {'Access-Control-Allow-Origin': '*', 'Content-Type': 'video/mp4', 'Content-Length': stat.size, 'Accept-Ranges': 'bytes'});
    return fs.createReadStream(job.file).pipe(response);
  }
  const match = /^bytes=(\d*)-(\d*)$/.exec(range);
  if (!match) return jsonResponse(response, 416, {error: 'invalid range'});
  const start = match[1] ? Number(match[1]) : 0;
  const end = match[2] ? Math.min(Number(match[2]), stat.size - 1) : stat.size - 1;
  if (start > end || start >= stat.size) return jsonResponse(response, 416, {error: 'range not satisfiable'});
  response.writeHead(206, {'Access-Control-Allow-Origin': '*', 'Content-Type': 'video/mp4', 'Content-Length': end - start + 1, 'Content-Range': `bytes ${start}-${end}/${stat.size}`, 'Accept-Ranges': 'bytes'});
  return fs.createReadStream(job.file, {start, end}).pipe(response);
}

const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, `http://${request.headers.host || 'localhost'}`);
    if (request.method === 'OPTIONS') {
      response.writeHead(204, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
      });
      return response.end();
    }
    if (request.method === 'GET' && url.pathname === '/') {
      const html = fs.readFileSync(path.join(root, 'index.html'));
      response.writeHead(200, {'Access-Control-Allow-Origin': '*', 'Content-Type': 'text/html; charset=utf-8', 'Content-Length': html.length});
      return response.end(html);
    }
    if (request.method === 'GET' && url.pathname.startsWith('/api/video/')) {
      return serveVideo(request, response, url.pathname.slice('/api/video/'.length));
    }
    if (request.method === 'POST' && url.pathname === '/api/generate') {
      const body = JSON.parse(await readBody(request));
      return jsonResponse(response, 200, await generateJob(body));
    }
    return jsonResponse(response, 404, {error: 'not found'});
  } catch (error) {
    console.error(error.stack || error);
    return jsonResponse(response, 500, {error: error.message || String(error)});
  }
});

server.listen(port, '127.0.0.1', () => {
  console.log(`Sherpa-ONNX talking video test: http://127.0.0.1:${port}`);
  console.log(`FLP models: ${modelDir}`);
  console.log(`JoyVASA metadata: ${joyMetadata}`);
  console.log(`Talking backend: ${backend}`);
  console.log(`ONNX provider: ${provider}`);
  console.log(`Talking profile: ${profile} (original or metal)`);
});
