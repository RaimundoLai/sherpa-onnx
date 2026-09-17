# Introduction

This folder contains `node-addon-api` wrapper for `sherpa-onnx`.

Caution: This folder is for developer only.

## Usage

```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=./install -DBUILD_SHARED_LIBS=ON ..
make -j install
export PKG_CONFIG_PATH=$PWD/install:$PKG_CONFIG_PATH
cd ../scripts/node-addon-api/
npm i
./node_modules/.bin/cmake-js compile --log-level verbose

# see test/test_asr_streaming_transducer.js
# for usages
```

Please see doc at <https://k2-fsa.github.io/sherpa/onnx/javascript-api/index.html>.

API doc can be found at <https://k2-fsa.github.io/sherpa/onnx/javascript-api/html/index.html>

## Face detection and recognition

The addon exposes an ONNX MediaPipe face detector and AuraFace primitives. Image codecs are
intentionally outside the addon: pass a packed `Buffer` or `Uint8Array` with
the image metadata.

```js
const {
  FaceDetector,
  AuraFaceRecognizer,
  FaceIdentityTracker,
  faceCosineSimilarity,
  faceSamePerson,
} = require('sherpa-onnx-node-addon-api');

const detector = new FaceDetector({
  model: './mediapipe/float/face_detector.onnx',
  landmarkModel: './mediapipe/float/face_landmark_detector.onnx',
  provider: 'cpu',
  scoreThreshold: 0.5,
});
const recognizer = new AuraFaceRecognizer({
  // Use glintr100.onnx from fal/AuraFace-v1.
  model: './glintr100.onnx',
  provider: 'cpu',
});

const frame = {
  data: rgbPixels,       // Uint8Array or Buffer, interleaved rows
  width: 1280,
  height: 720,
  channels: 3,
  format: 'rgb',         // rgb, bgr, rgba or bgra
};

// In Electron, use the worker-backed variants so detection and embedding do
// not block the main/renderer event loop.
const faces = await detector.detectAsync(frame);
const tracker = new FaceIdentityTracker({threshold: 0.45});
for (const face of faces) {
  face.embedding = await recognizer.computeAsync(frame, face);
  face.identityId = tracker.match(face.embedding);
}

// The UI can now select identityId values and draw a box around face.bbox.
// This addon only detects and identifies faces; it does not alter pixels.
```

`FaceDetector` accepts the MediaPipe Face Detection ONNX export with four output
tensors (split box coordinates and scores). When `landmarkModel` is supplied,
the companion MediaPipe Face Mesh graph refines the five points used by the
recognizer and talking-video pipeline. Legacy three-output RetinaFace and
nine-output TensorFlow/SCRFD graphs remain supported only for compatibility;
the default model paths no longer select them.
`AuraFaceRecognizer` performs ArcFace five-point
alignment when detector landmarks are provided and returns an L2-normalized
embedding; `faceCosineSimilarity(a, b)` can be used for a direct pairwise
comparison, or `faceSamePerson(a, b, threshold)` for a boolean result. The
default threshold is only a starting point and should be calibrated against
the application's camera and false-match requirements.

The talking-video bridge resolves `face_detector.onnx` and its companion
`face_landmark_detector.onnx` from the canonical model directory, or accepts
explicit `SHERPA_ONNX_FACE_DETECTOR_MODEL` and
`SHERPA_ONNX_FACE_LANDMARK_MODEL` paths. It does not select SCRFD or
`retinaface_det_static.onnx`.

## FasterLivePortrait ONNX model set

FasterLivePortrait exports several independent ONNX graphs. The addon can load
the graphs together, inspect their input/output contracts, and run one graph
with named tensors. This keeps the model-set layer independent from image
codecs and allows the JavaScript application to compose source preparation,
motion, stitching, and rendering in the same order as the upstream pipeline.

```js
const {
  FasterLivePortrait,
  imageToTensor,
  MediaPipeLandmarks,
} = require('sherpa-onnx-node-addon-api');

const portrait = new FasterLivePortrait({
  provider: 'cpu',
  numThreads: 4,
  models: {
    appearanceFeatureExtractor: './checkpoints/liveportrait_onnx/appearance_feature_extractor.onnx',
    motionExtractor: './checkpoints/liveportrait_onnx/motion_extractor.onnx',
    landmark: './checkpoints/liveportrait_onnx/landmark.onnx',
    warpingSpade: './checkpoints/liveportrait_onnx/warping_spade.onnx',
    stitching: './checkpoints/liveportrait_onnx/stitching.onnx',
    stitchingEyeRetarget: './checkpoints/liveportrait_onnx/stitching_eye.onnx',
    stitchingLipRetarget: './checkpoints/liveportrait_onnx/stitching_lip.onnx',
  },
});

// Always inspect this first: model exports can have different I/O names.
console.log(portrait.getModelInfo());

// Crop/resize the face before calling this helper. The standard appearance
// and motion exports expect [1, 3, 256, 256].
const faceCrop = {
  data: croppedRgb256,
  width: 256,
  height: 256,
  channels: 3,
  format: 'rgb',
};
const outputs = portrait.runImage('motionExtractor', faceCrop);

// MediaPipe Face Mesh results can be passed as pixel-coordinate tensors to
// later crop/landmark code. This helper does not load MediaPipe itself.
const faceLandmarks = new MediaPipeLandmarks(mediaPipeResult, {
  width: frame.width,
  height: frame.height,
  normalized: true,
});
console.log(faceLandmarks.bbox(), faceLandmarks.toTensor());
```

`detect()` and `run()` are low-level synchronous ONNX Runtime calls. For
Electron or other event-loop applications, use `await detectAsync()`,
`await runAsync()` / `await runImageAsync()`:
they copy the typed-array inputs and execute the graph in an N-API worker
thread, so CoreML/CUDA/CPU inference does not block the JS thread. Each input
is an object with `data`, `shape`, and optional ONNX `name`; output tensors
contain the output name, type, shape, and a copied typed array. `imageToTensor()` converts
packed RGB/BGR pixels to the normalized NCHW tensor used by the upstream
appearance and motion models; it does not crop or resize the image.

The MediaPipe helper accepts the landmark points produced by MediaPipe Face
Mesh (including `faceLandmarks`/`multiFaceLandmarks` result objects and
`{x, y, z}` points) and converts normalized coordinates to source-image
pixels. It uses the first detected face. MediaPipe is intentionally an input
boundary here; a MediaPipe runtime is not linked into this native addon.

### Original 與 Metal profile

Node bridge 將 warping 模型分成兩個 profile。`original` 使用上游完整的
`warping_spade.onnx`，作為品質基準；`metal` 使用獨立的
`warping_spade_fp16.onnx`，不會覆蓋或改名原版模型。Metal FP16 graph
仍然輸出 512x512，畫質比 256px graph 穩定，但是否更快取決於 macOS
與 CoreML 版本，請實機比較。

設定 Metal 模型路徑：

```bash
export SHERPA_ONNX_FLP_METAL_WARPING_MODEL=/path/to/metal/warping_spade_fp16.onnx
```

在 Web 測試頁選擇「原版」或「Metal/CoreML FP16」。Node API 也可傳入
`profile: 'original'` 或 `profile: 'metal'`。若未指定，預設為 `original`。

### Native MLX port (Apple Silicon)

MLX 的 Python bridge 只保留作為開發參考，不再是 Electron/MAS 的執行依賴。
目前 native layer 已完成：Swift/Cmlx framework ABI、ZIP64 `.npz` 讀取、
非同步 Rust/N-API worker，以及 FasterLivePortrait 的
appearance → dense-motion → warping → occlusion → SPADE 單幀 renderer。
App 的 native MLX 路徑會使用既有 ONNX motion extractor 與 stitching 作為
LivePortrait 的 head-pose/相容層，並以 Swift/Cmlx 原生執行完整的 JoyVASA
HuBERT audio encoder 與 diffusion motion generator；不再透過 Python 或
ONNX JoyVASA motion graph。長音訊會以 4 秒 window 串接，且每個 window
及每個輸出 frame 都清理 MLX temporary cache。
`mlxNativeIsAvailable()` 代表 framework 存在，
`mlxNativeRendererIsAvailable()` 則代表完整 native frame renderer 已連結；
沒有 framework 的 ONNX/CUDA/CoreML build 仍不會誤啟動 Python。

Native MLX build 需要 macOS 14 SDK、Metal Toolchain，以及 Xcode 建立的
`MLX.framework`、`MLXNN.framework`、`Cmlx.framework`。這些 framework 必須
和 native addon 一起簽名並放入 App bundle；不能只把 `.node` 複製進去：

```bash
export SHERPA_ONNX_MLX_FRAMEWORK_DIR=/Applications/workspace/build_space/Transcribe/extraResources/mlx-frameworks
export SHERPA_ONNX_MLX_NUMERICS_INCLUDE=/Applications/workspace/build_space/Transcribe/third_party/mlx-swift-numerics/include
npm run build --prefix /path/to/Transcribe/rust-binding -- --release
```

`SHERPA_ONNX_MLX_FRAMEWORK_DIR` is a build-time source directory. The
packaged addon resolves the frameworks from the App's
`Contents/Resources/extraResources/mlx-frameworks` directory; it never needs
`/private/tmp` at runtime. `afterPack.js` also detects whether the selected
Rust addon actually links MLX before copying these frameworks into a package.

這個 port 的 N-API 推論都應使用 Promise，例如
`FasterLivePortrait.runAsync()` 和 `JoyVASA.generateMotionSequenceAsync()`；
ONNX 的原版同步方法仍保留給既有低階呼叫者。模型權重仍須依各自來源授權，
不能因參考程式碼是 MIT 就把所有 `.npz` 權重標成 MIT。

### Optional 256px warping graph

The published `warping_spade.onnx` always computes a 512px portrait crop even
when the final video is smaller. On Apple Silicon this is usually the dominant
cost. The repository includes a shape-only converter that reuses the published
weights, changes the feature grid from 64x64 to 32x32, and produces a 256x256
crop. It also rebuilds the dense-motion sampling lattice, so the generated
graph remains a valid ONNX model:

```bash
uv run --with onnx --with numpy python scripts/fasterliveportrait/optimize_warping.py \
  /path/to/liveportrait_onnx/warping_spade.onnx \
  /path/to/liveportrait_onnx/warping_spade_256.onnx
```

The talking-video bridge can use the smaller graph without replacing the
original model:

```bash
export SHERPA_ONNX_FLP_WARPING_MODEL=/path/to/liveportrait_onnx/warping_spade_256.onnx
```

The bridge automatically downsamples the cached appearance feature to the
graph's declared `feature_3d` shape and accepts either 512px or 256px warping
outputs. This is a speed/quality trade-off: the 256px path is softer, so it is
only suitable for quick previews and is not selected by either profile.

## JoyVASA audio-to-motion ONNX

JoyVASA is exposed as a second-stage audio-to-motion API. First export the
audio encoder and one-step diffusion denoiser with the isolated `uv` runtime:

```bash
uv run --python .venv-joyvasa/bin/python scripts/joyvasa/export_onnx.py \
  --source-root /path/to/FasterLivePortrait \
  --motion-checkpoint /path/to/motion_generator_hubert_chinese.pt \
  --audio-model-path /path/to/hubert-base-ls960 \
  --output-dir ./joyvasa-onnx --verify
```

The source checkout and model weights are explicit inputs; the exporter does
not silently download large checkpoints. It writes `audio_encoder.onnx`,
`motion_generator.onnx`, and `joyvasa.json`. The current official JoyVASA
checkpoint uses HuBERT-style 768-dimensional hidden states, projected to the
motion model's feature dimension. Whisper/Crisp ASR embeddings are therefore
not drop-in replacements without a trained projection or a retrained motion
generator.

```js
const {JoyVASA} = require('sherpa-onnx-node-addon-api');
const joy = new JoyVASA({metadata: './joyvasa-onnx/joyvasa.json'});

// A single window is roughly 4 seconds for the default 100 frames @ 25 fps.
const motion = joy.sample(audio16kFloat32);
console.log(motion.motionShape, motion.motion);

// For a complete clip, this carries the previous motion/audio context.
const sequence = joy.generateMotionSequence(audio16kFloat32);
```

To run the included formal-model smoke test with a WAV file:

```bash
DYLD_LIBRARY_PATH=/path/to/sherpa-onnx/build/install/lib \
  npm run test:joyvasa -- \
  /path/to/joyvasa-onnx/joyvasa.json \
  /path/to/audio.wav
```

`sample()` runs the checkpoint-configured diffusion schedule in JavaScript
(the current Hugging Face checkpoint uses 50 steps) and invokes the ONNX
denoiser for each step, including audio classifier-free guidance. This keeps
stochastic sampling and control flow outside ONNX while preserving the
model's deterministic subgraphs. The returned motion coefficients are the
input for the separate FasterLivePortrait rendering stage; this API does not
perform text-to-speech or video encoding.

## HTML talking-video test page

The native addon cannot be loaded directly by a browser. The included local
Node bridge accepts an image and audio upload, decodes them with ffmpeg, runs
MediaPipe + JoyVASA + FasterLivePortrait through the addon, and returns an
H.264/AAC MP4 to the page:

```bash
cd scripts/node-addon-api
DYLD_LIBRARY_PATH=$PWD/../../build/install/lib \
  npm run start:talking-video
```

Open <http://127.0.0.1:8787> (or open `web/index.html` directly after starting
the server). On macOS the talking-video pipeline defaults to the `coreml`
provider (Apple CoreML/Metal where supported); set
`SHERPA_ONNX_TALKING_PROVIDER=cpu` to force CPU. The page defaults to `0`, which processes the complete audio;
enter `4` in “最長秒數” only when a quick test is wanted. The page defaults to 50 JoyVASA diffusion steps;
enter 20 for a faster, lower-cost test. The output long edge defaults to 512 pixels; enter `0` to preserve
the source resolution. Model sessions and source-image features are reused across requests. Output FPS defaults to
25; using 12.5 renders half as many portrait frames while keeping the audio duration synchronized. Multiple JoyVASA
windows are joined for longer audio; duration is not intrinsically limited to four seconds. The
bridge defaults to the model directories used by the local test, and accepts
`SHERPA_ONNX_FLP_MODEL_DIR`, `SHERPA_ONNX_JOYVASA_METADATA`, and
`SHERPA_ONNX_JOYVASA_TEMPLATE` to point to another model installation. Set
`SHERPA_ONNX_FLP_WARPING_MODEL` to opt into the generated 256px warping graph.
