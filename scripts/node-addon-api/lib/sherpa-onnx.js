/** @typedef {import('./types').WaveObject} WaveObject */
/**
 * @typedef {import('./types').OnlineRecognizerResult} OnlineRecognizerResult
 */
/**
 * @typedef {import('./types').OfflineRecognizerResult} OfflineRecognizerResult
 */

const addon = require('./addon.js')
const streaming_asr = require('./streaming-asr.js');
const non_streaming_asr = require('./non-streaming-asr.js');
const non_streaming_tts = require('./non-streaming-tts.js');
const vad = require('./vad.js');
const slid = require('./spoken-language-identification.js');
const sid = require('./speaker-identification.js');
const at = require('./audio-tagg.js');
const punct = require('./punctuation.js');
const kws = require('./keyword-spotter.js');
const sd = require('./non-streaming-speaker-diarization.js');
const speech_denoiser = require('./non-streaming-speech-denoiser.js');
const online_speech_denoiser = require('./online-speech-denoiser.js');
const resampler = require('./resampler.js');
const face = require('./face.js');
const faster_live_portrait = require('./faster-live-portrait.js');
const joyvasa = require('./joyvasa.js');
const talking_video = require('./talking-video.js');

module.exports = {
  OnlineRecognizer : streaming_asr.OnlineRecognizer,
  OfflineRecognizer : non_streaming_asr.OfflineRecognizer,
  OfflineTts : non_streaming_tts.OfflineTts,
  GenerationConfig : non_streaming_tts.GenerationConfig,
  readWave : addon.readWave,
  writeWave : addon.writeWave,
  Display : streaming_asr.Display,
  Vad : vad.Vad,
  CircularBuffer : vad.CircularBuffer,
  SpokenLanguageIdentification : slid.SpokenLanguageIdentification,
  SpeakerEmbeddingExtractor : sid.SpeakerEmbeddingExtractor,
  SpeakerEmbeddingManager : sid.SpeakerEmbeddingManager,
  AudioTagging : at.AudioTagging,
  OfflinePunctuation : punct.OfflinePunctuation,
  OnlinePunctuation : punct.OnlinePunctuation,
  KeywordSpotter : kws.KeywordSpotter,
  OfflineSpeakerDiarization : sd.OfflineSpeakerDiarization,
  OfflineSpeechDenoiser : speech_denoiser.OfflineSpeechDenoiser,
  OnlineSpeechDenoiser : online_speech_denoiser.OnlineSpeechDenoiser,
  LinearResampler : resampler.LinearResampler,
  FaceDetector : face.FaceDetector,
  // Deprecated compatibility alias. The default bundle uses MediaPipe,
  // not RetinaFace.
  RetinaFaceDetector : face.RetinaFaceDetector,
  AuraFaceRecognizer : face.AuraFaceRecognizer,
  FaceIdentityTracker : face.FaceIdentityTracker,
  faceCosineSimilarity : face.faceCosineSimilarity,
  faceSamePerson : face.faceSamePerson,
  FACE_DETECTOR_DEFAULTS : face.FACE_DETECTOR_DEFAULTS,
  createFaceDetectorConfig : face.createFaceDetectorConfig,
  FasterLivePortrait : faster_live_portrait.FasterLivePortrait,
  MediaPipeLandmarks : faster_live_portrait.MediaPipeLandmarks,
  imageToTensor : faster_live_portrait.imageToTensor,
  mediaPipeLandmarksToTensor : faster_live_portrait.mediaPipeLandmarksToTensor,
  JoyVASA : joyvasa.JoyVASA,
  TALKING_VIDEO_DEFAULTS : talking_video.TALKING_VIDEO_DEFAULTS,
  createTalkingVideoOptions : talking_video.createTalkingVideoOptions,
  createTalkingVideoPipeline : talking_video.createTalkingVideoPipeline,
  renderTalkingVideo : talking_video.renderTalkingVideo,
  version : addon.version,
  gitSha1 : addon.gitSha1,
  gitDate : addon.gitDate,
  onnxruntimeVersion : addon.onnxruntimeVersion,
}
