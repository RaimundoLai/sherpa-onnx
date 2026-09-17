'use strict';

const fs = require('fs');
const path = require('path');
const {readWave} = require('../lib/addon.js');
const {JoyVASA} = require('../lib/joyvasa.js');

function usage() {
  const command = 'node --napi-modules ./test/test_joyvasa.js <joyvasa.json> <audio.wav>';
  console.error(`Usage: ${command}`);
  process.exitCode = 2;
}

const metadata = process.argv[2];
const audioPath = process.argv[3];
if (!metadata || !audioPath) {
  usage();
} else if (!fs.existsSync(metadata) || !fs.existsSync(audioPath)) {
  throw new Error('JoyVASA metadata or audio file does not exist');
} else {
  const wave = readWave(path.resolve(audioPath));
  const joy = new JoyVASA({metadata: path.resolve(metadata)});
  const samplesPerWindow = joy.samplesPerWindow;
  const samples = wave.samples.length > samplesPerWindow
    ? wave.samples.slice(0, samplesPerWindow)
    : wave.samples;
  const result = joy.sample(samples, {sampleRate: wave.sampleRate, cfg: false});
  const expected = joy.nMotions * joy.motionFeatDim;
  if (result.motion.length !== expected) throw new Error('Unexpected JoyVASA motion shape');
  let absSum = 0;
  for (const value of result.motion) {
    if (!Number.isFinite(value)) throw new Error('JoyVASA produced a non-finite motion value');
    absSum += Math.abs(value);
  }
  if (absSum === 0) throw new Error('JoyVASA produced an all-zero motion result');
  console.log('JoyVASA test passed', {
    sampleRate: wave.sampleRate,
    inputSamples: samples.length,
    motionShape: result.motionShape,
    absSum: absSum.toFixed(3),
  });
}
