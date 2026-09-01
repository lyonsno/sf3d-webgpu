import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { PassThrough } from 'node:stream';

import { validateDinoWeightRepresentationAssay } from './dino_weight_representation_assay_contract.mjs';
import { createAssayWeightMiddleware } from './vite_dino_assay_config.mjs';

const source = {
  commit: '5a161a06c32ac9fe1ade001dd0d558ef355f21f9',
  kitVersion: '0.1.47',
  kitTarballSha256: '6d1fd5d778fa5c964393b5f53717c08c118c9a42720ce00aea3d888b2b587535',
  weightsSha256: '0e5c23c8c502492c0b4432006ac30f2d11b2a4c102a3bf9f21c2fcf094ddbffd',
  inputSha256: '3220e5ef5d598ae99f303834358d36e89d5a7dc73e6584b62e30de80c22eb624',
};

function arm(representation) {
  const packed = representation === 'f16-packed-u32';
  return {
    requestedRepresentation: representation,
    effectiveRepresentation: representation,
    browser: 'Chrome/152.0.7977.65',
    adapter: { vendor: 'apple', architecture: 'metal' },
    pageErrors: [],
    matrixStats: {
      representation,
      tensorCount: 192,
      sourceByteLength: 754_974_720,
      storageByteLength: packed ? 754_974_720 : 1_509_949_440,
      expandedFp32ByteLength: 1_509_949_440,
      savedVsExpandedFp32ByteLength: packed ? 754_974_720 : 0,
    },
    cooperative: {
      status: 'succeeded',
      completedItems: 24,
      totalItems: 24,
    },
    output: {
      shape: { N: 1297, dim: 1024 },
      elementCount: 1_328_128,
      byteLength: 5_312_512,
      finiteCount: 1_328_128,
      sha256: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    },
  };
}

function acceptedReport() {
  return {
    schema: 'sf3d.dino-weight-representation-browser-assay.v0',
    ok: true,
    phase: 'complete',
    source: { ...source },
    arms: {
      'f32-expanded': arm('f32-expanded'),
      'f16-packed-u32': arm('f16-packed-u32'),
    },
    comparison: {
      byteIdentical: true,
      firstDifferingIndex: null,
      maxAbsDelta: 0,
    },
  };
}

assert.doesNotThrow(() => validateDinoWeightRepresentationAssay(acceptedReport(), source));

for (const mutate of [
  report => { report.arms['f16-packed-u32'].effectiveRepresentation = 'f32-expanded'; },
  report => { report.arms['f16-packed-u32'].cooperative.completedItems = 23; },
  report => { report.source.weightsSha256 = 'stale'; },
  report => { report.arms['f16-packed-u32'].output.finiteCount--; },
  report => { report.comparison.byteIdentical = false; },
  report => { report.arms['f16-packed-u32'].matrixStats.tensorCount = 191; },
]) {
  const report = acceptedReport();
  mutate(report);
  assert.throws(() => validateDinoWeightRepresentationAssay(report, source));
}

console.log('DINO weight-representation assay false-closure contracts passed');

const servedBytes = Buffer.from('authenticated-weight-bytes');
const temporaryDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'sf3d-dino-assay-weight-'));
const weightPath = path.join(temporaryDirectory, 'weights.bin');
fs.writeFileSync(weightPath, servedBytes);
const middleware = createAssayWeightMiddleware(weightPath);
const response = new PassThrough();
const headers = new Map();
response.setHeader = (name, value) => headers.set(name.toLowerCase(), String(value));
const chunks = [];
response.on('data', chunk => chunks.push(chunk));
const finished = new Promise(resolve => response.on('finish', resolve));
middleware({ url: '/weights.bin?cache-bust=1' }, response, () => {
  throw new Error('weight request must not fall through');
});
await finished;
assert.equal(headers.get('content-length'), String(servedBytes.length));
assert.deepEqual(Buffer.concat(chunks), servedBytes);

let fellThrough = false;
middleware({ url: '/src/main.js' }, new PassThrough(), () => { fellThrough = true; });
assert.equal(fellThrough, true);
fs.rmSync(temporaryDirectory, { recursive: true, force: true });

console.log('DINO assay authenticated weight middleware contracts passed');
