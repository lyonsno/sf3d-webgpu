import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  assertPackedFp16LinearShaderContract,
  createPackedFp16LinearFixture,
  createTransposedPackedFp16LinearFixture,
  runLinearReference,
} from '../src/lib/packed_fp16_linear_assay.js';
import {
  createPackedFp16LinearAdmissionFailureReport,
  evaluatePackedFp16LinearSmokeReport,
} from '../src/lib/packed_fp16_linear_report.js';
import { packFp16WeightsToU32 } from '@kaminos/webgpu-inference-kit';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const hashOutput = output => crypto.createHash('sha256')
  .update(Buffer.from(new Float32Array(output).buffer))
  .digest('hex');
const shader = fs.readFileSync(path.join(root, 'src/shaders/linear_f16_packed.wgsl'), 'utf8');
assertPackedFp16LinearShaderContract(shader);

const fixture = createPackedFp16LinearFixture({ rows: 3, inDim: 5, outDim: 7 });
assert.equal(fixture.plan.effectiveRepresentation, 'f16-packed-u32');
assert.equal(fixture.plan.valueLoadOperation, 'wgsl-unpack2x16float');
assert.equal(fixture.fp16Bits.length, 35);
assert.equal(fixture.expandedWeights.byteLength, 140);
assert.equal(fixture.packedWeights.byteLength, 72);
assert.equal(fixture.plan.storageByteLength, 72);
assert.equal(fixture.plan.savedVsExpandedFp32ByteLength, 68);

const oracleBits = new Uint16Array([0x3c00, 0xc000, 0x4200]);
const oraclePacked = packFp16WeightsToU32(oracleBits);
assert.deepEqual(
  [...oraclePacked],
  [0xc0003c00, 0x00004200],
  'independent literal words establish low/high lane and odd-tail packing',
);
const independentlyDecodedBits = [];
for (const word of new Uint32Array([0xc0003c00, 0x00004200])) {
  independentlyDecodedBits.push(word & 0xffff, word >>> 16);
}
assert.deepEqual(independentlyDecodedBits, [0x3c00, 0xc000, 0x4200, 0x0000]);

const expandedOutput = runLinearReference({
  input: fixture.input,
  weights: fixture.expandedWeights,
  bias: fixture.bias,
  rows: fixture.rows,
  inDim: fixture.inDim,
  outDim: fixture.outDim,
});
const unpackedOutput = runLinearReference({
  input: fixture.input,
  weights: fixture.unpackedWeights,
  bias: fixture.bias,
  rows: fixture.rows,
  inDim: fixture.inDim,
  outDim: fixture.outDim,
});
assert.deepEqual(unpackedOutput, expandedOutput);

const transposedFixture = createTransposedPackedFp16LinearFixture(fixture);
const transposedOutput = runLinearReference({
  input: fixture.input,
  weights: transposedFixture.expandedWeights,
  bias: fixture.bias,
  rows: fixture.rows,
  inDim: fixture.inDim,
  outDim: fixture.outDim,
  transposed: true,
});
assert.deepEqual(transposedOutput, expandedOutput, 'native and transposed storage must agree');
assert.notDeepEqual(transposedFixture.expandedWeights, fixture.expandedWeights);

assert.throws(
  () => createPackedFp16LinearFixture({ rows: 0, inDim: 5, outDim: 7 }),
  /positive integers/,
);
assert.throws(
  () => assertPackedFp16LinearShaderContract(shader.replace('unpack2x16float', 'fakeUnpack')),
  /unpack2x16float/,
);
assert.throws(
  () => assertPackedFp16LinearShaderContract(
    shader.replace('weight[index >> 1u]', 'weight[index]'),
  ),
  /packed word index/,
);
assert.throws(
  () => assertPackedFp16LinearShaderContract(
    shader.replace('select(pair.x, pair.y, (index & 1u) == 1u)', 'select(pair.y, pair.x, true)'),
  ),
  /packed lane selection/,
);

const admittedReport = {
  schema: 'sf3d.packed-fp16-linear-browser-smoke.v1',
  ok: true,
  effectiveBackend: 'webgpu',
  effectiveRepresentation: 'f16-packed-u32',
  source: { revision: '1'.repeat(40), dirtyPaths: [], dirtyDiffSha256: null },
  kit: {
    packageVersion: '0.1.46',
    producerRevision: '2'.repeat(40),
    tarballSha256: '3'.repeat(64),
    tarballManifestSha256: '4'.repeat(64),
    installedManifestSha256: '4'.repeat(64),
    sourceManifestSha256: '4'.repeat(64),
    producerRemote: 'https://example.invalid/lyonsno/kaminos.git',
    producerTree: '5'.repeat(40),
  },
  requested: { rows: 1, inDim: 2, outDim: 3, representation: 'f16-packed-u32' },
  browser: { version: 'Chrome/fixture', userAgent: 'fixture-agent' },
  adapter: { vendor: 'fixture-gpu', architecture: 'fixture-architecture' },
  control: {
    storageByteLength: 24,
    output: [1, 2, 3],
    outputF32Base64: Buffer.from(new Float32Array([1, 2, 3]).buffer).toString('base64'),
    outputFinite: true,
    outputSha256: hashOutput([1, 2, 3]),
  },
  candidate: {
    storageByteLength: 12,
    output: [1, 2, 3],
    outputF32Base64: Buffer.from(new Float32Array([1, 2, 3]).buffer).toString('base64'),
    outputFinite: true,
    outputSha256: hashOutput([1, 2, 3]),
  },
  comparison: { exact: true, maxAbsDiff: 0 },
  layoutProbe: {
    shape: { rows: 1, inDim: 2, outDim: 3 },
    reference: {
      output: [1, 2, 3],
      outputF32Base64: Buffer.from(new Float32Array([1, 2, 3]).buffer).toString('base64'),
      outputFinite: true,
      outputSha256: hashOutput([1, 2, 3]),
    },
    native: {
      transposed: false,
      control: {
        storageByteLength: 24,
        output: [1, 2, 3],
        outputF32Base64: Buffer.from(new Float32Array([1, 2, 3]).buffer).toString('base64'),
        outputFinite: true,
        outputSha256: hashOutput([1, 2, 3]),
      },
      candidate: {
        storageByteLength: 12,
        output: [1, 2, 3],
        outputF32Base64: Buffer.from(new Float32Array([1, 2, 3]).buffer).toString('base64'),
        outputFinite: true,
        outputSha256: hashOutput([1, 2, 3]),
      },
      comparison: { exact: true, maxAbsDiff: 0 },
      referenceComparison: { exact: true, maxAbsDiff: 0 },
    },
    transposed: {
      transposed: true,
      control: {
        storageByteLength: 24,
        output: [1, 2, 3],
        outputF32Base64: Buffer.from(new Float32Array([1, 2, 3]).buffer).toString('base64'),
        outputFinite: true,
        outputSha256: hashOutput([1, 2, 3]),
      },
      candidate: {
        storageByteLength: 12,
        output: [1, 2, 3],
        outputF32Base64: Buffer.from(new Float32Array([1, 2, 3]).buffer).toString('base64'),
        outputFinite: true,
        outputSha256: hashOutput([1, 2, 3]),
      },
      comparison: { exact: true, maxAbsDiff: 0 },
      referenceComparison: { exact: true, maxAbsDiff: 0 },
    },
    crossLayout: {
      control: { exact: true, maxAbsDiff: 0 },
      candidate: { exact: true, maxAbsDiff: 0 },
    },
  },
  terminal: { phase: 'complete', primaryOutputWritten: true },
};
const expectedIdentity = {
  sourceRevision: admittedReport.source.revision,
  kitPackageVersion: admittedReport.kit.packageVersion,
  kitProducerRevision: admittedReport.kit.producerRevision,
  kitTarballSha256: admittedReport.kit.tarballSha256,
  kitProducerRemote: admittedReport.kit.producerRemote,
  kitProducerTree: admittedReport.kit.producerTree,
  kitManifestSha256: admittedReport.kit.tarballManifestSha256,
};
admittedReport.requestedIdentity = structuredClone(expectedIdentity);
assert.equal(
  evaluatePackedFp16LinearSmokeReport(admittedReport, expectedIdentity).ok,
  true,
);

const falseReports = [
  ['truncated candidate output', report => { report.candidate.output = [1]; }],
  ['changed candidate output', report => { report.candidate.output[0] = 999; }],
  ['hidden nonfinite output', report => { report.candidate.output[0] = Number.NaN; }],
  ['forged output hashes', report => {
    report.control.outputSha256 = 'f'.repeat(64);
    report.candidate.outputSha256 = 'f'.repeat(64);
  }],
  ['wrong requested shape', report => { report.requested.rows = 9; }],
  ['smaller but incorrect candidate allocation', report => { report.candidate.storageByteLength = 20; }],
  ['dirty source', report => {
    report.source.dirtyPaths = ['src/shaders/linear_f16_packed.wgsl'];
    report.source.dirtyDiffSha256 = 'a'.repeat(64);
  }],
  ['invented package identity', report => {
    report.kit.packageVersion = '9.9.9';
    report.kit.producerRevision = '9'.repeat(40);
    report.kit.tarballSha256 = '8'.repeat(64);
  }],
  ['coherently forged package manifests', report => {
    report.kit.producerTree = '6'.repeat(40);
    report.kit.tarballManifestSha256 = '7'.repeat(64);
    report.kit.installedManifestSha256 = '7'.repeat(64);
    report.kit.sourceManifestSha256 = '7'.repeat(64);
  }],
  ['requested identity differs from the caller contract', report => {
    report.requestedIdentity.kitProducerRevision = '8'.repeat(40);
  }],
  ['blank adapter identity', report => { report.adapter = {}; }],
  ['missing transposed layout probe', report => { delete report.layoutProbe.transposed; }],
  ['changed transposed packed output', report => {
    report.layoutProbe.transposed.candidate.output[0] = 99;
  }],
];
for (const [label, mutate] of falseReports) {
  const report = structuredClone(admittedReport);
  mutate(report);
  assert.equal(
    evaluatePackedFp16LinearSmokeReport(report, expectedIdentity).ok,
    false,
    `${label} must not be admitted`,
  );
}

const fallback = structuredClone(admittedReport);
fallback.effectiveBackend = 'cpu-fallback';
assert.match(evaluatePackedFp16LinearSmokeReport(fallback, expectedIdentity).errors.join('\n'), /effectiveBackend/);

const missingBrowser = structuredClone(admittedReport);
delete missingBrowser.browser;
assert.match(evaluatePackedFp16LinearSmokeReport(missingBrowser, expectedIdentity).errors.join('\n'), /browser identity/);

const missingOutput = structuredClone(admittedReport);
delete missingOutput.candidate.output;
assert.match(evaluatePackedFp16LinearSmokeReport(missingOutput, expectedIdentity).errors.join('\n'), /candidate output/);

const falseParity = structuredClone(admittedReport);
falseParity.comparison.exact = false;
falseParity.comparison.maxAbsDiff = 0.25;
assert.match(
  evaluatePackedFp16LinearSmokeReport(falseParity, expectedIdentity).errors.join('\n'),
  /comparison declarations/,
);

const failedBeforeOutput = structuredClone(admittedReport);
failedBeforeOutput.ok = false;
failedBeforeOutput.terminal = { phase: 'dispatch', primaryOutputWritten: false };
assert.match(
  evaluatePackedFp16LinearSmokeReport(failedBeforeOutput, expectedIdentity).errors.join('\n'),
  /terminal success/,
);

const admissionFailure = createPackedFp16LinearAdmissionFailureReport(
  admittedReport,
  ['layoutProbe transposed output mismatch'],
);
assert.equal(admissionFailure.ok, false);
assert.equal(admissionFailure.failure.phase, 'admission');
assert.equal(admissionFailure.terminal.primaryOutputWritten, true);
assert.deepEqual(admissionFailure.layoutProbe.transposed.candidate.output, [1, 2, 3]);
assert.equal(admissionFailure.layoutProbe.transposed.candidate.outputF32Base64.length > 0, true);

const diagnosticReference = structuredClone(admittedReport);
diagnosticReference.layoutProbe.reference.output = [Math.fround(1.0001), 2, 3];
diagnosticReference.layoutProbe.reference.outputF32Base64 = Buffer.from(
  new Float32Array([1.0001, 2, 3]).buffer,
).toString('base64');
diagnosticReference.layoutProbe.reference.outputSha256 = hashOutput([1.0001, 2, 3]);
for (const layout of ['native', 'transposed']) {
  diagnosticReference.layoutProbe[layout].referenceComparison = {
    exact: false,
    maxAbsDiff: Math.abs(Math.fround(1.0001) - 1),
  };
}
const diagnosticAdmission = evaluatePackedFp16LinearSmokeReport(diagnosticReference, expectedIdentity);
assert.equal(
  diagnosticAdmission.ok,
  true,
  `CPU reference drift is diagnostic when GPU comparisons are exact: ${diagnosticAdmission.errors.join('; ')}`,
);

console.log('packed fp16 linear contracts passed');
