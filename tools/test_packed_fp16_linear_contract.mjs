import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  assertPackedFp16LinearShaderContract,
  createPackedFp16LinearFixture,
  evaluatePackedFp16LinearSmokeReport,
  runLinearReference,
} from '../src/lib/packed_fp16_linear_assay.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
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

assert.throws(
  () => createPackedFp16LinearFixture({ rows: 0, inDim: 5, outDim: 7 }),
  /positive integers/,
);
assert.throws(
  () => assertPackedFp16LinearShaderContract(shader.replace('unpack2x16float', 'fakeUnpack')),
  /unpack2x16float/,
);

const admittedReport = {
  schema: 'sf3d.packed-fp16-linear-browser-smoke.v0',
  ok: true,
  effectiveBackend: 'webgpu',
  effectiveRepresentation: 'f16-packed-u32',
  source: { revision: 'fixture', dirtyDiffSha256: null },
  kit: { packageVersion: '0.1.46', producerRevision: 'ace42925', tarballSha256: 'fixture' },
  browser: { version: 'Chrome/fixture', userAgent: 'fixture-agent' },
  adapter: { vendor: 'fixture-gpu' },
  control: { storageByteLength: 140, output: [1, 2, 3], outputFinite: true, outputSha256: 'a'.repeat(64) },
  candidate: { storageByteLength: 72, output: [1, 2, 3], outputFinite: true, outputSha256: 'a'.repeat(64) },
  comparison: { exact: true, maxAbsDiff: 0 },
  terminal: { phase: 'complete', primaryOutputWritten: true },
};
assert.equal(evaluatePackedFp16LinearSmokeReport(admittedReport).ok, true);

const fallback = structuredClone(admittedReport);
fallback.effectiveBackend = 'cpu-fallback';
assert.match(evaluatePackedFp16LinearSmokeReport(fallback).errors.join('\n'), /effectiveBackend/);

const missingBrowser = structuredClone(admittedReport);
delete missingBrowser.browser;
assert.match(evaluatePackedFp16LinearSmokeReport(missingBrowser).errors.join('\n'), /browser identity/);

const missingOutput = structuredClone(admittedReport);
delete missingOutput.candidate.output;
assert.match(evaluatePackedFp16LinearSmokeReport(missingOutput).errors.join('\n'), /candidate output/);

const falseParity = structuredClone(admittedReport);
falseParity.comparison.exact = false;
falseParity.comparison.maxAbsDiff = 0.25;
assert.match(evaluatePackedFp16LinearSmokeReport(falseParity).errors.join('\n'), /exact output parity/);

const failedBeforeOutput = structuredClone(admittedReport);
failedBeforeOutput.ok = false;
failedBeforeOutput.terminal = { phase: 'dispatch', primaryOutputWritten: false };
assert.match(evaluatePackedFp16LinearSmokeReport(failedBeforeOutput).errors.join('\n'), /terminal success/);

console.log('packed fp16 linear contracts passed');
