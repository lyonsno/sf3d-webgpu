/**
 * parity_compare_core contract (deterministic, no GPU, no browser).
 *
 * Proves the SF3D adapter around @kaminos/webgpu-inference-kit parity
 * primitives (compareStages) and the minimal .npy reader (readNpy) against
 * synthetic inputs whose answers are known exactly:
 *   - exact match / one perturbed element / worst-stage selection
 *   - length mismatch, non-finite, Float64 and column-major inputs reject loudly
 *   - one-sided stages land in `missing`, never silently dropped
 *   - the triplane (scene_codes) layout-mismatch stage is stats-only
 *   - .npy v1.0 headers for <f4, <f2 (FP16 decode), <i4, <u4, |u1, <i8 (narrowed)
 *   - base64 -> Float32Array transfer decode round-trips
 *
 * Run: node tools/test_parity_compare_contract.mjs
 */
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import {
  WEBGPU_INFERENCE_KIT_VERSION,
  WEBGPU_PARITY_CAPTURE_SCHEMA,
  WEBGPU_PARITY_COMPARISON_SCHEMA,
} from '@kaminos/webgpu-inference-kit';
import {
  SF3D_PARITY_REPORT_SCHEMA,
  compareStages,
  decodeBase64Float32,
  decodeFloat16,
  loadReferenceStages,
  readNpy,
  summarizeValues,
} from './parity_compare_core.mjs';

const RUN_ID = 'sf3d-parity-contract-test';

// Multiples of 0.125 are exact in float32, so perturbations and error sums are exact.
function synthetic(n, seed = 0) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (((i * 7 + seed) % 23) - 11) * 0.125;
  return out;
}

// --- NPY writer for fixtures (format spec v1.0, 64-byte aligned header) ---
function npyBytes({ descr, shape, fortranOrder = false, payload, version = 1 }) {
  const shapeText = shape.length === 0 ? '()'
    : shape.length === 1 ? `(${shape[0]},)`
    : `(${shape.join(', ')})`;
  let dict = `{'descr': '${descr}', 'fortran_order': ${fortranOrder ? 'True' : 'False'}, 'shape': ${shapeText}, }`;
  const prefixLength = version === 1 ? 10 : 12;
  const pad = (64 - ((prefixLength + dict.length + 1) % 64)) % 64;
  dict = `${dict}${' '.repeat(pad)}\n`;
  const header = Buffer.alloc(prefixLength + dict.length);
  header.write('\x93NUMPY', 0, 'latin1');
  header[6] = version;
  header[7] = 0;
  if (version === 1) header.writeUInt16LE(dict.length, 8);
  else header.writeUInt32LE(dict.length, 8);
  header.write(dict, prefixLength, 'latin1');
  assert.equal((prefixLength + dict.length) % 64, 0, 'fixture header must be 64-byte aligned');
  return Buffer.concat([header, Buffer.from(payload.buffer, payload.byteOffset, payload.byteLength)]);
}

const fixtureDir = mkdtempSync(path.join(tmpdir(), 'sf3d-parity-contract-'));
function writeNpy(name, spec) {
  const filePath = path.join(fixtureDir, name);
  writeFileSync(filePath, npyBytes(spec));
  return filePath;
}

try {
  // 1. Exact match: zero error everywhere, metrics and schema identities exact.
  {
    const actual = synthetic(16);
    const reference = Float32Array.from(actual);
    const report = compareStages(
      { density: { values: actual, shape: [16] } },
      { density: { values: reference, shape: [1, 16, 1] } },
      { runId: RUN_ID },
    );
    assert.equal(report.schema, SF3D_PARITY_REPORT_SCHEMA);
    assert.equal(report.schema, 'sf3d.parity-comparison-report.v0');
    assert.equal(report.runId, RUN_ID);
    assert.equal(report.kitVersion, WEBGPU_INFERENCE_KIT_VERSION);
    assert.deepEqual(Object.keys(report.stages), ['density']);
    const stage = report.stages.density;
    assert.equal(stage.stageId, 'density');
    assert.equal(stage.mode, 'element-wise');
    assert.equal(stage.layoutMismatch, false);
    assert.deepEqual(stage.shape, { webgpu: [16], reference: [1, 16, 1] });
    assert.equal(stage.capture.schema, WEBGPU_PARITY_CAPTURE_SCHEMA);
    assert.equal(stage.capture.elementCount, 16);
    assert.equal(stage.capture.byteLength, 64);
    const c = stage.comparison;
    assert.equal(c.schema, WEBGPU_PARITY_COMPARISON_SCHEMA);
    assert.equal(c.stageId, 'density');
    assert.equal(c.runId, RUN_ID);
    assert.equal(c.comparisonDomain.mode, 'float32-metrics');
    assert.equal(c.sourceElementCount, 16);
    assert.equal(c.comparedElementCount, 16);
    assert.equal(c.sampling.mode, 'all');
    assert.equal(c.metrics.exactMatch, true);
    assert.equal(c.metrics.mismatchCount, 0);
    assert.equal(c.metrics.maxAbsoluteError, 0);
    assert.equal(c.metrics.meanAbsoluteError, 0);
    assert.equal(c.metrics.rootMeanSquareError, 0);
    assert.equal(c.metrics.l2Error, 0);
    assert.equal(c.metrics.relativeL2Error, 0);
    assert.equal(c.metrics.relativeL2Status, 'defined');
    assert.ok(c.metrics.cosineSimilarity > 0.999999, `cosine ${c.metrics.cosineSimilarity}`);
    assert.deepEqual(report.summary, {
      compared: 1,
      exactMatches: 1,
      statsOnly: 0,
      worstRelativeL2: { stageId: 'density', value: 0, status: 'defined' },
      worstMaxAbsoluteError: { stageId: 'density', value: 0, sourceIndex: 0 },
    });
    assert.deepEqual(report.missing, { webgpu: [], reference: [] });
    // Report is JSON-serializable without loss of the kit records.
    const roundTrip = JSON.parse(JSON.stringify(report));
    assert.equal(roundTrip.stages.density.comparison.metrics.exactMatch, true);
    console.log('ok  exact match -> exactMatch true, mismatchCount 0, schema identities exact');
  }

  // 2. One perturbed element: exact mismatch accounting and worst index.
  {
    const reference = synthetic(16);
    const actual = Float32Array.from(reference);
    actual[5] += 0.25;
    const report = compareStages(
      { density: { values: actual } },
      { density: { values: reference } },
      { runId: RUN_ID },
    );
    const m = report.stages.density.comparison.metrics;
    assert.equal(m.exactMatch, false);
    assert.equal(m.mismatchCount, 1);
    assert.equal(m.maxAbsoluteError, 0.25);
    assert.equal(m.worstSourceIndex, 5);
    assert.equal(m.worstActual, actual[5]);
    assert.equal(m.worstReference, reference[5]);
    assert.equal(m.meanAbsoluteError, 0.25 / 16);
    assert.equal(m.l2Error, 0.25);
    assert.equal(m.rootMeanSquareError, 0.25 / 4);
    assert.equal(report.summary.compared, 1);
    assert.equal(report.summary.exactMatches, 0);
    assert.deepEqual(report.summary.worstMaxAbsoluteError, { stageId: 'density', value: 0.25, sourceIndex: 5 });
    assert.equal(report.summary.worstRelativeL2.stageId, 'density');
    assert.ok(report.summary.worstRelativeL2.value > 0);
    console.log('ok  one perturbed element -> mismatchCount 1, maxAbs 0.25 at index 5');
  }

  // 3. Worst-stage selection across stages, plus integer-exact domain.
  {
    const refA = synthetic(8);
    const actA = Float32Array.from(refA); actA[1] += 0.125;
    const refB = synthetic(8, 3);
    const actB = Float32Array.from(refB); actB[7] += 1.0;
    const faces = Uint32Array.from([0, 1, 2, 2, 3, 0]);
    const report = compareStages(
      { camera_embed: { values: actA }, vertex_offset: { values: actB }, faces: { values: faces } },
      { camera_embed: { values: refA }, vertex_offset: { values: refB }, faces: { values: Uint32Array.from(faces) } },
      { runId: RUN_ID },
    );
    assert.equal(report.summary.compared, 3);
    assert.equal(report.summary.exactMatches, 1);
    assert.deepEqual(report.summary.worstMaxAbsoluteError, { stageId: 'vertex_offset', value: 1.0, sourceIndex: 7 });
    assert.equal(report.summary.worstRelativeL2.stageId, 'vertex_offset');
    assert.equal(report.stages.faces.comparison.comparisonDomain.mode, 'integer-exact');
    assert.equal(report.stages.faces.comparison.metrics.exactMatch, true);
    console.log('ok  worst stage selected across stages; integer stage compared exactly');
  }

  // 4. Length mismatch throws, naming the stage.
  {
    assert.throws(
      () => compareStages(
        { density: { values: synthetic(16) } },
        { density: { values: synthetic(15) } },
        { runId: RUN_ID },
      ),
      (err) => /density/.test(err.message) && /same length/.test(err.message),
    );
    console.log('ok  length mismatch throws with stage name');
  }

  // 5. Non-finite throws; Float64 throws (no silent narrowing).
  {
    const actual = synthetic(16);
    actual[3] = NaN;
    assert.throws(
      () => compareStages({ density: { values: actual } }, { density: { values: synthetic(16) } }, { runId: RUN_ID }),
      (err) => /density/.test(err.message) && /non-finite/.test(err.message),
    );
    assert.throws(
      () => compareStages(
        { density: { values: synthetic(4) } },
        { density: { values: Float64Array.from(synthetic(4)) } },
        { runId: RUN_ID },
      ),
      (err) => /Float64/.test(err.message),
    );
    console.log('ok  NaN and Float64 inputs reject');
  }

  // 6. One-sided stages land in `missing`, in stable order.
  {
    const report = compareStages(
      { density: { values: synthetic(4) }, webgpu_only: { values: synthetic(4) } },
      { density: { values: synthetic(4) }, camera_embed: { values: synthetic(4) }, vertex_offset: { values: synthetic(4) } },
      { runId: RUN_ID },
    );
    assert.deepEqual(Object.keys(report.stages), ['density']);
    assert.deepEqual(report.missing, {
      webgpu: ['camera_embed', 'vertex_offset'],
      reference: ['webgpu_only'],
    });
    console.log('ok  one-sided stages land in missing');
  }

  // 7. Layout-mismatch stage (triplane / scene_codes) is stats-only and never element-compared,
  //    even when lengths differ or values are only summarized on one side.
  {
    const refScene = Float32Array.from([-2, 0, 2, 4]);
    const report = compareStages(
      {
        density: { values: synthetic(4) },
        scene_codes: {
          stats: { min: -1, max: 3, mean: 0.5, count: 3 * 40 * 384 * 384 },
          layoutMismatch: true,
          layoutNote: 'WebGPU plane-major vs PyTorch NCHW',
        },
      },
      { density: { values: synthetic(4) }, scene_codes: { values: refScene, shape: [1, 3, 40, 384, 384].slice(0, 1).concat([4]) } },
      { runId: RUN_ID },
    );
    const stage = report.stages.scene_codes;
    assert.equal(stage.mode, 'stats-only');
    assert.equal(stage.layoutMismatch, true);
    assert.equal(stage.layoutNote, 'WebGPU plane-major vs PyTorch NCHW');
    assert.equal(stage.comparison, null);
    assert.equal(stage.capture, null);
    assert.deepEqual(stage.actual, { min: -1, max: 3, mean: 0.5, count: 3 * 40 * 384 * 384 });
    assert.deepEqual(stage.reference, { min: -2, max: 4, mean: 1, count: 4 });
    assert.equal(report.summary.compared, 1);
    assert.equal(report.summary.statsOnly, 1);
    assert.equal(report.summary.exactMatches, 1);
    // Values-bearing layout-mismatch stage summarizes from values.
    const fromValues = compareStages(
      { scene_codes: { values: Float32Array.from([1, 2, 3]), layoutMismatch: true } },
      { scene_codes: { values: Float32Array.from([1, 2, 3, 4]) } },
      { runId: RUN_ID },
    );
    assert.deepEqual(fromValues.stages.scene_codes.actual, { min: 1, max: 3, mean: 2, count: 3 });
    assert.deepEqual(fromValues.summary.worstRelativeL2, { stageId: null, value: null, status: null });
    assert.deepEqual(fromValues.summary.worstMaxAbsoluteError, { stageId: null, value: null, sourceIndex: null });
    console.log('ok  layout-mismatch stage is stats-only with explicit flag');
  }

  // 8. Element-wise stage on a column-major reference rejects (flattening order would lie).
  {
    assert.throws(
      () => compareStages(
        { density: { values: synthetic(4) } },
        { density: { values: synthetic(4), shape: [2, 2], fortranOrder: true } },
        { runId: RUN_ID },
      ),
      (err) => /density/.test(err.message) && /column-major|fortran/i.test(err.message),
    );
    console.log('ok  column-major multi-dim reference rejects');
  }

  // 9. Missing values on an element-wise stage reject; runId is required.
  {
    assert.throws(
      () => compareStages({ density: { shape: [4] } }, { density: { values: synthetic(4) } }, { runId: RUN_ID }),
      (err) => /density/.test(err.message) && /values/.test(err.message),
    );
    assert.throws(() => compareStages({}, {}, {}), /runId/);
    console.log('ok  element-wise stage without values rejects; runId required');
  }

  // 10. summarizeValues is exact on small inputs and counts non-finite.
  {
    assert.deepEqual(summarizeValues(Float32Array.from([1, -3, 2])), { min: -3, max: 2, mean: 0, count: 3 });
    assert.deepEqual(summarizeValues(Float32Array.from([1, NaN, 3])), { min: 1, max: 3, mean: 2, count: 3, nonFiniteCount: 1 });
    console.log('ok  summarizeValues exact');
  }

  // 11. FP16 decode of known bit patterns.
  {
    assert.equal(decodeFloat16(0x3c00), 1.0);
    assert.equal(decodeFloat16(0xc100), -2.5);
    assert.equal(decodeFloat16(0x0000), 0.0);
    assert.ok(Object.is(decodeFloat16(0x8000), -0));
    assert.equal(decodeFloat16(0x3800), 0.5);
    assert.equal(decodeFloat16(0x7bff), 65504);
    assert.equal(decodeFloat16(0x0001), 2 ** -24);
    assert.equal(decodeFloat16(0x3555), 0.333251953125);
    assert.equal(decodeFloat16(0x7c00), Infinity);
    assert.equal(decodeFloat16(0xfc00), -Infinity);
    assert.ok(Number.isNaN(decodeFloat16(0x7e00)));
    console.log('ok  FP16 decode of known bit patterns');
  }

  // 12. readNpy <f4 (3,) and <f2 (3,) with FP16 decode to Float32Array.
  {
    const f4 = readNpy(writeNpy('f4.npy', { descr: '<f4', shape: [3], payload: Float32Array.from([1.0, -2.5, 0.0]) }));
    assert.equal(f4.dtype, '<f4');
    assert.deepEqual(f4.shape, [3]);
    assert.equal(f4.fortranOrder, false);
    assert.equal(f4.elementCount, 3);
    assert.ok(f4.data instanceof Float32Array);
    assert.deepEqual(Array.from(f4.data), [1.0, -2.5, 0.0]);

    const f2 = readNpy(writeNpy('f2.npy', { descr: '<f2', shape: [3], payload: Uint16Array.from([0x3c00, 0xc100, 0x0000]) }));
    assert.equal(f2.dtype, '<f2');
    assert.deepEqual(f2.shape, [3]);
    assert.ok(f2.data instanceof Float32Array, 'FP16 must decode to Float32Array for the kit');
    assert.deepEqual(Array.from(f2.data), [1.0, -2.5, 0.0]);
    console.log('ok  readNpy <f4 and <f2 (FP16 -> Float32Array)');
  }

  // 13. readNpy integer dtypes, multi-dim shape, 0-d shape, v2.0 header.
  {
    const i4 = readNpy(writeNpy('i4.npy', { descr: '<i4', shape: [2, 2], payload: Int32Array.from([1, -2, 3, -4]) }));
    assert.ok(i4.data instanceof Int32Array);
    assert.deepEqual(i4.shape, [2, 2]);
    assert.deepEqual(Array.from(i4.data), [1, -2, 3, -4]);

    const u4 = readNpy(writeNpy('u4.npy', { descr: '<u4', shape: [2], payload: Uint32Array.from([7, 4294967295]) }));
    assert.ok(u4.data instanceof Uint32Array);
    assert.deepEqual(Array.from(u4.data), [7, 4294967295]);

    const u1 = readNpy(writeNpy('u1.npy', { descr: '|u1', shape: [4], payload: Uint8Array.from([0, 1, 254, 255]) }));
    assert.ok(u1.data instanceof Uint8Array);
    assert.deepEqual(Array.from(u1.data), [0, 1, 254, 255]);

    const i8 = readNpy(writeNpy('i8.npy', { descr: '<i8', shape: [3], payload: BigInt64Array.from([0n, 535881n, -7n]) }));
    assert.ok(i8.data instanceof Int32Array, '<i8 narrows to Int32Array when every value fits');
    assert.equal(i8.dtype, '<i8');
    assert.deepEqual(Array.from(i8.data), [0, 535881, -7]);
    assert.throws(
      () => readNpy(writeNpy('i8-wide.npy', { descr: '<i8', shape: [1], payload: BigInt64Array.from([2n ** 40n]) })),
      (err) => err instanceof RangeError && /Int32/.test(err.message),
    );

    const scalar = readNpy(writeNpy('scalar.npy', { descr: '<f4', shape: [], payload: Float32Array.from([4.5]) }));
    assert.deepEqual(scalar.shape, []);
    assert.equal(scalar.elementCount, 1);
    assert.deepEqual(Array.from(scalar.data), [4.5]);

    const v2 = readNpy(writeNpy('v2.npy', { descr: '<f4', shape: [2], payload: Float32Array.from([8, 9]), version: 2 }));
    assert.deepEqual(Array.from(v2.data), [8, 9]);

    const fortran = readNpy(writeNpy('fortran.npy', { descr: '<f4', shape: [2, 2], fortranOrder: true, payload: Float32Array.from([1, 2, 3, 4]) }));
    assert.equal(fortran.fortranOrder, true);
    console.log('ok  readNpy <i4, <u4, |u1, <i8 (narrowed), 0-d, v2.0 header, fortran flag');
  }

  // 14. readNpy rejects unsupported dtype, bad magic, and truncated payload with clear errors.
  {
    assert.throws(
      () => readNpy(writeNpy('f8.npy', { descr: '<f8', shape: [1], payload: Float64Array.from([1]) })),
      (err) => /unsupported/i.test(err.message) && /<f8/.test(err.message),
    );
    assert.throws(
      () => readNpy(writeNpy('be.npy', { descr: '>f4', shape: [1], payload: Float32Array.from([1]) })),
      (err) => /unsupported/i.test(err.message) && />f4/.test(err.message),
    );
    const bad = path.join(fixtureDir, 'bad.npy');
    writeFileSync(bad, Buffer.from('not an npy file at all, definitely'));
    assert.throws(() => readNpy(bad), /magic/i);
    const truncated = path.join(fixtureDir, 'truncated.npy');
    const full = npyBytes({ descr: '<f4', shape: [4], payload: Float32Array.from([1, 2, 3, 4]) });
    writeFileSync(truncated, full.subarray(0, full.length - 4));
    assert.throws(() => readNpy(truncated), (err) => /expected 16/.test(err.message) && /12/.test(err.message));
    console.log('ok  readNpy rejects unsupported dtype, bad magic, truncated payload');
  }

  // 15. loadReferenceStages reads present stages, skips absent ones, and preserves dtype/shape.
  {
    const stages = loadReferenceStages(fixtureDir, ['f4', 'absent_stage', 'i4']);
    assert.deepEqual(Object.keys(stages).sort(), ['f4', 'i4']);
    assert.equal(stages.f4.dtype, '<f4');
    assert.deepEqual(stages.f4.shape, [3]);
    assert.equal(stages.f4.fortranOrder, false);
    assert.ok(stages.f4.values instanceof Float32Array);
    assert.equal(stages.f4.file, path.join(fixtureDir, 'f4.npy'));
    assert.deepEqual(stages.i4.shape, [2, 2]);
    console.log('ok  loadReferenceStages reads present .npy stages only');
  }

  // 16. Observed-writer conformance: files written by numpy itself (tools/fixtures/npy,
  //     see manifest.json for generator/version). Guards the wire format beyond the
  //     hand-built headers above. Storage order is returned raw; fortran files keep
  //     their column-major byte order and set the flag.
  {
    const fixtureRoot = path.join(path.dirname(new URL(import.meta.url).pathname), 'fixtures', 'npy');
    const manifest = JSON.parse(readFileSync(path.join(fixtureRoot, 'manifest.json'), 'utf8'));
    assert.equal(manifest.generator, 'numpy');
    const expectedStorageOrder = {
      'f4_fortran_2x2.npy': [1, 3, 2, 4],
    };
    let checked = 0;
    for (const [file, spec] of Object.entries(manifest.files)) {
      const record = readNpy(path.join(fixtureRoot, file));
      assert.equal(record.dtype, spec.dtype, file);
      assert.deepEqual(record.shape, spec.shape, file);
      assert.equal(record.fortranOrder, spec.fortran_order, file);
      const expected = expectedStorageOrder[file] ?? spec.c_order_values;
      const actual = Array.from(record.data);
      assert.equal(actual.length, expected.length, file);
      for (let i = 0; i < expected.length; i++) {
        assert.ok(Object.is(actual[i], expected[i]), `${file}[${i}] ${actual[i]} !== ${expected[i]}`);
      }
      if (spec.dtype.startsWith('<f')) assert.ok(record.data instanceof Float32Array, `${file} must decode to Float32Array`);
      if (spec.dtype === '<i8') assert.ok(record.data instanceof Int32Array, `${file} must narrow to Int32Array`);
      checked += 1;
    }
    assert.equal(checked, 9);
    console.log(`ok  readNpy conforms on ${checked} numpy ${manifest.numpyVersion}-written fixtures`);
  }

  // 17. base64 transfer decode round-trips Float32Array bytes exactly.
  {
    const values = synthetic(1001);
    const b64 = Buffer.from(values.buffer, values.byteOffset, values.byteLength).toString('base64');
    const decoded = decodeBase64Float32(b64);
    assert.ok(decoded instanceof Float32Array);
    assert.equal(decoded.byteOffset, 0);
    assert.deepEqual(Array.from(decoded), Array.from(values));
    assert.throws(() => decodeBase64Float32(Buffer.from([1, 2, 3]).toString('base64')), /multiple of 4/);
    console.log('ok  base64 -> Float32Array round-trip exact');
  }

  console.log('\nparity compare contract: all checks passed');
} finally {
  rmSync(fixtureDir, { recursive: true, force: true });
}
