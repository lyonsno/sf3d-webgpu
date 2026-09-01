import assert from 'node:assert/strict';

export const DINO_ASSAY_SCHEMA = 'sf3d.dino-weight-representation-browser-assay.v0';
export const DINO_MATRIX_TENSOR_COUNT = 192;
export const DINO_MATRIX_SOURCE_BYTES = 754_974_720;
export const DINO_MATRIX_EXPANDED_BYTES = 1_509_949_440;
export const DINO_OUTPUT_ELEMENT_COUNT = 1_328_128;
export const DINO_OUTPUT_BYTES = 5_312_512;

const REPRESENTATIONS = Object.freeze(['f32-expanded', 'f16-packed-u32']);

function requireIdentity(actual, expected, field) {
  assert.equal(actual, expected, `${field} must match the expected effective source`);
}

export function validateDinoWeightRepresentationAssay(report, expectedSource) {
  assert.equal(report?.schema, DINO_ASSAY_SCHEMA, 'unexpected assay schema');
  assert.equal(report.ok, true, 'assay must be explicitly successful');
  assert.equal(report.phase, 'complete', 'assay must reach its terminal phase');

  for (const field of [
    'commit',
    'kitVersion',
    'kitTarballSha256',
    'weightsSha256',
    'inputSha256',
  ]) {
    requireIdentity(report.source?.[field], expectedSource?.[field], `source.${field}`);
  }

  for (const representation of REPRESENTATIONS) {
    const arm = report.arms?.[representation];
    assert.ok(arm, `missing ${representation} arm`);
    assert.equal(arm.requestedRepresentation, representation,
      `${representation} requested identity drifted`);
    assert.equal(arm.effectiveRepresentation, representation,
      `${representation} silently fell back`);
    assert.match(arm.browser || '', /Chrome\/\d+/, `${representation} browser identity missing`);
    assert.ok(arm.adapter && typeof arm.adapter === 'object',
      `${representation} adapter identity missing`);
    assert.deepEqual(arm.pageErrors, [], `${representation} emitted page errors`);

    const stats = arm.matrixStats;
    assert.equal(stats?.representation, representation,
      `${representation} matrix stats identity drifted`);
    assert.equal(stats?.tensorCount, DINO_MATRIX_TENSOR_COUNT,
      `${representation} did not materialize all DINO matrices`);
    assert.equal(stats?.sourceByteLength, DINO_MATRIX_SOURCE_BYTES,
      `${representation} source-byte authority drifted`);
    assert.equal(stats?.expandedFp32ByteLength, DINO_MATRIX_EXPANDED_BYTES,
      `${representation} expanded-byte authority drifted`);
    const expectedStorage = representation === 'f16-packed-u32'
      ? DINO_MATRIX_SOURCE_BYTES
      : DINO_MATRIX_EXPANDED_BYTES;
    const expectedSaving = representation === 'f16-packed-u32'
      ? DINO_MATRIX_SOURCE_BYTES
      : 0;
    assert.equal(stats?.storageByteLength, expectedStorage,
      `${representation} storage-byte authority drifted`);
    assert.equal(stats?.savedVsExpandedFp32ByteLength, expectedSaving,
      `${representation} saving authority drifted`);

    assert.equal(arm.cooperative?.status, 'succeeded',
      `${representation} cooperative execution did not complete`);
    assert.equal(arm.cooperative?.completedItems, 24,
      `${representation} returned a partial DINO execution`);
    assert.equal(arm.cooperative?.totalItems, 24,
      `${representation} hid or changed the DINO denominator`);

    assert.deepEqual(arm.output?.shape, { N: 1297, dim: 1024 },
      `${representation} output shape drifted`);
    assert.equal(arm.output?.elementCount, DINO_OUTPUT_ELEMENT_COUNT,
      `${representation} output was partial`);
    assert.equal(arm.output?.byteLength, DINO_OUTPUT_BYTES,
      `${representation} output byte length drifted`);
    assert.equal(arm.output?.finiteCount, DINO_OUTPUT_ELEMENT_COUNT,
      `${representation} output contains non-finite values`);
    assert.match(arm.output?.sha256 || '', /^[0-9a-f]{64}$/,
      `${representation} output hash missing`);
  }

  assert.equal(report.comparison?.byteIdentical, true,
    'expanded and packed DINO outputs are not byte-identical');
  assert.equal(report.comparison?.firstDifferingIndex, null,
    'comparison retained a differing output index');
  assert.equal(report.comparison?.maxAbsDelta, 0,
    'comparison retained a numerical delta');
  assert.equal(
    report.arms['f32-expanded'].output.sha256,
    report.arms['f16-packed-u32'].output.sha256,
    'arm output hashes differ',
  );
  return true;
}
