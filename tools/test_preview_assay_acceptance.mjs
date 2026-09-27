import assert from 'node:assert/strict';
import { acceptPreviewAssay } from './preview_assay_acceptance.mjs';

const valid = {
  requested: { partial: false, reducedFactor: 4 },
  result: {
    final: { vertices: 9988, faces: 19976 },
    reducedPreview: { mesh: { vertices: 1158, faces: 2312 } },
    browserErrors: [],
  },
};
assert.doesNotThrow(() => acceptPreviewAssay(valid));
assert.throws(() => acceptPreviewAssay({ ...valid, result: {
  ...valid.result, final: { vertices: 0, faces: 0 },
} }), /final mesh is empty/);
assert.throws(() => acceptPreviewAssay({ ...valid, result: {
  ...valid.result, reducedPreview: null,
} }), /reduced preview/);
assert.throws(() => acceptPreviewAssay({ ...valid, result: {
  ...valid.result, browserErrors: ['device lost'],
} }), /browser errors/);
console.log('preview assay rejects empty final output and missing candidate');
