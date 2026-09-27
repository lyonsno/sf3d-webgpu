import assert from 'node:assert/strict';
import { acceptPreviewAssay } from './preview_assay_acceptance.mjs';

const valid = {
  requested: { partial: false },
  result: {
    final: { vertices: 9988, faces: 19976 },
    previewMesh: { vertices: 2662, faces: 5324 },
    browserErrors: [],
  },
};
assert.doesNotThrow(() => acceptPreviewAssay(valid));
assert.throws(() => acceptPreviewAssay({ ...valid, result: {
  ...valid.result, final: { vertices: 0, faces: 0 },
} }), /final mesh is empty/);
assert.throws(() => acceptPreviewAssay({ ...valid, result: {
  ...valid.result, previewMesh: null,
} }), /final triplane preview/);
assert.throws(() => acceptPreviewAssay({ ...valid, result: {
  ...valid.result, browserErrors: ['device lost'],
} }), /browser errors/);
assert.throws(() => acceptPreviewAssay({
  requested: { intermediateStage: 'block-0-fuse-out' },
  result: { ...valid.result, previewMesh: null, intermediatePreview: null },
}), /intermediate preview/);
assert.throws(() => acceptPreviewAssay({
  requested: { intermediateStages: ['block-0-fuse-out', 'block-1-fuse-out'] },
  result: { ...valid.result, previewMesh: null, intermediatePreviews: [
    { stageId: 'block-0-fuse-out', mesh: { faces: 3200 } },
  ] },
}), /missing intermediate preview block-1-fuse-out/);
console.log('preview assay rejects empty final output and missing candidate');
