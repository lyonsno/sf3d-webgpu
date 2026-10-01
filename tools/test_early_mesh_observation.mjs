import assert from 'node:assert/strict';
import { mock } from 'node:test';

const events = [];
const vertices = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
const faces = new Uint32Array([0, 1, 2]);
mock.module('../src/lib/inference.js', { namedExports: {
  runInference: async () => { events.push('extracted'); return { vertices, faces, numVertices: 3, numFaces: 1 }; },
} });
mock.module('../src/lib/clip_estimator.js', { namedExports: {
  estimateMaterials: async () => { events.push('materials'); return { roughness: 0.5, metallic: 0 }; },
} });
mock.module('../src/lib/texture_baker.js', { namedExports: {
  unwrapUV: () => { events.push('unwrap'); return {}; },
  rasterizeUV: () => ({}), bakeTexture: async () => ({}), exportGLB: () => new ArrayBuffer(8),
} });
globalThis.document = { createElement: () => ({ getContext: () => ({ drawImage() {}, getImageData: () => ({ data: [] }) }) }) };
const { runFullPipelineToGlb } = await import('../src/lib/full_pipeline.js');
const result = await runFullPipelineToGlb({}, {}, {}, {}, {
  onMeshExtracted: async mesh => {
    events.push('observed');
    assert.deepEqual([...mesh.vertices], [...vertices]);
    assert.deepEqual([...mesh.faces], [...faces]);
    mesh.vertices.fill(999);
    mesh.faces.fill(999);
    throw new Error('viewer failure');
  },
  onObservationError: ({ stageId }) => { events.push(`error:${stageId}`); },
});
assert.deepEqual(events, ['extracted', 'observed', 'error:mesh-extracted', 'materials', 'unwrap']);
assert.equal(result.vertices[0], 0, 'the observer cannot mutate the mesh being textured');
assert.equal(result.faces[0], 0);
assert.equal(result.glb.byteLength, 8, 'observer failure must not prevent export');
console.log('Full mesh snapshot precedes finishing, isolates mutable arrays, and fails open');
