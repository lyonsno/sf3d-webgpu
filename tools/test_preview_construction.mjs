import assert from 'node:assert/strict';
import { mock } from 'node:test';
import { createPreviewTetGrid, extractPreviewMesh } from '../src/lib/preview_geometry.js';

const allocations = [];
let scope;
let submissions = 0;
mock.module('../src/lib/gpu.js', { namedExports: {
  createStorageBuffer: (_device, data) => {
    const buffer = { data, destroyed: false, destroy() { this.destroyed = true; } };
    scope.push({ buffer, size: data.byteLength }); allocations.push(buffer);
    return buffer;
  },
  captureGpuBufferAllocations: fn => { scope = []; return { value: fn(), allocations: scope }; },
  readBuffer: async (_device, data) => data,
} });
const { decodePreviewMesh } = await import('../src/lib/preview_geometry_gpu.js');
const device = { createCommandEncoder: () => ({ finish() {} }), queue: { submit() { submissions++; } } };
const decoder = { decode(_encoder, positions, _planes, count) {
  return {
    density: Float32Array.from({ length: count }, (_, i) => 10 + 0.6 - Math.hypot(...positions.data.subarray(i * 3, i * 3 + 3))),
    vertex_offset: new Float32Array(count * 3),
  };
} };
const whole = await decodePreviewMesh(device, null, decoder, {}, 6);
const updates = [];
submissions = 0;
const streamed = await decodePreviewMesh(device, null, decoder, {}, 6, 384, {
  layersPerSlab: 2,
  onSlab: async sample => {
    assert.equal(submissions, updates.length + 1, 'one completed slab per submission');
    assert.ok(sample.completedSamples <= sample.totalSamples);
    for (let i = 2; i < sample.mesh.vertices.length; i += 3) {
      assert.ok(sample.mesh.vertices[i] <= sample.maxZ + 1e-6, 'no geometry beyond known samples');
    }
    updates.push(sample);
    await Promise.resolve();
    assert.equal(submissions, updates.length, 'observer settles before next slab');
  },
});
assert.equal(updates.length, 4, 'slab observations must occur during decoding');
assert.deepEqual(updates.map(s => s.completedLayers), [2, 4, 6, 7]);
assert.equal(updates.at(-1).completedSamples, 343);
assert.deepEqual(streamed.mesh, whole.mesh, 'streaming must preserve final topology and coordinates');
assert.ok(allocations.every(buffer => buffer.destroyed));

const grid = createPreviewTetGrid(6);
const density = new Float32Array(grid.numVertices).fill(NaN);
const known = 3 * 49;
density.fill(11, 0, known);
const partial = extractPreviewMesh(grid, density, null, { completedLayers: 3 });
assert.equal(partial.numFaces, 0, 'unknown space must not cap the known occupied region');
assert.throws(() => extractPreviewMesh(grid, density, null), /non-finite/);
await assert.rejects(decodePreviewMesh(device, null, decoder, {}, 6, 384, {
  layersPerSlab: 2, onSlab: () => { throw new Error('observer broke'); },
}), /observer broke/);
assert.ok(allocations.every(buffer => buffer.destroyed), 'observer failure retires current slab');
let produced = 0;
const regionUpdates = [];
const regions = await decodePreviewMesh(device, null, decoder, {}, 6, 24, {
  onSlab: sample => regionUpdates.push(sample.completedLayers),
  produceRegions: async consume => {
    for (const completedRows of [8, 16, 24]) {
      produced++;
      await consume({ buffer: null, completedRows, totalRows: 24 });
    }
  },
});
assert.equal(produced, 3, 'query must be driven by completed feature regions');
assert.deepEqual(regionUpdates, [2, 4, 7]);
assert.deepEqual(regions.mesh, whole.mesh);
console.log('Spatial construction: actual sequential slabs, no unknown-space caps, exact final mesh, cleanup');
