import assert from 'node:assert/strict';
import { createPreviewTetGrid, createReducedTriplanePlan, extractPreviewMesh } from '../src/lib/preview_geometry.js';
import { marchingTetrahedra, scaleTensor } from '../src/lib/marching_tet.js';

assert.throws(() => createPreviewTetGrid(0), /resolution/);
assert.throws(() => createPreviewTetGrid(2.5), /resolution/);
assert.deepEqual(createReducedTriplanePlan(4), {
  sourceSize: 96, inputSize: 24, outputSize: 96, factor: 4,
});
assert.equal(createReducedTriplanePlan(2).outputSize, 192);
assert.throws(() => createReducedTriplanePlan(5), /divide/);

const grid = createPreviewTetGrid(4);
assert.equal(grid.numVertices, 125);
assert.equal(grid.numTets, 384);
assert.equal(grid.gridVertices.length, 375);
assert.equal(grid.indices.length, 1536);
assert.deepEqual(Array.from(grid.gridVertices.slice(0, 3)), [0, 0, 0]);
assert.deepEqual(Array.from(grid.gridVertices.slice(-3)), [1, 1, 1]);
for (const index of grid.indices) assert.ok(index >= 0 && index < grid.numVertices);

const positions = scaleTensor(grid.gridVertices, [0, 1], [-0.87, 0.87]);
const sdf = new Float32Array(grid.numVertices);
for (let i = 0; i < grid.numVertices; i += 1) {
  const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
  sdf[i] = 0.55 - Math.hypot(x, y, z);
}
const mesh = marchingTetrahedra(positions, sdf, grid.indices, null, 160);
assert.ok(mesh.numVertices > 0 && mesh.numFaces > 0);
for (const value of mesh.vertices) assert.ok(Number.isFinite(value) && Math.abs(value) < 0.87);
for (const index of mesh.faces) assert.ok(index < mesh.numVertices);
const density = Float32Array.from(sdf, (value) => value + 10);
const extracted = extractPreviewMesh(grid, density, new Float32Array(grid.numVertices * 3));
assert.equal(extracted.numFaces, mesh.numFaces);
assert.throws(() => extractPreviewMesh(grid, new Float32Array(3), null), /density/);
assert.throws(() => extractPreviewMesh(grid, density, new Float32Array(3)), /offset/);
const invalid = density.slice();
invalid[0] = Number.NaN;
assert.throws(() => extractPreviewMesh(grid, invalid, null), /non-finite/);
console.log(`preview grid: ${grid.numVertices} samples, ${grid.numTets} tetrahedra, ${mesh.numFaces} sphere faces`);
