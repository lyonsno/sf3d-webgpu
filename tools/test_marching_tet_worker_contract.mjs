#!/usr/bin/env node
/**
 * Marching-tet worker dispatcher contract (marching_tet.js runMarchingTetOnWorker
 * + validateMarchingTetReply), against a fake worker that runs the REAL
 * marchingTetrahedra on a tiny synthetic grid.
 *
 * Fail-first invariants:
 *   1. Worker path is byte-identical (vertices + faces) to the main-thread path.
 *   2. The caller's sdf / vertexOffsets are copied, never detached.
 *   3. Malformed replies (bad lengths, out-of-range face index, non-finite
 *      vertex, missing buffers) reject loudly instead of passing bad geometry.
 */
import assert from 'node:assert/strict';
import {
  marchingTetrahedra, runMarchingTetOnWorker, scaleTensor, validateMarchingTetReply,
} from '../src/lib/marching_tet.js';

// Two tets sharing a face, grid in [0,1]; sdf mixes signs so both are crossed.
const gridVertices = new Float32Array([
  0, 0, 0,   1, 0, 0,   0, 1, 0,   0, 0, 1,   1, 1, 1,
]);
const tetIndices = new Int32Array([0, 1, 2, 3,  1, 2, 3, 4]);
const sdf = new Float32Array([1.0, -0.5, 0.25, -2.0, 0.75]);
const vertexOffsets = new Float32Array(5 * 3).map((_, i) => ((i * 7) % 5 - 2) * 0.1);
const bbox = [-0.87, 0.87];
const resolution = 160;

const gridPositions = scaleTensor(gridVertices, [0, 1], bbox);
const expected = marchingTetrahedra(gridPositions, sdf, tetIndices, vertexOffsets, resolution);
assert.ok(expected.numVertices > 0 && expected.numFaces > 0, 'synthetic grid must produce a mesh');

function makeFakeWorker(behavior) {
  const listeners = { message: [], error: [], messageerror: [] };
  return {
    posted: [],
    addEventListener: (t, fn) => listeners[t].push(fn),
    removeEventListener: (t, fn) => { listeners[t] = listeners[t].filter(f => f !== fn); },
    _emit: (type, data) => { for (const fn of [...listeners[type]]) fn(data); },
    postMessage(msg, transfer) { this.posted.push({ msg, transfer }); behavior(this, msg); },
  };
}

// Fake worker that behaves exactly like marching_tet_worker.js against the synthetic grid.
const realWorker = makeFakeWorker((self, msg) => {
  const positions = scaleTensor(gridVertices, [0, 1], msg.bbox);
  const mesh = marchingTetrahedra(positions, new Float32Array(msg.sdf), tetIndices,
    msg.vertexOffsets ? new Float32Array(msg.vertexOffsets) : null, msg.resolution);
  queueMicrotask(() => self._emit('message', { data: {
    id: msg.id, ok: true, vertices: mesh.vertices.buffer, faces: mesh.faces.buffer,
    numVertices: mesh.numVertices, numFaces: mesh.numFaces,
  } }));
});

// 1 + 2. Byte-identical and non-detaching.
{
  const sdfBefore = sdf.byteLength, offBefore = vertexOffsets.byteLength;
  const mesh = await runMarchingTetOnWorker(realWorker, { sdf, vertexOffsets, bbox, resolution }, { timeoutMs: 5000 });
  assert.equal(mesh.numVertices, expected.numVertices);
  assert.equal(mesh.numFaces, expected.numFaces);
  assert.equal(Buffer.compare(Buffer.from(mesh.vertices.buffer), Buffer.from(expected.vertices.buffer)), 0, 'vertices byte-identical');
  assert.equal(Buffer.compare(Buffer.from(mesh.faces.buffer), Buffer.from(expected.faces.buffer)), 0, 'faces byte-identical');
  assert.equal(sdf.byteLength, sdfBefore, 'caller sdf not detached');
  assert.equal(vertexOffsets.byteLength, offBefore, 'caller vertexOffsets not detached');
  assert.equal(realWorker.posted.length, 1);
  assert.deepEqual(realWorker.posted[0].msg.bbox, bbox);
  assert.equal(realWorker.posted[0].transfer.length, 2, 'sdf + offsets copies transferred');
  console.log('ok  worker mesh byte-identical to main thread; inputs not detached');
}

// 3. Malformed replies reject.
const bad = (patch) => makeFakeWorker((self, msg) => {
  const good = {
    id: msg.id, ok: true,
    vertices: expected.vertices.slice().buffer, faces: expected.faces.slice().buffer,
    numVertices: expected.numVertices, numFaces: expected.numFaces,
  };
  queueMicrotask(() => self._emit('message', { data: { ...good, ...patch(good) } }));
});
const cases = [
  ['vertices length', (g) => ({ vertices: new Float32Array(g.numVertices * 3 - 1).buffer }), /vertices length/],
  ['faces length', (g) => ({ faces: new Uint32Array(g.numFaces * 3 + 3).buffer }), /faces length/],
  ['face index out of range', (g) => { const f = new Uint32Array(g.faces); f[0] = g.numVertices; return { faces: f.buffer }; }, /out of range/],
  ['non-finite vertex', (g) => { const v = new Float32Array(g.vertices); v[1] = NaN; return { vertices: v.buffer }; }, /non-finite/],
  ['missing buffers', () => ({ vertices: null }), /ArrayBuffers/],
  ['zero faces', () => ({ numFaces: 0, faces: new Uint32Array(0).buffer }), /numFaces invalid/],
];
for (const [name, patch, re] of cases) {
  await assert.rejects(
    () => runMarchingTetOnWorker(bad(patch), { sdf, vertexOffsets, bbox, resolution }, { timeoutMs: 5000 }),
    (err) => /worker output invalid/.test(err.message) && re.test(err.message),
    `${name} must reject`);
  console.log(`ok  rejects malformed reply: ${name}`);
}
await assert.rejects(() => runMarchingTetOnWorker(realWorker, { sdf: [1, 2], vertexOffsets, bbox, resolution }), /Float32Array/);
validateMarchingTetReply({
  vertices: expected.vertices.slice().buffer, faces: expected.faces.slice().buffer,
  numVertices: expected.numVertices, numFaces: expected.numFaces,
});

console.log('\nMARCHING TET WORKER CONTRACT PASSED');
