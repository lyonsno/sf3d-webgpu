#!/usr/bin/env node
/**
 * Worker-reply validation contract (src/lib/worker_reply_validation.js).
 *
 * Review 2026-09-16 (fresh GPT-5.6 Sol, b8222a2..e73835c, HIGH): the
 * preprocessing boundary accepted length-correct non-finite CHW data and the
 * UV-unwrap boundary never checked normals/face-assignment lengths, value
 * finiteness, face-index bounds, or assignment validity. A malformed worker
 * reply must throw at the boundary; it must never become an apparently
 * successful offload.
 */
import assert from 'node:assert/strict';
import {
  UV_ATLAS_CHART_COUNT,
  validatePreprocessReply,
  validateUvUnwrapReply,
} from '../src/lib/worker_reply_validation.js';

// --- Preprocess ---
{
  const len = 3 * 4 * 4;
  const good = validatePreprocessReply({ chwBuffer: new Float32Array(len).fill(0.5).buffer }, len);
  assert.ok(good instanceof Float32Array); assert.equal(good.length, len);
  assert.throws(() => validatePreprocessReply({ chwBuffer: new Float32Array(len - 1).buffer }, len), /CHW length 47 != expected 48/);
  const nan = new Float32Array(len).fill(0.5); nan[7] = NaN;
  assert.throws(() => validatePreprocessReply({ chwBuffer: nan.buffer }, len), /preprocess CHW value non-finite at 7/);
  const inf = new Float32Array(len).fill(0.5); inf[len - 1] = Infinity;
  assert.throws(() => validatePreprocessReply({ chwBuffer: inf.buffer }, len), /non-finite at 47/);
  assert.throws(() => validatePreprocessReply({}, len), /must carry chwBuffer/);
  console.log('ok  preprocess reply: shape + finiteness enforced');
}

// --- UV unwrap ---
function goodUv() {
  const nv = 4, nf = 2;
  return {
    newNumVertices: nv, newNumFaces: nf,
    newVertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]).buffer,
    newNormals: new Float32Array(nv * 3).fill(0.577).buffer,
    uvs: new Float32Array(nv * 2).fill(0.25).buffer,
    newFaces: new Uint32Array([0, 1, 2, 1, 2, 3]).buffer,
    faceAssignment: new Uint8Array([0, 5]).buffer,
  };
}
{
  const r = validateUvUnwrapReply(goodUv());
  assert.equal(r.newNumVertices, 4); assert.equal(r.newNumFaces, 2);
  assert.ok(r.newNormals instanceof Float32Array && r.faceAssignment instanceof Uint8Array);
  assert.equal(UV_ATLAS_CHART_COUNT, 6);
  console.log('ok  uv-unwrap reply: valid reply returned as typed arrays');
}
const uvFalsifiers = [
  ['truncated normals', d => { d.newNormals = new Float32Array(3).buffer; }, /newNormals length 3 != 12/],
  ['empty normals', d => { d.newNormals = new Float32Array(0).buffer; }, /newNormals length 0 != 12/],
  ['truncated face assignment', d => { d.faceAssignment = new Uint8Array([0]).buffer; }, /faceAssignment length 1 != 2/],
  ['non-finite vertex', d => { const v = new Float32Array(d.newVertices); v[4] = NaN; d.newVertices = v.buffer; }, /newVertices value non-finite at 4/],
  ['non-finite normal', d => { const v = new Float32Array(d.newNormals); v[0] = -Infinity; d.newNormals = v.buffer; }, /newNormals value non-finite at 0/],
  ['non-finite uv', d => { const v = new Float32Array(d.uvs); v[5] = NaN; d.uvs = v.buffer; }, /uvs value non-finite at 5/],
  ['face index out of range', d => { const f = new Uint32Array(d.newFaces); f[5] = 4; d.newFaces = f.buffer; }, /face index 4 out of range \(4 vertices\) at 5/],
  ['invalid assignment value', d => { d.faceAssignment = new Uint8Array([0, 6]).buffer; }, /faceAssignment value 6 out of range \(6 charts\) at 1/],
  ['vertex length mismatch', d => { d.newVertices = new Float32Array(9).buffer; }, /newVertices length 9 != 12/],
  ['missing buffer', d => { delete d.uvs; }, /must carry uvs ArrayBuffer/],
  ['zero faces', d => { d.newNumFaces = 0; }, /newNumFaces invalid: 0/],
];
for (const [name, mutate, re] of uvFalsifiers) {
  const d = goodUv(); mutate(d);
  assert.throws(() => validateUvUnwrapReply(d), re, `${name} must throw with its reason`);
  console.log(`ok  uv-unwrap rejects: ${name}`);
}

console.log('\nWORKER REPLY VALIDATION CONTRACT PASSED');
