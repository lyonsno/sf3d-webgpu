#!/usr/bin/env node
/**
 * CLIP prep contract (clip_prep_core.js + the worker dispatcher).
 *
 * Fail-first invariants:
 *   1. patchEmbedClip: CLS row is exactly classEmb + posEmb[0]; a patch/dim
 *      entry equals an independently computed brute-force sum + posEmb.
 *   2. blendClipPixels matches the exact PyTorch order (float, grey blend,
 *      multiply by mask) on hand-computed pixels; wrong byte length rejects.
 *   3. prepareClipEmbeddings is deterministic and byte-identical across calls.
 *   4. The worker dispatcher (runClipPrep) delegates exactly one request per
 *      call after a one-time init, returns embeddings byte-identical to the
 *      main-thread path, and rejects malformed replies instead of passing bad
 *      embeddings downstream.
 */
import assert from 'node:assert/strict';
import {
  CLIP_HIDDEN_DIM, CLIP_IMAGE_SIZE, CLIP_NUM_TOKENS, CLIP_PATCH_DIM, CLIP_PATCH_SIZE,
  blendClipPixels, patchEmbedClip, prepareClipEmbeddings, preprocessForClip,
  validateClipEmbeddings, validateClipPrepWeights,
} from '../src/lib/clip_prep_core.js';
import { runClipPrep } from '../src/lib/clip_estimator.js';

// Deterministic pseudo-random (LCG) so the contract is stable run to run.
function lcg(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 2 ** 32; }; }
function fill(n, rnd, scale = 1) { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = (rnd() * 2 - 1) * scale; return a; }

const rnd = lcg(7);
const weights = validateClipPrepWeights({
  conv1W: fill(CLIP_HIDDEN_DIM * CLIP_PATCH_DIM, rnd, 0.05),
  classEmb: fill(CLIP_HIDDEN_DIM, rnd),
  posEmb: fill(CLIP_NUM_TOKENS * CLIP_HIDDEN_DIM, rnd),
});

// 1. Patch embedding structure.
{
  const image = fill(3 * CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE, rnd);
  const out = patchEmbedClip(image, weights);
  assert.equal(out.length, CLIP_NUM_TOKENS * CLIP_HIDDEN_DIM);
  for (let d = 0; d < CLIP_HIDDEN_DIM; d++) {
    assert.equal(out[d], Math.fround(weights.classEmb[d] + weights.posEmb[d]), `CLS dim ${d}`);
  }
  // Independent brute force for patch (py=3, px=5), d=17.
  const py = 3, px = 5, d = 17, S = CLIP_IMAGE_SIZE, P = CLIP_PATCH_SIZE;
  let sum = 0;
  for (let c = 0; c < 3; c++) for (let dy = 0; dy < P; dy++) for (let dx = 0; dx < P; dx++) {
    sum += image[c * S * S + (py * P + dy) * S + (px * P + dx)]
         * weights.conv1W[d * CLIP_PATCH_DIM + c * P * P + dy * P + dx];
  }
  const token = py * 7 + px + 1;
  const expected = Math.fround(Math.fround(sum) + weights.posEmb[token * CLIP_HIDDEN_DIM + d]);
  assert.equal(out[token * CLIP_HIDDEN_DIM + d], expected, 'patch (3,5) dim 17 brute force');
  assert.throws(() => patchEmbedClip(new Float32Array(10), weights), /clip image must be/);
  console.log('ok  patch embedding: CLS row + brute-force patch entry match');
}

// 2. Blend order on hand-computed pixels.
{
  // one opaque pixel, one half-alpha pixel, one transparent pixel, one solid grey
  const rgba8 = new Uint8ClampedArray([
    255, 0, 0, 255,      // opaque red   → (1,0,0)*1 = (1,0,0)
    255, 0, 0, 128,      // alpha 128/255: a≈0.50196; r=(1*a+0.5*(1-a))*a
    255, 255, 255, 0,    // transparent  → all 0 after multiply by mask
    128, 128, 128, 255,  // grey         → 128/255
  ]);
  const out = blendClipPixels(rgba8, 2, 2);
  assert.equal(out.length, 16);
  assert.equal(out[0], 1); assert.equal(out[1], 0); assert.equal(out[2], 0); assert.equal(out[3], 1);
  const a = Math.fround(128 / 255);
  // JS evaluates the blend in double precision; only the Float32Array store rounds.
  assert.equal(out[4], Math.fround((1 * a + 0.5 * (1 - a)) * a));
  assert.equal(out[8], 0); assert.equal(out[9], 0); assert.equal(out[10], 0);
  assert.equal(out[12], Math.fround(128 / 255));
  assert.throws(() => blendClipPixels(new Uint8ClampedArray(15), 2, 2), /must be 16 RGBA bytes/);
  console.log('ok  alpha-blend matches the PyTorch order');
}

// 3. Determinism of the full prep on a synthetic 512x512 cond image.
let mainEmbeddings;
const W = 512, H = 512;
const rgba = new Uint8ClampedArray(W * H * 4);
{
  const r = lcg(11);
  for (let i = 0; i < rgba.length; i++) rgba[i] = Math.floor(r() * 256);
  const a = prepareClipEmbeddings(rgba, W, H, weights);
  const b = prepareClipEmbeddings(rgba, W, H, weights);
  assert.equal(Buffer.compare(Buffer.from(a.buffer), Buffer.from(b.buffer)), 0, 'deterministic');
  validateClipEmbeddings(a);
  // Also: preprocessForClip on non-cond size resizes first (no throw, right shape).
  const small = preprocessForClip(blendClipPixels(new Uint8ClampedArray(4 * 4 * 4).fill(200), 4, 4), 4, 4);
  assert.equal(small.length, 3 * CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE);
  mainEmbeddings = a;
  console.log('ok  prepareClipEmbeddings deterministic + shape-valid');
}

// 4. Worker dispatcher against a fake worker that runs the real core.
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
{
  let initialized = null;
  const worker = makeFakeWorker((self, msg) => {
    if (msg.type === 'init') {
      initialized = validateClipPrepWeights({
        conv1W: new Float32Array(msg.conv1W), classEmb: new Float32Array(msg.classEmb), posEmb: new Float32Array(msg.posEmb),
      });
      queueMicrotask(() => self._emit('message', { data: { id: msg.id, ok: true, initialized: true } }));
      return;
    }
    const emb = prepareClipEmbeddings(new Uint8ClampedArray(msg.rgba), msg.width, msg.height, initialized);
    queueMicrotask(() => self._emit('message', { data: { id: msg.id, ok: true, embeddings: emb.buffer } }));
  });
  const first = await runClipPrep(worker, rgba, W, H, weights, { timeoutMs: 5000 });
  assert.equal(Buffer.compare(Buffer.from(first.buffer), Buffer.from(mainEmbeddings.buffer)), 0,
    'worker embeddings byte-identical to main thread');
  assert.equal(worker.posted.filter(p => p.msg.type === 'init').length, 1, 'one init');
  assert.equal(worker.posted.filter(p => p.msg.type !== 'init').length, 1, 'one prep request');
  // Caller's rgba must survive (we transfer a copy, not the caller's buffer).
  assert.equal(rgba.byteLength, W * H * 4, 'caller rgba not detached');
  const second = await runClipPrep(worker, rgba, W, H, weights, { timeoutMs: 5000 });
  assert.equal(worker.posted.filter(p => p.msg.type === 'init').length, 1, 'init not repeated');
  assert.equal(Buffer.compare(Buffer.from(second.buffer), Buffer.from(mainEmbeddings.buffer)), 0);
  console.log('ok  runClipPrep: one-time init, one request per call, byte-identical');
}
{
  const withMarker = (marker) => ({
    conv1W: weights.conv1W,
    classEmb: Object.assign(weights.classEmb.slice(), { 0: marker }),
    posEmb: weights.posEmb,
  });
  let installedMarker = null;
  const worker = makeFakeWorker((self, msg) => {
    if (msg.type === 'init') {
      installedMarker = new Float32Array(msg.classEmb)[0];
      queueMicrotask(() => self._emit('message', { data: { id: msg.id, ok: true, initialized: true } }));
      return;
    }
    const embeddings = new Float32Array(CLIP_NUM_TOKENS * CLIP_HIDDEN_DIM);
    embeddings[0] = installedMarker;
    queueMicrotask(() => self._emit('message', { data: { id: msg.id, ok: true, embeddings: embeddings.buffer } }));
  });
  const firstWeights = withMarker(1);
  const secondWeights = withMarker(2);
  const first = await runClipPrep(worker, new Uint8ClampedArray(4), 1, 1, firstWeights, { timeoutMs: 5000 });
  const second = await runClipPrep(worker, new Uint8ClampedArray(4), 1, 1, secondWeights, { timeoutMs: 5000 });
  assert.equal(first[0], 1, 'first preparation request uses its initialized tensors');
  assert.equal(second[0], 2,
    'a replacement weight identity must reinitialize the shared worker before its preparation request');
  assert.equal(worker.posted.filter(p => p.msg.type === 'init').length, 2,
    'one worker receives one initialization per distinct weight identity');
  console.log('ok  runClipPrep rebinds a shared worker for replacement weights');
}
{
  const withMarker = (marker) => ({
    conv1W: weights.conv1W,
    classEmb: Object.assign(weights.classEmb.slice(), { 0: marker }),
    posEmb: weights.posEmb,
  });
  let installedMarker = null;
  const worker = makeFakeWorker((self, msg) => {
    const reply = data => queueMicrotask(() => self._emit('message', { data: { id: msg.id, ok: true, ...data } }));
    if (msg.type === 'init') { installedMarker = new Float32Array(msg.classEmb)[0]; reply({ initialized: true }); return; }
    const embeddings = new Float32Array(CLIP_NUM_TOKENS * CLIP_HIDDEN_DIM);
    embeddings[0] = installedMarker;
    reply({ embeddings: embeddings.buffer });
  });
  const [first, second] = await Promise.all([
    runClipPrep(worker, new Uint8ClampedArray(4), 1, 1, withMarker(3), { timeoutMs: 5000 }),
    runClipPrep(worker, new Uint8ClampedArray(4), 1, 1, withMarker(4), { timeoutMs: 5000 }),
  ]);
  assert.equal(first[0], 3, 'the first queued request cannot be rebound underneath itself');
  assert.equal(second[0], 4, 'the replacement waits for the first request, then uses its own tensors');
  assert.equal(worker.posted.filter(p => p.msg.type === 'init').length, 2,
    'concurrent replacement requests serialize distinct worker generations');
  console.log('ok  runClipPrep serializes concurrent replacement-weight requests');
}
{
  const worker = makeFakeWorker((self, msg) => {
    if (msg.type === 'init') { queueMicrotask(() => self._emit('message', { data: { id: msg.id, ok: true, initialized: true } })); return; }
    queueMicrotask(() => self._emit('message', { data: { id: msg.id, ok: true, embeddings: new ArrayBuffer(16) } }));
  });
  await assert.rejects(
    () => runClipPrep(worker, rgba, W, H, weights, { timeoutMs: 5000 }),
    /worker output invalid: clip embeddings must be/,
    'malformed reply must reject, never pass bad embeddings downstream');
  console.log('ok  runClipPrep rejects malformed worker output');
}

console.log('\nCLIP PREP CONTRACT PASSED');
