#!/usr/bin/env node
/**
 * Pure-helper contract for the producer's host-facing identity surfaces
 * (Wake host answers 4 and 5, 2026-09-16): explicit worker module URLs and
 * weights source, run/clock identity with effective device topology, and a
 * dispose that releases producer-loaded buffers but never injected ones.
 */
import assert from 'node:assert/strict';
import { productRouteWorkerModuleUrls, PRODUCT_ROUTE_WORKER_ROLES } from '../src/lib/product_route.js';
import {
  SF3D_PRODUCER_RUN_IDENTITY_SCHEMA,
  buildRunIdentity,
  releaseLoadedWeights,
  resolveWeightsUrl,
} from '../src/lib/sf3d_producer.js';

// Worker module URLs: one per role, absolute file/http URLs ending in the worker file.
const urls = productRouteWorkerModuleUrls();
assert.deepEqual(Object.keys(urls).sort(), Object.keys(PRODUCT_ROUTE_WORKER_ROLES).sort());
for (const [role, url] of Object.entries(urls)) {
  assert.match(url, /^(file|https?):\/\//, `${role} url is absolute`);
  assert.match(url, /_worker\.js$/, `${role} url names a worker module`);
}
console.log('ok  worker module urls explicit, one per role');

// Weights URL resolution.
assert.equal(resolveWeightsUrl('weights.bin', 'https://host.example/app/'), 'https://host.example/app/weights.bin');
assert.equal(resolveWeightsUrl('https://huggingface.co/x/weights.bin', 'https://host.example/'), 'https://huggingface.co/x/weights.bin');
assert.throws(() => resolveWeightsUrl(''), /non-empty/);
console.log('ok  weights url resolved against the host document');

// Run identity: clock + topology + revisions.
const id = buildRunIdentity({ runId: 'r1', routeId: 'sf3d.image-to-mesh.webgpu-local.v0', startedAtMs: 10, finishedAtMs: 52.5, deviceInjected: true, commit: 'abc', kitVersion: '0.1.48' });
assert.equal(id.schema, SF3D_PRODUCER_RUN_IDENTITY_SCHEMA);
assert.equal(id.deviceTopology, 'host-injected-device');
assert.equal(id.durationMs, 42.5); assert.equal(id.clock, 'performance.now');
assert.ok(Number.isFinite(id.timeOrigin), 'declared performance time origin');
assert.equal(buildRunIdentity({ runId: 'r2', routeId: 'x', startedAtMs: 0, finishedAtMs: 1, deviceInjected: false, commit: 'abc', kitVersion: '0.1.48' }).deviceTopology, 'producer-owned-device');
console.log('ok  run identity binds clock, topology, revisions');

// Dispose semantics: producer-loaded GPU buffers destroyed; CPU tensors and non-buffers untouched.
let destroyed = 0;
const weights = { a: { destroy() { destroyed += 1; } }, b: new Float32Array(4), c: { destroy() { destroyed += 1; } } };
assert.equal(releaseLoadedWeights(weights), 2);
assert.equal(destroyed, 2);
assert.equal(releaseLoadedWeights(null), 0);
console.log('ok  releaseLoadedWeights destroys only GPU buffers');

// loadWeights returns a component tree, not a flat map. Follow arrays and
// packed-resource wrappers, deduplicate shared buffers, and leave CPU payloads
// opaque rather than walking every element of a model-sized typed array.
const retired = [];
const buffer = name => ({ destroy() { retired.push(name); } });
const q = buffer('q');
const packed = buffer('packed');
const bias = buffer('bias');
const cpu = new Float32Array([1, 2, 3]);
Object.defineProperty(cpu, 'doNotEnumerate', {
  enumerable: true,
  get() { throw new Error('CPU tensor contents must remain opaque'); },
});
const nestedWeights = {
  imageTokenizer: { blocks: [{
    attn: { q: { weight: q, bias } },
    mlp: { fc1: { weight: { buffer: packed, representation: 'f16-packed-u32' } } },
  }] },
  cameraEmbedder: { bias },
  cpu,
  bytes: new ArrayBuffer(8),
  _rawGet() { throw new Error('release must not invoke lazy upload accessors'); },
};
nestedWeights.alias = nestedWeights.imageTokenizer;
nestedWeights.self = nestedWeights;
assert.equal(releaseLoadedWeights(nestedWeights), 3, 'all distinct nested weight buffers must retire');
assert.deepEqual(retired.sort(), ['bias', 'packed', 'q']);
assert.deepEqual([...cpu], [1, 2, 3]);
console.log('ok  nested component weights retire once; aliases, cycles, CPU tensors and lazy readers are safe');

console.log('\nSF3D PRODUCER HELPERS CONTRACT PASSED');
