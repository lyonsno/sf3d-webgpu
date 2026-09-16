#!/usr/bin/env node
/**
 * Injected-device contract for initGPU (gpu.js).
 *
 * A Kaminos host owns the live GPUDevice; SF3D must run on that exact device
 * rather than requesting its own. Fail-first invariants:
 *   1. initGPU({ device, adapter }) returns those exact objects, never touches
 *      navigator.gpu, and marks the result injected.
 *   2. An injected device without an adapter is accepted (adapter null) but a
 *      non-device object is rejected loudly.
 *   3. With nothing injected and no WebGPU, initGPU still fails loud (no
 *      silent fallback to a fake device).
 */
import assert from 'node:assert/strict';
import { initGPU } from '../src/lib/gpu.js';

// No navigator.gpu in Node: any attempt to request an adapter must throw.
// (Node defines navigator as a getter-only global; redefine it explicitly.)
Object.defineProperty(globalThis, 'navigator', { value: { gpu: undefined }, configurable: true, writable: true });

const fakeQueue = { submit() {}, onSubmittedWorkDone: async () => {} };
const fakeDevice = { queue: fakeQueue, lost: new Promise(() => {}), createBuffer() {}, limits: { maxBufferSize: 1 } };
const fakeAdapter = { limits: { maxBufferSize: 1 }, info: { vendor: 'fake' } };

// 1. Exact injected objects, no adapter request.
{
  const gpu = await initGPU({ device: fakeDevice, adapter: fakeAdapter });
  assert.equal(gpu.device, fakeDevice, 'injected device returned as-is');
  assert.equal(gpu.adapter, fakeAdapter, 'injected adapter returned as-is');
  assert.equal(gpu.injected, true);
  console.log('ok  injected device/adapter returned without requesting');
}

// 2. Adapter optional; non-device rejected.
{
  const gpu = await initGPU({ device: fakeDevice });
  assert.equal(gpu.device, fakeDevice);
  assert.equal(gpu.adapter, null);
  await assert.rejects(() => initGPU({ device: { notAQueue: true } }), /injected device must expose queue\.submit/);
  console.log('ok  adapter optional; non-device rejected');
}

// 3. Nothing injected + no WebGPU → loud failure, no fake device.
await assert.rejects(() => initGPU(), /WebGPU is not supported/);
console.log('ok  no injection and no WebGPU fails loud');

console.log('\nGPU INJECTION CONTRACT PASSED');
