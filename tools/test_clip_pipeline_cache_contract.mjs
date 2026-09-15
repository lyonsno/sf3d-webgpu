#!/usr/bin/env node
/**
 * CLIP inline-pipeline cache contract (clip_estimator.js getClipInlinePipeline
 * over the kit's createWebGpuResourceCaches).
 *
 * The inline add / GELU / fused-attention shaders bake their sizes into the
 * WGSL and used to be compiled synchronously on the main thread once per block
 * per run (3 shaders × 12 blocks = 36 pipeline compiles per estimate). The
 * contract: one shader module + one compute pipeline per distinct
 * (label, code) per device, reused across calls and across "blocks"; distinct
 * code still gets its own pipeline; a second device gets its own cache.
 */
import assert from 'node:assert/strict';
import { getClipInlinePipeline } from '../src/lib/clip_estimator.js';

function makeFakeDevice() {
  const device = {
    shaderModules: 0,
    computePipelines: 0,
    createShaderModule(desc) { this.shaderModules += 1; return { kind: 'module', label: desc.label, code: desc.code }; },
    createComputePipeline(desc) {
      this.computePipelines += 1;
      assert.equal(desc.layout, 'auto');
      assert.equal(desc.compute.entryPoint, 'main');
      assert.equal(desc.compute.module.kind, 'module');
      return { kind: 'pipeline', label: desc.label, module: desc.compute.module };
    },
  };
  return device;
}

const codeA = '@compute @workgroup_size(256) fn main() { /* count=38400 */ }';
const codeB = '@compute @workgroup_size(256) fn main() { /* count=153600 */ }';

const device = makeFakeDevice();
const p1 = getClipInlinePipeline(device, 'sf3d.clip.add', codeA);
// "12 blocks" worth of repeat calls
for (let block = 0; block < 12; block++) {
  assert.equal(getClipInlinePipeline(device, 'sf3d.clip.add', codeA), p1, 'same pipeline object reused');
}
assert.equal(device.shaderModules, 1, 'one module for 13 calls');
assert.equal(device.computePipelines, 1, 'one pipeline for 13 calls');
console.log('ok  same (label, code) compiles once and is reused across blocks');

const p2 = getClipInlinePipeline(device, 'sf3d.clip.gelu', codeB);
assert.notEqual(p2, p1);
assert.equal(device.shaderModules, 2);
assert.equal(device.computePipelines, 2);
// Same label, different baked size → distinct pipeline (must not alias).
const p3 = getClipInlinePipeline(device, 'sf3d.clip.gelu', codeA);
assert.notEqual(p3, p2, 'different code under the same label must not alias');
assert.equal(device.computePipelines, 3);
console.log('ok  distinct code gets its own pipeline; label alone never aliases');

const other = makeFakeDevice();
getClipInlinePipeline(other, 'sf3d.clip.add', codeA);
assert.equal(other.computePipelines, 1, 'second device compiles its own');
assert.equal(device.computePipelines, 3, 'first device untouched');
console.log('ok  caches are per device');

console.log('\nCLIP PIPELINE CACHE CONTRACT PASSED');
