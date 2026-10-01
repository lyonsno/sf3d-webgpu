import assert from 'node:assert/strict';
import { mock } from 'node:test';
let scope, fences = 0;
const buffers = [];
const alloc = size => {
  const buffer = { size, destroyed: false, destroy() { this.destroyed = true; } };
  scope.push({ buffer, size }); buffers.push(buffer); return buffer;
};
mock.module('../src/lib/gpu.js', { namedExports: {
  createEmptyBuffer: (_device, size) => alloc(size),
  createStorageBuffer: (_device, data) => alloc(data.byteLength),
  captureGpuBufferAllocations: fn => { scope = []; return { value: fn(), allocations: scope }; },
} });
mock.module('../src/lib/shader_ops.js', { namedExports: { dispatchConv2dChannelRange() {} } });
globalThis.GPUBufferUsage = { UNIFORM: 64 };
const { streamPostProcessor } = await import('../src/lib/post_processor_spatial.js');
const device = {
  createShaderModule() {}, createComputePipeline: () => ({ getBindGroupLayout() {} }), createBindGroup() {},
  createCommandEncoder: () => ({ finish() {}, beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }) }),
  queue: { submit() {}, async onSubmittedWorkDone() { if (++fences === 4) throw new Error('settlement-secondary'); } },
};
const primary = new Error('observer-primary');
await assert.rejects(streamPostProcessor(device, {}, { convLayers: Array.from({ length: 4 }, () => ({})) }, {
  config: { channels: 1, outChannels: 1, size: 4, scale: 1 }, rowsPerRegion: 2,
  onRegion() { throw primary; },
}), error => {
  assert.equal(error, primary, 'cleanup must preserve the initiating failure');
  assert.equal(error.previewSettlementError.message, 'settlement-secondary');
  return true;
});
assert.ok(buffers.every(buffer => buffer.destroyed), 'every owned buffer must retire despite both errors');
console.log('Spatial preview preserves primary and settlement failures and retires buffers');
