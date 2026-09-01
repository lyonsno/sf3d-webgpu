import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { SF3DImageTokenizer } from '../src/lib/sf3d_backbone.js';
import { extractTensor } from '../src/lib/weights.js';

globalThis.GPUBufferUsage = globalThis.GPUBufferUsage || {
  STORAGE: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
  UNIFORM: 8,
};

function createMappedDevice() {
  const allocations = [];
  return {
    allocations,
    createBuffer(descriptor) {
      const storage = new ArrayBuffer(descriptor.size);
      const buffer = {
        descriptor,
        getMappedRange: () => storage,
        unmap() {},
      };
      allocations.push(buffer);
      return buffer;
    },
  };
}

{
  const device = createMappedDevice();
  const fp16Bits = new Uint16Array([0x3c00, 0xc000, 0x3555, 0x7bff, 0x0001]);
  const resource = extractTensor(device, fp16Bits.buffer, {
    dtype: 1,
    offset: 0,
    size: fp16Bits.byteLength,
    shape: [1, fp16Bits.length],
  }, {
    representation: 'f16-packed-u32',
  });

  assert.equal(resource.representation, 'f16-packed-u32');
  assert.equal(resource.sourceDtype, 'fp16');
  assert.equal(resource.elementCount, fp16Bits.length);
  assert.equal(resource.sourceByteLength, 10);
  assert.equal(resource.storageByteLength, 12);
  assert.equal(resource.expandedFp32ByteLength, 20);
  assert.equal(resource.buffer.descriptor.size, 12);
  assert.deepEqual(
    [...new Uint32Array(resource.buffer.getMappedRange())],
    [0xc0003c00, 0x7bff3555, 0x00000001],
  );
}

function createPipelineDevice(shaderCodes) {
  return {
    createShaderModule({ code }) {
      shaderCodes.push(code);
      return { code };
    },
    createComputePipeline({ compute }) {
      return {
        code: compute.module.code,
        getBindGroupLayout: () => ({}),
      };
    },
  };
}

{
  const shaderCodes = [];
  const tokenizer = new SF3DImageTokenizer(createPipelineDevice(shaderCodes));
  tokenizer.init();
  assert.ok(tokenizer.pipelines.linearPacked, 'tokenizer must compile packed linear');
  assert.ok(tokenizer.pipelines.linearGeluPacked, 'tokenizer must compile packed linear+GELU');
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  for (const shaderName of ['linear_f16_packed.wgsl', 'linear_gelu_f16_packed.wgsl']) {
    const shader = fs.readFileSync(path.join(root, 'src/shaders', shaderName), 'utf8');
    assert.match(shader, /var<storage, read> weight: array<u32>;/);
    assert.match(shader, /unpack2x16float/);
    assert.match(shader, /weight\[index >> 1u\]/);
    assert.match(shader, /select\(pair\.x, pair\.y, \(index & 1u\) == 1u\)/);
  }
}

function createDispatchHarness() {
  const bindGroups = [];
  const pipelines = [];
  const device = {
    createBindGroup({ layout, entries }) {
      const bindGroup = { layout, entries };
      bindGroups.push(bindGroup);
      return bindGroup;
    },
  };
  const encoder = {
    beginComputePass() {
      return {
        setPipeline(pipeline) { pipelines.push(pipeline); },
        setBindGroup() {},
        dispatchWorkgroups() {},
        end() {},
      };
    },
  };
  return { device, encoder, bindGroups, pipelines };
}

for (const method of ['_dispatchLinear', '_dispatchLinearGelu']) {
  const harness = createDispatchHarness();
  const tokenizer = Object.create(SF3DImageTokenizer.prototype);
  tokenizer.device = harness.device;
  tokenizer._cachedUniform = () => ({ id: 'params' });
  tokenizer.pipelines = {
    linear: { id: 'linear-f32', getBindGroupLayout: () => ({ id: 'linear-f32-layout' }) },
    linearPacked: { id: 'linear-packed', getBindGroupLayout: () => ({ id: 'linear-packed-layout' }) },
    linearGelu: { id: 'linear-gelu-f32', getBindGroupLayout: () => ({ id: 'linear-gelu-f32-layout' }) },
    linearGeluPacked: { id: 'linear-gelu-packed', getBindGroupLayout: () => ({ id: 'linear-gelu-packed-layout' }) },
  };
  const packedBuffer = { id: 'packed-buffer' };
  const weight = { representation: 'f16-packed-u32', buffer: packedBuffer };
  tokenizer[method](
    harness.encoder,
    { id: 'input' },
    { id: 'output' },
    weight,
    { id: 'bias' },
    3,
    5,
    7,
  );

  assert.equal(harness.bindGroups[0].entries[2].resource.buffer, packedBuffer);
  assert.equal(
    harness.pipelines[0].id,
    method === '_dispatchLinear' ? 'linear-packed' : 'linear-gelu-packed',
  );
}

console.log('image tokenizer packed-weight contracts passed');
