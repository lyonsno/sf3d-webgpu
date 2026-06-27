/**
 * shader_ops.js — WebGPU compute dispatch wrappers for each shader.
 *
 * Each function creates a pipeline, binds buffers, and dispatches.
 * Pipelines are cached by device for reuse.
 */

import conv2dWGSL from '../shaders/conv2d.wgsl?raw';
import conv1x1WGSL from '../shaders/conv1x1.wgsl?raw';
import convTranspose2dWGSL from '../shaders/conv_transpose2d.wgsl?raw';
import activationsWGSL from '../shaders/activations.wgsl?raw';
import groupnormWGSL from '../shaders/groupnorm.wgsl?raw';
import pixelshuffleWGSL from '../shaders/pixelshuffle.wgsl?raw';
import upsampleWGSL from '../shaders/upsample.wgsl?raw';

import { createStorageBuffer, createEmptyBuffer } from './gpu.js';

const pipelineCache = new Map();
const uniformCache = new Map();
const MAX_WG_DIM = 65535;

function cachedUniform(device, data) {
  const bytes = new Uint8Array(data.buffer || data);
  let h = 0;
  for (let i = 0; i < bytes.length; i++) h = (h * 31 + bytes[i]) | 0;
  const key = `u_${bytes.length}_${h}`;
  if (uniformCache.has(key)) return uniformCache.get(key);
  const buf = device.createBuffer({
    size: Math.max(bytes.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(bytes);
  buf.unmap();
  uniformCache.set(key, buf);
  return buf;
}

// Cache for dummy bias buffers (one per device)
let dummyBiasBuf = null;
function getDummyBias(device) {
  if (!dummyBiasBuf) {
    dummyBiasBuf = createStorageBuffer(device, new Float32Array([0]));
  }
  return dummyBiasBuf;
}

/**
 * Split a total workgroup count into 2D dispatch (x, y) to stay within limits.
 * Returns [wgX, wgY] where wgX * wgY >= totalWG and wgX <= MAX_WG_DIM.
 */
function splitWorkgroups(totalWG) {
  if (totalWG <= MAX_WG_DIM) return [totalWG, 1];
  const wgX = MAX_WG_DIM;
  const wgY = Math.ceil(totalWG / MAX_WG_DIM);
  return [wgX, wgY];
}

function getOrCreatePipeline(device, key, code, entryPoint) {
  if (pipelineCache.has(key)) return pipelineCache.get(key);
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint },
  });
  pipelineCache.set(key, pipeline);
  return pipeline;
}

function ceil(a, b) { return Math.ceil(a / b); }

/**
 * Dispatch conv2d (3x3 or arbitrary kernel).
 * Returns output buffer [outC, outH, outW].
 */
export function dispatchConv2d(device, encoder, inputBuf, weightBuf, biasBuf, params) {
  const { inC, inH, inW, outC, kH, kW, padH, padW, strideH, strideW } = params;
  const outH = Math.floor((inH + 2 * padH - kH) / strideH) + 1;
  const outW = Math.floor((inW + 2 * padW - kW) / strideW) + 1;
  const hasBias = biasBuf ? 1 : 0;

  const pipeline = getOrCreatePipeline(device, 'conv2d', conv2dWGSL, 'conv2d_main');

  const uniformData = new Uint32Array([inC, inH, inW, outC, outH, outW, kH, kW, padH, padW, strideH, strideW, hasBias]);
  const uniformBuf = cachedUniform(device, uniformData);

  const dummyBias = biasBuf || getDummyBias(device);
  const outputBuf = createEmptyBuffer(device, outC * outH * outW * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: dummyBias } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(ceil(outW, 16), ceil(outH, 16), outC);
  pass.end();

  return { buffer: outputBuf, outC, outH, outW };
}

/**
 * Dispatch 1x1 conv.
 */
export function dispatchConv1x1(device, encoder, inputBuf, weightBuf, biasBuf, params) {
  const { inC, outC, H, W } = params;
  const hasBias = biasBuf ? 1 : 0;

  const pipeline = getOrCreatePipeline(device, 'conv1x1', conv1x1WGSL, 'conv1x1_main');

  const totalWG = ceil(outC * H * W, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([inC, outC, H, W, hasBias, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const dummyBias = biasBuf || getDummyBias(device);
  const outputBuf = createEmptyBuffer(device, outC * H * W * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: dummyBias } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return { buffer: outputBuf, C: outC, H, W };
}

/**
 * Dispatch element-wise activation.
 * op: 0=relu, 1=silu, 2=add, 3=add_relu, 4=sigmoid
 */
export function dispatchActivation(device, encoder, inputA, inputB, count, op) {
  const pipeline = getOrCreatePipeline(device, 'activation', activationsWGSL, 'activation_main');

  const totalWG = ceil(count, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([count, op, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const dummyB = inputB || getDummyBias(device);
  const outputBuf = createEmptyBuffer(device, count * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputA } },
      { binding: 2, resource: { buffer: dummyB } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return outputBuf;
}

/**
 * Dispatch GroupNorm (two-pass: stats then normalize).
 */
export function dispatchGroupNorm(device, encoder, inputBuf, scaleBuf, biasBuf, params) {
  const { C, H, W, numGroups, eps = 1e-5 } = params;

  const statsPipeline = getOrCreatePipeline(device, 'gn_stats', groupnormWGSL, 'groupnorm_stats');
  const normPipeline = getOrCreatePipeline(device, 'gn_norm', groupnormWGSL, 'groupnorm_normalize');

  // Uniform: C, H, W, numGroups, eps (f32), numWorkgroupsX (u32)
  const normTotalWG = ceil(C * H * W, 256);
  const [normWgX, normWgY] = splitWorkgroups(normTotalWG);
  const uniformArr = new ArrayBuffer(24);
  const u32View = new Uint32Array(uniformArr);
  const f32View = new Float32Array(uniformArr);
  u32View[0] = C; u32View[1] = H; u32View[2] = W; u32View[3] = numGroups;
  f32View[4] = eps;
  u32View[5] = normWgX;

  const uniformBuf = cachedUniform(device, new Uint8Array(uniformArr));

  const statsBuf = createEmptyBuffer(device, numGroups * 2 * 4);
  const outputBuf = createEmptyBuffer(device, C * H * W * 4);

  // Pass 1: compute stats
  const statsBindGroup = device.createBindGroup({
    layout: statsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: scaleBuf } },
      { binding: 3, resource: { buffer: biasBuf } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: statsBuf } },
    ],
  });

  const pass1 = encoder.beginComputePass();
  pass1.setPipeline(statsPipeline);
  pass1.setBindGroup(0, statsBindGroup);
  pass1.dispatchWorkgroups(ceil(numGroups, 256));
  pass1.end();

  // Pass 2: normalize
  const normBindGroup = device.createBindGroup({
    layout: normPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: scaleBuf } },
      { binding: 3, resource: { buffer: biasBuf } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: statsBuf } },
    ],
  });

  const pass2 = encoder.beginComputePass();
  pass2.setPipeline(normPipeline);
  pass2.setBindGroup(0, normBindGroup);
  pass2.dispatchWorkgroups(normWgX, normWgY);
  pass2.end();

  return outputBuf;
}

/**
 * Dispatch PixelShuffle.
 */
export function dispatchPixelShuffle(device, encoder, inputBuf, params) {
  const { inC, inH, inW, scaleFactor } = params;
  const outC = inC / (scaleFactor * scaleFactor);
  const outH = inH * scaleFactor;
  const outW = inW * scaleFactor;

  const pipeline = getOrCreatePipeline(device, 'pixelshuffle', pixelshuffleWGSL, 'pixelshuffle_main');

  const totalWG = ceil(outC * outH * outW, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([inC, inH, inW, outC, scaleFactor, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const outputBuf = createEmptyBuffer(device, outC * outH * outW * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return { buffer: outputBuf, C: outC, H: outH, W: outW };
}

/**
 * Dispatch bilinear/nearest upsample.
 */
export function dispatchUpsample(device, encoder, inputBuf, params) {
  const { C, inH, inW, outH, outW, mode = 1 } = params; // mode: 0=nearest, 1=bilinear

  const pipeline = getOrCreatePipeline(device, 'upsample', upsampleWGSL, 'upsample_main');

  const totalWG = ceil(C * outH * outW, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([C, inH, inW, outH, outW, mode, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const outputBuf = createEmptyBuffer(device, C * outH * outW * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return { buffer: outputBuf, C, H: outH, W: outW };
}

/**
 * Dispatch transposed 2D convolution (deconvolution).
 * ConvTranspose2d(inC, outC, kernel_size=stride, stride=stride)
 */
export function dispatchConvTranspose2d(device, encoder, inputBuf, weightBuf, biasBuf, params) {
  const { inC, inH, inW, outC, stride } = params;
  const kH = stride, kW = stride;
  const outH = inH * stride;
  const outW = inW * stride;
  const hasBias = biasBuf ? 1 : 0;

  const pipeline = getOrCreatePipeline(device, 'conv_transpose2d', convTranspose2dWGSL, 'conv_transpose2d_main');

  const totalWG = ceil(outC * outH * outW, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([inC, inH, inW, outC, outH, outW, kH, kW, stride, stride, hasBias, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const dummyBias = biasBuf || getDummyBias(device);
  const outputBuf = createEmptyBuffer(device, outC * outH * outW * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: dummyBias } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return { buffer: outputBuf, C: outC, H: outH, W: outW };
}
