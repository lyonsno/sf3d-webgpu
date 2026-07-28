/**
 * post_processor.js — PixelShuffle upsampling network for SF3D triplane features.
 *
 * Architecture (from network.py PixelShuffleUpsampleNetwork):
 *   Input: [1024, 27648] channel-first triplane features from backbone
 *   Reshape to 3 planes: [3, 1024, 96, 96]
 *   Per-plane processing (shared weights):
 *     Conv2d(1024→1024, 3×3, pad=1) + ReLU  ×3
 *     Conv2d(1024→640, 3×3, pad=1)           (640 = 40 * 4²)
 *     PixelShuffle(4)                         → [40, 384, 384]
 *   Output: [3, 40, 384, 384] triplane feature planes
 */

import { createEmptyBuffer } from './gpu.js';
import { dispatchConv2d, dispatchActivation, dispatchPixelShuffle } from './shader_ops.js';

const POST_CONFIG = {
  inChannels: 1024,
  outChannels: 40,
  scaleFactor: 4,
  convLayers: 4,
  kernelSize: 3,
  planeSize: 96,
  // output_channels = out_channels * scale_factor² = 40 * 16 = 640
  lastConvOut: 640,
};

export const POST_PROCESSOR_PLANE_STAGE_IDS = Object.freeze([
  'gather',
  'conv-0-relu',
  'conv-1-relu',
  'conv-2-relu',
  'conv-3',
  'pixel-shuffle-copy',
]);

/**
 * Dispatch the post-processor for all 3 triplane planes.
 *
 * @param {GPUDevice} device
 * @param {GPUCommandEncoder} encoder
 * @param {GPUBuffer} triplanesBuf - [1024, 27648] channel-first backbone output
 * @param {Object} weights - post_processor weights from loadWeights
 * @returns {{ buffer: GPUBuffer, C: number, H: number, W: number }} - [3, 40, 384, 384]
 */
export function createPostProcessorOutput(device) {
  const { outChannels, scaleFactor, planeSize } = POST_CONFIG;
  const outH = planeSize * scaleFactor; // 384
  const outW = planeSize * scaleFactor; // 384
  const outPlaneElements = outChannels * outH * outW;

  // Output buffer for all 3 planes: [3, 40, 384, 384]
  const outputBuf = createEmptyBuffer(device, 3 * outPlaneElements * 4);
  return {
    buffer: outputBuf,
    C: outChannels,
    H: outH,
    W: outW,
    numPlanes: 3,
  };
}

/**
 * Encode one dependency-independent triplane plane into a caller-owned output.
 *
 * The four convolutions inside a plane remain sequential and therefore stay in
 * one command buffer for this first cooperative cut. Different planes share
 * weights but no intermediate buffers, so command boundaries between planes do
 * not change numerical ordering or output layout.
 */
export function dispatchPostProcessorPlane(
  device,
  encoder,
  triplanesBuf,
  weights,
  output,
  plane,
) {
  const state = createPostProcessorPlaneState(
    device,
    triplanesBuf,
    weights,
    output,
    plane,
  );
  for (let stageIndex = 0; stageIndex < POST_PROCESSOR_PLANE_STAGE_IDS.length; stageIndex++) {
    dispatchPostProcessorPlaneStage(device, encoder, state, stageIndex);
  }
}

/**
 * Create the persistent state that crosses command boundaries for one plane.
 *
 * Buffer allocation remains inside the stage that first produces each value,
 * so constructing state does not silently perform GPU work or front-load memory.
 */
export function createPostProcessorPlaneState(
  device,
  triplanesBuf,
  weights,
  output,
  plane,
) {
  if (!Number.isSafeInteger(plane) || plane < 0 || plane >= 3) {
    throw new RangeError('postprocessor plane must be an integer in [0, 3)');
  }
  if (!device || !triplanesBuf || !weights || !output?.buffer) {
    throw new TypeError('postprocessor plane state requires device, input, weights, and output');
  }
  if (!Array.isArray(weights.convLayers) || weights.convLayers.length !== POST_CONFIG.convLayers) {
    throw new RangeError(`postprocessor requires exactly ${POST_CONFIG.convLayers} convolution layers`);
  }
  return {
    device,
    triplanesBuf,
    weights,
    output,
    plane,
    nextStageIndex: 0,
    current: null,
    complete: false,
  };
}

/**
 * Encode one exact stage for one plane.
 *
 * The caller may invoke all stages into one encoder (legacy route) or finish
 * and submit after each stage (cooperative route). State enforces the dependency
 * order so skipped, repeated, or out-of-order command duties fail loud.
 */
export function dispatchPostProcessorPlaneStage(device, encoder, state, stageIndex) {
  if (device !== state?.device) {
    throw new Error('postprocessor plane state belongs to a different GPUDevice');
  }
  if (!Number.isSafeInteger(stageIndex)
      || stageIndex < 0
      || stageIndex >= POST_PROCESSOR_PLANE_STAGE_IDS.length) {
    throw new RangeError(
      `postprocessor stage must be an integer in [0, ${POST_PROCESSOR_PLANE_STAGE_IDS.length})`,
    );
  }
  if (stageIndex !== state.nextStageIndex) {
    throw new Error(
      `postprocessor plane ${state.plane} expected stage ${state.nextStageIndex}, got ${stageIndex}`,
    );
  }

  const {
    inChannels,
    scaleFactor,
    planeSize,
    lastConvOut,
  } = POST_CONFIG;
  const planePixels = planeSize * planeSize;
  const planeElements = inChannels * planePixels;

  if (stageIndex === 0) {
    const planeBuf = createEmptyBuffer(device, planeElements * 4);
    _dispatchGatherPlane(
      device,
      encoder,
      state.triplanesBuf,
      planeBuf,
      inChannels,
      planeSize * planeSize * 3,
      state.plane * planePixels,
      planePixels,
    );
    state.current = {
      buffer: planeBuf,
      outC: inChannels,
      outH: planeSize,
      outW: planeSize,
    };
  } else if (stageIndex <= 4) {
    const layerIndex = stageIndex - 1;
    const isLast = layerIndex === POST_CONFIG.convLayers - 1;
    const curOutC = isLast ? lastConvOut : inChannels;
    const convResult = dispatchConv2d(
      device,
      encoder,
      state.current.buffer,
      state.weights.convLayers[layerIndex].weight,
      state.weights.convLayers[layerIndex].bias,
      {
        inC: state.current.outC,
        inH: state.current.outH,
        inW: state.current.outW,
        outC: curOutC,
        kH: 3,
        kW: 3,
        padH: 1,
        padW: 1,
        strideH: 1,
        strideW: 1,
      },
    );
    if (isLast) {
      state.current = {
        buffer: convResult.buffer,
        outC: curOutC,
        outH: convResult.outH,
        outW: convResult.outW,
      };
    } else {
      const reluBuf = dispatchActivation(
        device,
        encoder,
        convResult.buffer,
        null,
        curOutC * convResult.outH * convResult.outW,
        0,
      );
      state.current = {
        buffer: reluBuf,
        outC: curOutC,
        outH: convResult.outH,
        outW: convResult.outW,
      };
    }
  } else {
    const psResult = dispatchPixelShuffle(device, encoder, state.current.buffer, {
      inC: lastConvOut,
      inH: planeSize,
      inW: planeSize,
      scaleFactor,
    });
    const outPlaneElements = state.output.C * state.output.H * state.output.W;
    const outOffset = state.plane * outPlaneElements * 4;
    encoder.copyBufferToBuffer(
      psResult.buffer,
      0,
      state.output.buffer,
      outOffset,
      outPlaneElements * 4,
    );
    state.complete = true;
  }

  state.nextStageIndex++;
  return {
    plane: state.plane,
    stageIndex,
    stageId: POST_PROCESSOR_PLANE_STAGE_IDS[stageIndex],
    complete: state.complete,
  };
}

export function dispatchPostProcessor(device, encoder, triplanesBuf, weights) {
  const output = createPostProcessorOutput(device);

  // Preserve the legacy route exactly: all three planes are still encoded into
  // the caller's one command encoder and submitted by the caller as one buffer.
  for (let plane = 0; plane < output.numPlanes; plane++) {
    dispatchPostProcessorPlane(device, encoder, triplanesBuf, weights, output, plane);
  }

  return output;
}

// --- Internal: gather one triplane plane from strided layout ---

let _gatherPipeline = null;

function _dispatchGatherPlane(device, encoder, srcBuf, dstBuf, numChannels, totalSpatial, spatialOffset, spatialSize) {
  // Gather: for each channel c and spatial index s in [0, spatialSize):
  //   dst[c * spatialSize + s] = src[c * totalSpatial + spatialOffset + s]
  if (!_gatherPipeline) {
    _gatherPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: `
            struct P { numChannels: u32, totalSpatial: u32, spatialOffset: u32, spatialSize: u32, numWgX: u32 }
            @group(0) @binding(0) var<uniform> p: P;
            @group(0) @binding(1) var<storage, read> src: array<f32>;
            @group(0) @binding(2) var<storage, read_write> dst: array<f32>;
            @compute @workgroup_size(256)
            fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
              let idx = (wgid.x + wgid.y * p.numWgX) * 256u + lid.x;
              let total = p.numChannels * p.spatialSize;
              if (idx >= total) { return; }
              let c = idx / p.spatialSize;
              let s = idx % p.spatialSize;
              dst[c * p.spatialSize + s] = src[c * p.totalSpatial + p.spatialOffset + s];
            }
          `,
        }),
        entryPoint: 'main',
      },
    });
  }

  const total = numChannels * spatialSize;
  const totalWG = Math.ceil(total / 256);
  const wgX = Math.min(totalWG, 65535);
  const wgY = Math.ceil(totalWG / 65535);

  const params = device.createBuffer({
    size: 20,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(params.getMappedRange()).set([numChannels, totalSpatial, spatialOffset, spatialSize, wgX]);
  params.unmap();

  const bg = device.createBindGroup({
    layout: _gatherPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: srcBuf } },
      { binding: 2, resource: { buffer: dstBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(_gatherPipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();
}
