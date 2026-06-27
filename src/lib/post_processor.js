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

/**
 * Dispatch the post-processor for all 3 triplane planes.
 *
 * @param {GPUDevice} device
 * @param {GPUCommandEncoder} encoder
 * @param {GPUBuffer} triplanesBuf - [1024, 27648] channel-first backbone output
 * @param {Object} weights - post_processor weights from loadWeights
 * @returns {{ buffer: GPUBuffer, C: number, H: number, W: number }} - [3, 40, 384, 384]
 */
export function dispatchPostProcessor(device, encoder, triplanesBuf, weights) {
  const { inChannels, outChannels, scaleFactor, planeSize, lastConvOut } = POST_CONFIG;
  const planePixels = planeSize * planeSize; // 9216
  const planeElements = inChannels * planePixels; // 1024 * 9216
  const outH = planeSize * scaleFactor; // 384
  const outW = planeSize * scaleFactor; // 384
  const outPlaneElements = outChannels * outH * outW;

  // Output buffer for all 3 planes: [3, 40, 384, 384]
  const outputBuf = createEmptyBuffer(device, 3 * outPlaneElements * 4);

  // Process each of the 3 triplane planes with shared weights
  for (let plane = 0; plane < 3; plane++) {
    // Extract one plane: offset into [1024, 27648] where each channel row
    // has 27648 spatial elements = 3 × 96 × 96. Plane i occupies columns
    // [i*9216, (i+1)*9216) within each channel row.
    //
    // But the data layout is [C, 3*96*96] contiguous, meaning plane 0 is
    // channels × [0..9215], plane 1 is channels × [9216..18431], etc.
    // We can use a sub-buffer view for each plane.
    //
    // Actually, the backbone output is [1024, 27648] in channel-first
    // (C-major) layout: element [c, s] = buf[c * 27648 + s].
    // Plane p occupies spatial [p*9216..(p+1)*9216).
    // We need [1024, 9216] = [C, H*W] for Conv2d.
    //
    // We need to gather plane p's spatial slice. This is a strided copy
    // that WebGPU copyBufferToBuffer can't do directly (non-contiguous).
    // Use a simple gather shader or accept the overhead of a full copy.
    // For now, use a gather dispatch.

    const planeBuf = createEmptyBuffer(device, planeElements * 4);
    _dispatchGatherPlane(device, encoder, triplanesBuf, planeBuf,
      inChannels, planeSize * planeSize * 3, plane * planePixels, planePixels);

    // Run 4 conv layers
    let current = { buffer: planeBuf, outC: inChannels, outH: planeSize, outW: planeSize };

    for (let i = 0; i < 4; i++) {
      const isLast = (i === 3);
      const curOutC = isLast ? lastConvOut : inChannels;

      const convResult = dispatchConv2d(device, encoder, current.buffer,
        weights.convs[i].weight, weights.convs[i].bias, {
          inC: current.outC,
          inH: current.outH,
          inW: current.outW,
          outC: curOutC,
          kH: 3, kW: 3,
          padH: 1, padW: 1,
          strideH: 1, strideW: 1,
        });

      if (!isLast) {
        // ReLU activation
        const reluBuf = dispatchActivation(device, encoder,
          convResult.buffer, null, curOutC * convResult.outH * convResult.outW, 0);
        current = { buffer: reluBuf, outC: curOutC, outH: convResult.outH, outW: convResult.outW };
      } else {
        current = { buffer: convResult.buffer, outC: curOutC, outH: convResult.outH, outW: convResult.outW };
      }
    }

    // PixelShuffle: [640, 96, 96] → [40, 384, 384]
    const psResult = dispatchPixelShuffle(device, encoder, current.buffer, {
      inC: lastConvOut,
      inH: planeSize,
      inW: planeSize,
      scaleFactor,
    });

    // Copy this plane's output to the correct offset in the combined output buffer
    const outOffset = plane * outPlaneElements * 4;
    encoder.copyBufferToBuffer(psResult.buffer, 0, outputBuf, outOffset, outPlaneElements * 4);
  }

  return { buffer: outputBuf, C: outChannels, H: outH, W: outW, numPlanes: 3 };
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
