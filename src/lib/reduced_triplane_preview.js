import { captureGpuBufferAllocations, createEmptyBuffer, createStorageBuffer } from './gpu.js';
import { dispatchActivation, dispatchConv2d, dispatchPixelShuffle } from './shader_ops.js';
import { createReducedTriplanePlan } from './preview_geometry.js';
import { decodePreviewMesh } from './preview_geometry_gpu.js';

const pipelineByDevice = new WeakMap();
const GATHER_SHADER = `
struct P { sourceSize: u32, previewSize: u32, factor: u32, plane: u32, channels: u32, numWgX: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> source: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let i = (wgid.x + wgid.y * p.numWgX) * 256u + lid.x;
  let pixels = p.previewSize * p.previewSize;
  if (i >= p.channels * pixels) { return; }
  let channel = i / pixels;
  let pixel = i % pixels;
  let y = (pixel / p.previewSize) * p.factor + p.factor / 2u;
  let x = (pixel % p.previewSize) * p.factor + p.factor / 2u;
  let sourcePixels = p.sourceSize * p.sourceSize;
  output[i] = source[channel * 3u * sourcePixels + p.plane * sourcePixels + y * p.sourceSize + x];
}`;

function gatherPipeline(device) {
  let pipeline = pipelineByDevice.get(device);
  if (!pipeline) {
    pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: GATHER_SHADER }), entryPoint: 'main' },
    });
    pipelineByDevice.set(device, pipeline);
  }
  return pipeline;
}

export async function decodeReducedBackbonePreviewMesh({
  device, backboneBuf, postProcessorWeights, decoder, decoderWeights,
  factor = 4, resolution = 40,
}) {
  const plan = createReducedTriplanePlan(factor);
  const started = performance.now();
  const { value: output, allocations } = captureGpuBufferAllocations(() => {
    const encoder = device.createCommandEncoder({ label: `preview:postprocessor:factor-${factor}` });
    const outputPlaneBytes = 40 * plan.outputSize ** 2 * 4;
    const output = createEmptyBuffer(device, 3 * outputPlaneBytes, 0, 'preview:triplanes');
    const pipeline = gatherPipeline(device);
    for (let plane = 0; plane < 3; plane += 1) {
      const pixels = plan.inputSize ** 2;
      const gathered = createEmptyBuffer(device, 1024 * pixels * 4, 0, `preview:gather:${plane}`);
      const uniforms = createStorageBuffer(device, new Uint32Array([
        plan.sourceSize, plan.inputSize, factor, plane, 1024, Math.ceil(1024 * pixels / 256),
      ]), GPUBufferUsage.UNIFORM, `preview:gather-uniform:${plane}`);
      const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniforms } },
          { binding: 1, resource: { buffer: backboneBuf } },
          { binding: 2, resource: { buffer: gathered } },
        ],
      });
      const gather = encoder.beginComputePass();
      gather.setPipeline(pipeline);
      gather.setBindGroup(0, group);
      gather.dispatchWorkgroups(Math.ceil(1024 * pixels / 256));
      gather.end();
      let current = { buffer: gathered, outC: 1024, outH: plan.inputSize, outW: plan.inputSize };
      for (let layer = 0; layer < 4; layer += 1) {
        const outC = layer === 3 ? 640 : 1024;
        const result = dispatchConv2d(device, encoder, current.buffer,
          postProcessorWeights.convLayers[layer].weight,
          postProcessorWeights.convLayers[layer].bias,
          { inC: current.outC, inH: current.outH, inW: current.outW, outC,
            kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });
        current = layer === 3 ? result : {
          buffer: dispatchActivation(device, encoder, result.buffer, null,
            outC * result.outH * result.outW, 0),
          outC, outH: result.outH, outW: result.outW,
        };
      }
      const upsampled = dispatchPixelShuffle(device, encoder, current.buffer,
        { inC: 640, inH: plan.inputSize, inW: plan.inputSize, scaleFactor: 4 });
      encoder.copyBufferToBuffer(upsampled.buffer, 0, output, plane * outputPlaneBytes, outputPlaneBytes);
    }
    device.queue.submit([encoder.finish()]);
    return output;
  });
  try {
    await device.queue.onSubmittedWorkDone();
    const postprocessMs = performance.now() - started;
    const preview = await decodePreviewMesh(device, output, decoder, decoderWeights,
      resolution, plan.outputSize);
    return {
      mesh: preview.mesh,
      metrics: {
        factor,
        sourcePlaneSize: plan.sourceSize,
        previewPlaneSize: plan.outputSize,
        postprocessMs,
        query: preview.metrics,
        totalMs: performance.now() - started,
        postprocessTransientBytes: allocations.reduce((sum, item) => sum + item.size, 0),
      },
    };
  } finally {
    for (const { buffer } of allocations) buffer.destroy();
  }
}
