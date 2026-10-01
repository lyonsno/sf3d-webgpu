import spatialWGSL from '../shaders/post_processor_spatial.wgsl?raw';
import { captureGpuBufferAllocations, createEmptyBuffer, createStorageBuffer } from './gpu.js';
import { dispatchConv2dChannelRange } from './shader_ops.js';

const pipelines = new WeakMap();
const SF3D = Object.freeze({ size: 96, channels: 1024, outChannels: 40, scale: 4 });

// Four radius-one convolutions need three extra rows at the first output layer.
export function spatialLayerEnds(end, size) {
  return [3, 2, 1, 0].map(halo => Math.min(size, end + halo));
}

function dispatchLayout(device, encoder, input, output, config, plane, start, end, kind) {
  let cache = pipelines.get(device);
  if (!cache) { cache = new Map(); pipelines.set(device, cache); }
  if (!cache.has(kind)) cache.set(kind, device.createComputePipeline({ layout: 'auto',
    compute: { module: device.createShaderModule({ code: spatialWGSL }), entryPoint: kind } }));
  const pipeline = cache.get(kind);
  const { size, scale } = config;
  const channels = kind === 'gather' ? config.channels : config.outChannels;
  const count = kind === 'gather' ? channels * size * size : channels * (end - start) * scale * size * scale;
  const groups = Math.ceil(count / 256), x = Math.min(65535, groups);
  const uniform = createStorageBuffer(device, new Uint32Array([channels, size, scale, plane, start, end - start, x]), GPUBufferUsage.UNIFORM);
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniform } },
    { binding: 1, resource: { buffer: input } },
    { binding: 2, resource: { buffer: output } },
  ] });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(x, Math.ceil(groups / x)); pass.end();
}

/** Borrow complete XY and growing XZ/YZ rows inside each awaited observation. */
export async function streamPostProcessor(device, input, weights, {
  onRegion, rowsPerRegion = 16, config = SF3D,
}) {
  const { size, channels, outChannels, scale } = config;
  if (!Number.isSafeInteger(rowsPerRegion) || rowsPerRegion <= 0) throw new RangeError('rowsPerRegion must be positive');
  if (typeof onRegion !== 'function') throw new TypeError('onRegion must be a function');
  if (weights?.convLayers?.length !== 4) throw new RangeError('four convolution layers required');
  for (const value of [size, channels, outChannels, scale]) {
    if (!Number.isSafeInteger(value) || value <= 0) throw new RangeError('invalid spatial postprocessor dimensions');
  }
  const owned = [];
  const allocate = fn => {
    const captured = captureGpuBufferAllocations(fn);
    owned.push(...captured.allocations);
    return captured.value;
  };
  const output = allocate(() => createEmptyBuffer(device, 3 * outChannels * (size * scale) ** 2 * 4));
  const makePlane = plane => allocate(() => ({ plane, ends: [0, 0, 0, 0],
    buffers: [channels, channels, channels, channels, outChannels * scale ** 2]
      .map(c => createEmptyBuffer(device, c * size * size * 4)),
  }));
  const submit = async encode => {
    const encoder = device.createCommandEncoder({ label: 'preview:spatial-postprocessor' });
    allocate(() => encode(encoder));
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  };
  const gather = (encoder, state) => dispatchLayout(device, encoder, input, state.buffers[0], config, state.plane, 0, size, 'gather');
  const advance = (encoder, state, end) => {
    const ends = spatialLayerEnds(end, size);
    for (let layer = 0; layer < 4; layer++) {
      const start = state.ends[layer], count = ends[layer] - start;
      if (!count) continue;
      const outC = layer === 3 ? outChannels * scale ** 2 : channels;
      dispatchConv2dChannelRange(device, encoder, state.buffers[layer], weights.convLayers[layer].weight,
        weights.convLayers[layer].bias, state.buffers[layer + 1], {
          inC: channels, inH: size, inW: size, outC, kH: 3, kW: 3,
          padH: 1, padW: 1, strideH: 1, strideW: 1, applyRelu: layer < 3,
        }, { channelStart: 0, channelCount: outC, rowStart: start, rowCount: count });
    }
    const start = state.ends[3];
    dispatchLayout(device, encoder, state.buffers[4], output, config, state.plane, start, end, 'shuffle');
    state.ends = ends;
  };
  try {
    const xy = makePlane(0);
    await submit(encoder => { gather(encoder, xy); advance(encoder, xy, size); });
    for (const buffer of xy.buffers) buffer.destroy();
    const depthPlanes = [makePlane(1), makePlane(2)];
    await submit(encoder => { for (const plane of depthPlanes) gather(encoder, plane); });
    for (let end = Math.min(rowsPerRegion, size); ; end = Math.min(end + rowsPerRegion, size)) {
      await submit(encoder => { for (const plane of depthPlanes) advance(encoder, plane, end); });
      await onRegion({ buffer: output, completedRows: end * scale, totalRows: size * scale });
      if (end === size) break;
    }
  } finally {
    // Consumer queries may have submitted work before throwing.
    try { await device.queue.onSubmittedWorkDone(); }
    finally { for (const { buffer } of owned) buffer.destroy(); }
  }
}
