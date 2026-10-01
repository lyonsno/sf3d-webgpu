import shader from '../shaders/token_similarity.wgsl?raw';

const pipelines = new WeakMap();

// One scalar per image patch. Token zero is CLS, not an image patch.
export async function readTokenSimilarity(device, tokensBuf, { dim, width, height, patchIndex }) {
  const count = width * height;
  if (![dim, width, height].every(n => Number.isSafeInteger(n) && n > 0)
      || !Number.isSafeInteger(patchIndex) || patchIndex < 0 || patchIndex >= count
      || tokensBuf.size < (count + 1) * dim * 4) {
    throw new RangeError('invalid token-similarity shape or reference patch');
  }
  if (!pipelines.has(device)) {
    pipelines.set(device, device.createComputePipelineAsync({
      label: 'learn-token-similarity', layout: 'auto',
      compute: { module: device.createShaderModule({ code: shader }), entryPoint: 'main' },
    }));
  }
  const pipeline = await pipelines.get(device);
  const owned = [];
  const buffer = descriptor => { const b = device.createBuffer(descriptor); owned.push(b); return b; };
  try {
    const params = buffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    new Uint32Array(params.getMappedRange()).set([dim, count, patchIndex, 0]);
    params.unmap();
    const output = buffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const staging = buffer({ size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'learn-token-similarity' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: tokensBuf } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: params } },
    ] }));
    pass.dispatchWorkgroups(count);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, staging, 0, count * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return new Float32Array(staging.getMappedRange().slice(0));
  } finally {
    for (const b of owned) b.destroy();
  }
}
