/** Tiny synthetic tensors for exercising the real loader, not model numerics. */
export function weightFixture({ normX = true, seed = 0, fullClipPrep = false } = {}) {
  const names = new Set();
  const add = (...values) => values.forEach(value => names.add(value));
  const pair = prefix => add(`${prefix}.weight`, `${prefix}.bias`);
  add('image_tokenizer.image_mean', 'image_tokenizer.image_std',
    'image_tokenizer.model.embeddings.cls_token', 'image_tokenizer.model.embeddings.position_embeddings');
  pair('image_tokenizer.model.embeddings.patch_embeddings.projection');
  pair('image_tokenizer.model.layernorm');
  for (let i = 0; i < 24; i++) {
    const p = `image_tokenizer.model.encoder.layer.${i}`;
    for (const suffix of ['norm1', 'norm2', 'attention.attention.query', 'attention.attention.key',
      'attention.attention.value', 'attention.output.dense', 'mlp.fc1', 'mlp.fc2',
      'norm1_modulation.linear2', 'norm2_modulation.linear2']) pair(`${p}.${suffix}`);
    add(`${p}.layer_scale1.lambda1`, `${p}.layer_scale2.lambda1`);
  }
  pair('camera_embedder.linear');
  add('tokenizer.embeddings', 'backbone.latent_init');
  for (const suffix of ['norm_triplane', 'proj_triplane', 'norm_image', 'proj_image',
    'norm_latent', 'proj_latent', 'proj_out']) pair(`backbone.${suffix}`);
  for (let b = 0; b < 4; b++) {
    const p = `backbone.main_blocks.${b}`;
    for (const fuse of ['fuse_block_in', 'fuse_block_out']) {
      for (const suffix of ['attn.wq.weight', 'attn.wk.weight', 'attn.wv.weight']) add(`${p}.${fuse}.${suffix}`);
      for (const suffix of ['attn.proj', 'norm_z1', 'norm_z2', 'ff.net.0.proj', 'ff.net.2']) pair(`${p}.${fuse}.${suffix}`);
      if (normX) pair(`${p}.${fuse}.norm_x`);
    }
    for (let i = 0; i < 3; i++) {
      for (const suffix of ['norm1', 'norm2', 'norm3', 'attn1.proj', 'attn2.proj',
        'ff.net.0.proj', 'ff.net.2']) pair(`${p}.transformer_block.${i}.${suffix}`);
      for (const suffix of ['attn1.wq.weight', 'attn1.wk.weight', 'attn1.wv.weight',
        'attn2.wq.weight', 'attn2.wk.weight', 'attn2.wv.weight']) add(`${p}.transformer_block.${i}.${suffix}`);
    }
  }
  for (const i of [0, 2, 4, 6]) pair(`post_processor.upsample.${i}`);
  for (const head of ['density', 'features', 'perturb_normal', 'vertex_offset']) pair(`decoder.heads.${head}.0`);
  pair('image_estimator.model.visual.ln_pre');
  pair('image_estimator.model.visual.ln_post');
  for (let i = 0; i < 12; i++) {
    const p = `image_estimator.model.visual.transformer.resblocks.${i}`;
    for (const suffix of ['ln_1', 'ln_2', 'attn.out_proj', 'mlp.c_fc', 'mlp.c_proj']) pair(`${p}.${suffix}`);
    add(`${p}.attn.in_proj_weight`, `${p}.attn.in_proj_bias`);
  }
  add('image_estimator.model.visual.conv1.weight', 'image_estimator.model.visual.class_embedding',
    'image_estimator.model.visual.positional_embedding');
  const clipPrepShapes = fullClipPrep ? new Map([
    ['image_estimator.model.visual.conv1.weight', [768, 3072]],
    ['image_estimator.model.visual.class_embedding', [768]],
    ['image_estimator.model.visual.positional_embedding', [50, 768]],
  ]) : new Map();
  const tensorByteSize = name => (clipPrepShapes.get(name) || [1]).reduce((size, dim) => size * dim, 4);
  const headerSize = 16 + names.size * 160;
  const bytes = new Uint8Array(headerSize + [...names].reduce((size, name) => size + tensorByteSize(name), 0));
  const view = new DataView(bytes.buffer);
  [0x33445346, 1, names.size, headerSize].forEach((value, i) => view.setUint32(i * 4, value, true));
  const values = new Map();
  let payloadOffset = 0;
  [...names].forEach((name, i) => {
    const entry = 16 + i * 160;
    bytes.set(new TextEncoder().encode(name), entry);
    const shape = clipPrepShapes.get(name) || [1];
    const byteSize = tensorByteSize(name);
    view.setUint32(entry + 132, shape.length, true);
    shape.forEach((dimension, axis) => view.setUint32(entry + 136 + axis * 4, dimension, true));
    view.setUint32(entry + 152, payloadOffset, true);
    view.setUint32(entry + 156, byteSize, true);
    view.setFloat32(headerSize + payloadOffset, seed + i + 1, true);
    values.set(name, seed + i + 1);
    payloadOffset += byteSize;
  });
  return { bytes, values };
}

export function fakeWeightDevice() {
  const buffers = [];
  const clipPipelines = [];
  const device = {
    buffers, clipPipelines, failBufferAt: null, failClipPipeline: false,
    limits: { maxBufferSize: 2147483648, maxStorageBufferBindingSize: 1073741824, maxComputeInvocationsPerWorkgroup: 256 },
    features: new Set(),
    queue: { submit() {}, writeBuffer() {}, async onSubmittedWorkDone() {} },
    createBuffer({ size, label }) {
      if (buffers.length === device.failBufferAt) throw new Error('injected allocation failure');
      const data = new ArrayBuffer(size);
      const buffer = { device, label, data, destroyed: 0,
        getMappedRange() { return data; }, unmap() {}, destroy() { this.destroyed++; } };
      buffers.push(buffer);
      return buffer;
    },
    createShaderModule(desc) { return { ...desc, device, async getCompilationInfo() { return { messages: [] }; } }; },
    createComputePipeline(desc) {
      const pipeline = { ...desc, device, getBindGroupLayout() { return {}; } };
      if (desc.compute.module.label?.startsWith('SF3D CLIP')) {
        clipPipelines.push(pipeline);
        if (device.failClipPipeline) throw new Error('injected pipeline failure');
      }
      return pipeline;
    },
    createBindGroupLayout(desc) { return desc; },
    createPipelineLayout(desc) { return desc; },
    pushErrorScope() {}, async popErrorScope() { return null; },
  };
  return device;
}

export function installWeightFetch(fixture) {
  const original = globalThis.fetch;
  globalThis.fetch = async url => String(url).endsWith('.wgsl')
    ? new Response('@compute @workgroup_size(1) fn main() {}', { headers: { 'content-type': 'text/wgsl' } })
    : new Response(fixture.bytes);
  return () => { globalThis.fetch = original; };
}
