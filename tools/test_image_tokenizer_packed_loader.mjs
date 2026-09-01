import assert from 'node:assert/strict';

import { loadWeights } from '../src/lib/weights.js';

globalThis.GPUBufferUsage = globalThis.GPUBufferUsage || {
  STORAGE: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
};

function requiredTensorNames() {
  const entries = new Map();
  const add = (name, matrix = false) => entries.set(name, { matrix });
  const pair = (prefix, matrix = false) => {
    add(`${prefix}.weight`, matrix);
    add(`${prefix}.bias`);
  };

  for (const name of [
    'image_tokenizer.image_mean',
    'image_tokenizer.image_std',
    'image_tokenizer.model.embeddings.cls_token',
    'image_tokenizer.model.embeddings.position_embeddings',
    'image_tokenizer.model.encoder.layer.0.layer_scale1.lambda1',
  ]) add(name);
  pair('image_tokenizer.model.embeddings.patch_embeddings.projection');
  pair('image_tokenizer.model.layernorm');
  entries.delete('image_tokenizer.model.encoder.layer.0.layer_scale1.lambda1');

  for (let layer = 0; layer < 24; layer++) {
    const prefix = `image_tokenizer.model.encoder.layer.${layer}`;
    pair(`${prefix}.norm1`);
    pair(`${prefix}.attention.attention.query`, true);
    pair(`${prefix}.attention.attention.key`, true);
    pair(`${prefix}.attention.attention.value`, true);
    pair(`${prefix}.attention.output.dense`, true);
    add(`${prefix}.layer_scale1.lambda1`);
    pair(`${prefix}.norm2`);
    pair(`${prefix}.mlp.fc1`, true);
    pair(`${prefix}.mlp.fc2`, true);
    add(`${prefix}.layer_scale2.lambda1`);
    pair(`${prefix}.norm1_modulation.linear2`, true);
    pair(`${prefix}.norm2_modulation.linear2`, true);
  }

  pair('camera_embedder.linear');
  add('tokenizer.embeddings');
  add('backbone.latent_init');
  for (const name of ['norm_triplane', 'proj_triplane', 'norm_image', 'proj_image',
    'norm_latent', 'proj_latent', 'proj_out']) pair(`backbone.${name}`);

  const addAttention = prefix => {
    add(`${prefix}.wq.weight`);
    add(`${prefix}.wk.weight`);
    add(`${prefix}.wv.weight`);
    pair(`${prefix}.proj`);
  };
  const addFeedForward = prefix => {
    pair(`${prefix}.net.0.proj`);
    pair(`${prefix}.net.2`);
  };
  const addFuse = prefix => {
    addAttention(`${prefix}.attn`);
    pair(`${prefix}.norm_z1`);
    pair(`${prefix}.norm_z2`);
    addFeedForward(`${prefix}.ff`);
  };
  const addBasic = prefix => {
    pair(`${prefix}.norm1`);
    addAttention(`${prefix}.attn1`);
    pair(`${prefix}.norm2`);
    addAttention(`${prefix}.attn2`);
    pair(`${prefix}.norm3`);
    addFeedForward(`${prefix}.ff`);
  };
  for (let block = 0; block < 4; block++) {
    const prefix = `backbone.main_blocks.${block}`;
    addFuse(`${prefix}.fuse_block_in`);
    for (let layer = 0; layer < 3; layer++) {
      addBasic(`${prefix}.transformer_block.${layer}`);
    }
    addFuse(`${prefix}.fuse_block_out`);
  }

  for (const index of [0, 2, 4, 6]) pair(`post_processor.upsample.${index}`);
  return entries;
}

function makeWeightFile() {
  const entries = [...requiredTensorNames()].map(([name, descriptor]) => ({ name, ...descriptor }));
  const headerSize = 16 + entries.length * 160;
  const dataSize = entries.reduce((sum, entry) => sum + (entry.matrix ? 4 : 8), 0);
  const bytes = new Uint8Array(headerSize + dataSize);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x33445346, true);
  view.setUint32(4, 1, true);
  view.setUint32(8, entries.length, true);
  view.setUint32(12, headerSize, true);
  let relativeOffset = 0;
  for (let index = 0; index < entries.length; index++) {
    const entry = entries[index];
    const offset = 16 + index * 160;
    const encodedName = new TextEncoder().encode(entry.name);
    assert.ok(encodedName.length < 128, `fixture tensor name too long: ${entry.name}`);
    bytes.set(encodedName, offset);
    view.setUint32(offset + 128, entry.matrix ? 1 : 0, true);
    view.setUint32(offset + 132, entry.matrix ? 2 : 1, true);
    view.setUint32(offset + 136, entry.matrix ? 1 : 2, true);
    if (entry.matrix) view.setUint32(offset + 140, 2, true);
    view.setUint32(offset + 152, relativeOffset, true);
    const size = entry.matrix ? 4 : 8;
    view.setUint32(offset + 156, size, true);
    if (entry.matrix) {
      view.setUint16(headerSize + relativeOffset, 0x3c00, true);
      view.setUint16(headerSize + relativeOffset + 2, 0xc000, true);
    } else {
      view.setFloat32(headerSize + relativeOffset, 0.5, true);
      view.setFloat32(headerSize + relativeOffset + 4, -0.25, true);
    }
    relativeOffset += size;
  }
  return bytes;
}

function createDevice() {
  return {
    createBuffer(descriptor) {
      const storage = new ArrayBuffer(descriptor.size);
      return {
        descriptor,
        size: descriptor.size,
        getMappedRange: () => storage,
        unmap() {},
      };
    },
  };
}

const fixture = makeWeightFile();
const originalFetch = globalThis.fetch;
globalThis.fetch = async () => new Response(fixture.slice(), {
  status: 200,
  headers: { 'content-length': String(fixture.byteLength) },
});

try {
  const packed = await loadWeights(createDevice(), 'fixture.bin', null, {
    imageTokenizerMatrixRepresentation: 'f16-packed-u32',
  });
  const packedMatrices = packed.imageTokenizer.blocks.flatMap(block => [
    block.attn.q.weight,
    block.attn.k.weight,
    block.attn.v.weight,
    block.attn.proj.weight,
    block.mlp.fc1.weight,
    block.mlp.fc2.weight,
    block.norm1Mod.weight,
    block.norm2Mod.weight,
  ]);
  assert.equal(packedMatrices.length, 192);
  assert.ok(packedMatrices.every(weight => weight.representation === 'f16-packed-u32'));
  assert.ok(packedMatrices.every(weight => weight.storageByteLength === 4));
  assert.equal(packed.imageTokenizer.blocks[0].norm1.weight.representation, undefined);
  assert.deepEqual(packed.weightRepresentations.imageTokenizerMatrices, {
    representation: 'f16-packed-u32',
    tensorCount: 192,
    sourceByteLength: 768,
    storageByteLength: 768,
    expandedFp32ByteLength: 1536,
    savedVsExpandedFp32ByteLength: 768,
  });

  const expanded = await loadWeights(createDevice(), 'fixture.bin', null, {
    imageTokenizerMatrixRepresentation: 'f32-expanded',
  });
  const expandedMatrices = expanded.imageTokenizer.blocks.flatMap(block => [
    block.attn.q.weight,
    block.attn.k.weight,
    block.attn.v.weight,
    block.attn.proj.weight,
    block.mlp.fc1.weight,
    block.mlp.fc2.weight,
    block.norm1Mod.weight,
    block.norm2Mod.weight,
  ]);
  assert.equal(expandedMatrices.length, 192);
  assert.ok(expandedMatrices.every(weight => weight.representation === 'f32-expanded'));
  assert.ok(expandedMatrices.every(weight => weight.storageByteLength === 8));
  assert.equal(expanded.weightRepresentations.imageTokenizerMatrices.savedVsExpandedFp32ByteLength, 0);

  await assert.rejects(
    loadWeights(createDevice(), 'fixture.bin', null, {
      imageTokenizerMatrixRepresentation: 'fp8-native',
    }),
    /unsupported imageTokenizerMatrixRepresentation/,
  );
} finally {
  globalThis.fetch = originalFetch;
}

console.log('image tokenizer packed loader contracts passed');
