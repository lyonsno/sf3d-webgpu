import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { readSf3dWeightInventory } from './lib/sf3d_weight_inventory.mjs';

const MAGIC = 0x33445346;
const ENTRY_SIZE = 160;
const ALIGNMENT = 16;

function align(value) {
  return value + ((ALIGNMENT - (value % ALIGNMENT)) % ALIGNMENT);
}

function buildWeightFile(tensors) {
  const headerSize = 16 + tensors.length * ENTRY_SIZE;
  let payloadSize = 0;
  const entries = tensors.map(tensor => {
    const entry = { ...tensor, offset: payloadSize };
    payloadSize = align(payloadSize + tensor.data.byteLength);
    return entry;
  });
  const bytes = Buffer.alloc(headerSize + payloadSize);
  bytes.writeUInt32LE(MAGIC, 0);
  bytes.writeUInt32LE(1, 4);
  bytes.writeUInt32LE(entries.length, 8);
  bytes.writeUInt32LE(headerSize, 12);
  for (let index = 0; index < entries.length; index++) {
    const tensor = entries[index];
    const entryOffset = 16 + index * ENTRY_SIZE;
    bytes.write(tensor.name, entryOffset, 128, 'ascii');
    bytes.writeUInt32LE(tensor.dtype, entryOffset + 128);
    bytes.writeUInt32LE(tensor.shape.length, entryOffset + 132);
    tensor.shape.forEach((dimension, dimensionIndex) => {
      bytes.writeUInt32LE(dimension, entryOffset + 136 + dimensionIndex * 4);
    });
    bytes.writeUInt32LE(tensor.offset, entryOffset + 152);
    bytes.writeUInt32LE(tensor.data.byteLength, entryOffset + 156);
    tensor.data.copy(bytes, headerSize + tensor.offset);
  }
  return bytes;
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sf3d-weight-inventory-'));
const weightPath = path.join(root, 'weights.bin');
const weightBytes = buildWeightFile([
  { name: 'decoder.linear.weight', dtype: 1, shape: [3, 5], data: Buffer.alloc(30, 0x11) },
  { name: 'post_processor.conv.weight', dtype: 1, shape: [2, 3, 3, 3], data: Buffer.alloc(108, 0x22) },
  { name: 'tokenizer.embeddings', dtype: 1, shape: [1, 7, 4], data: Buffer.alloc(56, 0x33) },
  { name: 'image_tokenizer.image_mean', dtype: 1, shape: [1, 1, 3, 1, 1], data: Buffer.alloc(6, 0x35) },
  { name: 'image_estimator.proj.weight', dtype: 0, shape: [2, 2], data: Buffer.alloc(16, 0x44) },
]);
fs.writeFileSync(weightPath, weightBytes);

const source = {
  kind: 'hugging-face',
  repoId: 'BasinShapers/sf3d-webgpu-weights',
  revision: 'a'.repeat(40),
  file: 'weights.bin',
};
const inventory = await readSf3dWeightInventory(weightPath, { source });
assert.equal(inventory.schema, 'sf3d.weight-inventory.v1');
assert.deepEqual(inventory.source, source);
assert.equal(inventory.artifact.byteLength, weightBytes.byteLength);
assert.equal(
  inventory.artifact.sha256,
  crypto.createHash('sha256').update(weightBytes).digest('hex'),
);
assert.equal(inventory.format.magic, 'SF3D');
assert.equal(inventory.format.version, 1);
assert.equal(inventory.format.tensorCount, 5);
assert.equal(inventory.tensors.length, 5);
assert.deepEqual(inventory.summary.bySourceDtype, {
  fp16: { tensorCount: 4, sourceByteLength: 200 },
  fp32: { tensorCount: 1, sourceByteLength: 16 },
});
assert.deepEqual(inventory.summary.fp16Classes, {
  matrix: { tensorCount: 1, sourceByteLength: 30 },
  postprocessorConvolution: { tensorCount: 1, sourceByteLength: 108 },
  tokenizerEmbedding: { tensorCount: 1, sourceByteLength: 56 },
  remaining: { tensorCount: 1, sourceByteLength: 6 },
});
assert.equal(inventory.summary.expandedFp32GpuByteLength, 416);
assert.equal(inventory.summary.fp16StorageSavingsByteLength, 200);
const incompleteShape = inventory.tensors.find(tensor => tensor.name === 'image_tokenizer.image_mean');
assert.equal(incompleteShape.declaredRank, 5);
assert.deepEqual(incompleteShape.shape, [1, 1, 3, 1]);
assert.equal(incompleteShape.shapeComplete, false);
assert.equal(incompleteShape.elementCount, 3);

async function rejectsMutation(label, mutate, pattern) {
  const mutatedPath = path.join(root, `${label}.bin`);
  const mutated = Buffer.from(weightBytes);
  mutate(mutated);
  fs.writeFileSync(mutatedPath, mutated);
  await assert.rejects(readSf3dWeightInventory(mutatedPath, { source }), pattern);
}

await rejectsMutation('bad-magic', bytes => bytes.writeUInt32LE(0, 0), /magic/i);
await rejectsMutation('bad-version', bytes => bytes.writeUInt32LE(7, 4), /version/i);
await rejectsMutation('truncated-range', bytes => {
  bytes.writeUInt32LE(bytes.byteLength, 16 + 156);
}, /outside|range|size/i);
await rejectsMutation('overlap', bytes => {
  bytes.writeUInt32LE(0, 16 + ENTRY_SIZE + 152);
}, /overlap/i);
await rejectsMutation('duplicate-name', bytes => {
  bytes.fill(0, 16 + ENTRY_SIZE, 16 + ENTRY_SIZE + 128);
  bytes.write('decoder.linear.weight', 16 + ENTRY_SIZE, 128, 'ascii');
}, /duplicate/i);

await assert.rejects(
  readSf3dWeightInventory(weightPath, { source: { ...source, revision: 'main' } }),
  /revision/i,
);

console.log('SF3D weight inventory contracts passed');
