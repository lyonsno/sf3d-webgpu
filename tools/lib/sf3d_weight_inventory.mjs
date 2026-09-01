import crypto from 'node:crypto';
import fs from 'node:fs';
import { open, stat } from 'node:fs/promises';

const MAGIC = 0x33445346;
const VERSION = 1;
const ENTRY_SIZE = 160;
const NAME_SIZE = 128;
const GIT_SHA = /^[0-9a-f]{40}$/;

function validateSource(source) {
  if (!source || typeof source !== 'object' || Array.isArray(source)) {
    throw new TypeError('source identity is required');
  }
  if (source.kind !== 'hugging-face') throw new Error('source kind must be hugging-face');
  if (typeof source.repoId !== 'string' || !source.repoId.includes('/')) {
    throw new Error('source repoId must identify a Hugging Face repository');
  }
  if (!GIT_SHA.test(source.revision || '')) {
    throw new Error('source revision must be an exact 40-character Git SHA');
  }
  if (typeof source.file !== 'string' || source.file.length === 0) {
    throw new Error('source file identity is required');
  }
  return {
    kind: source.kind,
    repoId: source.repoId,
    revision: source.revision,
    file: source.file,
  };
}

function checkedProduct(values, label) {
  let product = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new Error(`${label} dimensions must be positive safe integers`);
    }
    product *= value;
    if (!Number.isSafeInteger(product)) throw new Error(`${label} element count is not a safe integer`);
  }
  return product;
}

function emptyCount() {
  return { tensorCount: 0, sourceByteLength: 0 };
}

function addCount(target, size) {
  target.tensorCount += 1;
  target.sourceByteLength += size;
}

async function sha256File(filePath) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(filePath)) hash.update(chunk);
  return hash.digest('hex');
}

export async function readSf3dWeightInventory(filePath, { source } = {}) {
  const admittedSource = validateSource(source);
  const fileStat = await stat(filePath);
  if (!fileStat.isFile()) throw new Error('weight artifact must be a regular file');
  if (!Number.isSafeInteger(fileStat.size) || fileStat.size < 16) {
    throw new Error('weight artifact is too small for an SF3D header');
  }

  const handle = await open(filePath, 'r');
  let header;
  try {
    const prefix = Buffer.alloc(16);
    const prefixRead = await handle.read(prefix, 0, prefix.length, 0);
    if (prefixRead.bytesRead !== prefix.length) throw new Error('weight artifact header is truncated');
    const magic = prefix.readUInt32LE(0);
    if (magic !== MAGIC) throw new Error(`invalid SF3D weight magic 0x${magic.toString(16)}`);
    const version = prefix.readUInt32LE(4);
    if (version !== VERSION) throw new Error(`unsupported SF3D weight version ${version}`);
    const tensorCount = prefix.readUInt32LE(8);
    const headerSize = prefix.readUInt32LE(12);
    const expectedHeaderSize = 16 + tensorCount * ENTRY_SIZE;
    if (!Number.isSafeInteger(expectedHeaderSize) || headerSize !== expectedHeaderSize) {
      throw new Error(`header size ${headerSize} does not match ${tensorCount} tensor entries`);
    }
    if (headerSize > fileStat.size) throw new Error('tensor table extends outside the weight artifact');
    header = Buffer.alloc(headerSize);
    const headerRead = await handle.read(header, 0, header.length, 0);
    if (headerRead.bytesRead !== header.length) throw new Error('tensor table is truncated');
  } finally {
    await handle.close();
  }

  const tensorCount = header.readUInt32LE(8);
  const headerSize = header.readUInt32LE(12);
  const names = new Set();
  const tensors = [];
  for (let index = 0; index < tensorCount; index++) {
    const entryOffset = 16 + index * ENTRY_SIZE;
    const nameField = header.subarray(entryOffset, entryOffset + NAME_SIZE);
    const terminator = nameField.indexOf(0);
    const nameBytes = terminator >= 0 ? nameField.subarray(0, terminator) : nameField;
    const name = nameBytes.toString('ascii');
    if (!name || /[^\x20-\x7e]/.test(name)) throw new Error(`tensor ${index} has an invalid name`);
    if (names.has(name)) throw new Error(`duplicate tensor name: ${name}`);
    names.add(name);

    const dtypeCode = header.readUInt32LE(entryOffset + 128);
    if (dtypeCode !== 0 && dtypeCode !== 1) throw new Error(`${name} has unsupported dtype ${dtypeCode}`);
    const dtype = dtypeCode === 1 ? 'fp16' : 'fp32';
    const ndim = header.readUInt32LE(entryOffset + 132);
    if (ndim > 32) throw new Error(`${name} rank ${ndim} is not credible`);
    const shapeComplete = ndim <= 4;
    const shape = [];
    for (let dimension = 0; dimension < Math.min(ndim, 4); dimension++) {
      shape.push(header.readUInt32LE(entryOffset + 136 + dimension * 4));
    }
    const relativeOffset = header.readUInt32LE(entryOffset + 152);
    const sourceByteLength = header.readUInt32LE(entryOffset + 156);
    const elementByteLength = dtype === 'fp16' ? 2 : 4;
    if (sourceByteLength % elementByteLength !== 0) {
      throw new Error(`${name} size ${sourceByteLength} is not divisible by its dtype byte width`);
    }
    const byteDerivedElementCount = sourceByteLength / elementByteLength;
    const elementCount = shapeComplete ? checkedProduct(shape, name) : byteDerivedElementCount;
    if (shapeComplete && sourceByteLength !== elementCount * elementByteLength) {
      throw new Error(`${name} size ${sourceByteLength} does not match shape-derived ${elementCount * elementByteLength}`);
    }
    const absoluteOffset = headerSize + relativeOffset;
    const absoluteEnd = absoluteOffset + sourceByteLength;
    if (!Number.isSafeInteger(absoluteEnd) || absoluteOffset < headerSize || absoluteEnd > fileStat.size) {
      throw new Error(`${name} data range is outside the weight artifact`);
    }
    tensors.push({
      name,
      dtype,
      declaredRank: ndim,
      shape,
      shapeComplete,
      elementCount,
      relativeOffset,
      absoluteOffset,
      sourceByteLength,
    });
  }

  const ranges = [...tensors].sort((left, right) => left.absoluteOffset - right.absoluteOffset);
  for (let index = 1; index < ranges.length; index++) {
    const previous = ranges[index - 1];
    const current = ranges[index];
    if (current.absoluteOffset < previous.absoluteOffset + previous.sourceByteLength) {
      throw new Error(`tensor data ranges overlap: ${previous.name} and ${current.name}`);
    }
  }

  const bySourceDtype = { fp16: emptyCount(), fp32: emptyCount() };
  const fp16Classes = {
    matrix: emptyCount(),
    postprocessorConvolution: emptyCount(),
    tokenizerEmbedding: emptyCount(),
    remaining: emptyCount(),
  };
  for (const tensor of tensors) {
    addCount(bySourceDtype[tensor.dtype], tensor.sourceByteLength);
    if (tensor.dtype !== 'fp16') continue;
    if (tensor.shapeComplete && tensor.shape.length === 2) {
      addCount(fp16Classes.matrix, tensor.sourceByteLength);
    } else if (tensor.shapeComplete && tensor.shape.length === 4
        && tensor.name.startsWith('post_processor.')) {
      addCount(fp16Classes.postprocessorConvolution, tensor.sourceByteLength);
    } else if (tensor.name === 'tokenizer.embeddings') {
      addCount(fp16Classes.tokenizerEmbedding, tensor.sourceByteLength);
    } else {
      addCount(fp16Classes.remaining, tensor.sourceByteLength);
    }
  }
  const expandedFp32GpuByteLength = bySourceDtype.fp32.sourceByteLength
    + bySourceDtype.fp16.sourceByteLength * 2;

  return {
    schema: 'sf3d.weight-inventory.v1',
    source: admittedSource,
    artifact: {
      byteLength: fileStat.size,
      sha256: await sha256File(filePath),
    },
    format: {
      magic: 'SF3D',
      version: VERSION,
      headerByteLength: headerSize,
      tensorCount,
      entryByteLength: ENTRY_SIZE,
    },
    summary: {
      bySourceDtype,
      fp16Classes,
      expandedFp32GpuByteLength,
      fp16StorageSavingsByteLength: bySourceDtype.fp16.sourceByteLength,
    },
    tensors,
  };
}
