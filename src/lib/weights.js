/**
 * weights.js — Load SF3D weights from flat binary format.
 *
 * Binary format (from convert_weights.py):
 *   Header: 4 (magic) + 4 (version) + 4 (num_tensors) + 4 (header_size) = 16 bytes
 *   Tensor table: num_tensors × 160 bytes each
 *     128 bytes: name (null-padded ASCII)
 *     4 bytes: dtype (0=fp32, 1=fp16)
 *     4 bytes: ndim
 *     16 bytes: shape (4 x u32)
 *     4 bytes: offset
 *     4 bytes: size
 *   Weight data: packed tensors
 */

import { createStorageBuffer } from './gpu.js';

const MAGIC = 0x33445346; // "SF3D" in little-endian
const ENTRY_SIZE = 160;

function parseHeader(buffer) {
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== MAGIC) {
    throw new Error(`Invalid weight file magic: 0x${magic.toString(16)} (expected 0x${MAGIC.toString(16)})`);
  }
  const version = view.getUint32(4, true);
  if (version !== 1) throw new Error(`Unsupported weight file version: ${version}`);

  const numTensors = view.getUint32(8, true);
  const headerSize = view.getUint32(12, true);

  const tensors = new Map();
  for (let i = 0; i < numTensors; i++) {
    const off = 16 + i * ENTRY_SIZE;
    const nameBytes = new Uint8Array(buffer, off, 128);
    let nameEnd = nameBytes.indexOf(0);
    if (nameEnd === -1) nameEnd = 128;
    const name = new TextDecoder().decode(nameBytes.slice(0, nameEnd));

    const dtype = view.getUint32(off + 128, true);
    const ndim = view.getUint32(off + 132, true);
    const shape = [];
    for (let d = 0; d < ndim; d++) {
      shape.push(view.getUint32(off + 136 + d * 4, true));
    }
    const dataOffset = view.getUint32(off + 152, true);
    const size = view.getUint32(off + 156, true);
    tensors.set(name, { dtype, shape, offset: dataOffset + headerSize, size });
  }

  return { tensors, headerSize };
}

function fp16ToFp32(h) {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) {
    if (mant === 0) return sign ? -0.0 : 0.0;
    let val = mant / 1024.0 * Math.pow(2, -14);
    return sign ? -val : val;
  }
  if (exp === 31) return mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
  const val = Math.pow(2, exp - 15) * (1 + mant / 1024.0);
  return sign ? -val : val;
}

function extractTensor(device, buffer, info) {
  const { dtype, offset, size } = info;
  const raw = extractBytes(buffer, offset, size);
  if (dtype === 0) {
    // fp32 — raw bytes are already float32
    const fp32 = new Float32Array(raw.buffer, raw.byteOffset, size / 4);
    return createStorageBuffer(device, fp32);
  } else {
    const fp16 = new Uint16Array(raw.buffer, raw.byteOffset, size / 2);
    const fp32 = new Float32Array(fp16.length);
    for (let i = 0; i < fp16.length; i++) fp32[i] = fp16ToFp32(fp16[i]);
    return createStorageBuffer(device, fp32);
  }
}

function extractTensorCPU(buffer, info) {
  const { dtype, offset, size } = info;
  const raw = extractBytes(buffer, offset, size);
  if (dtype === 0) {
    const fp32 = new Float32Array(raw.buffer, raw.byteOffset, size / 4);
    return new Float32Array(fp32); // copy to decouple from chunk
  }
  const fp16 = new Uint16Array(raw.buffer, raw.byteOffset, size / 2);
  const fp32 = new Float32Array(fp16.length);
  for (let i = 0; i < fp16.length; i++) fp32[i] = fp16ToFp32(fp16[i]);
  return fp32;
}

/**
 * Extract a byte range from the chunked buffer.
 * Returns a Uint8Array view if the range falls within a single chunk,
 * otherwise copies into a new buffer (only for tensors that span chunk boundaries).
 */
function extractBytes(chunkedBuffer, offset, size) {
  if (chunkedBuffer instanceof ArrayBuffer) {
    // Legacy single-buffer path
    return new Uint8Array(chunkedBuffer, offset, size);
  }
  // chunkedBuffer is { chunks, offsets, totalSize }
  const { chunks, offsets } = chunkedBuffer;
  const end = offset + size;

  // Find the first chunk that contains the start
  let startChunk = 0;
  for (let i = 0; i < offsets.length; i++) {
    if (offsets[i] + chunks[i].length > offset) { startChunk = i; break; }
  }

  const localOffset = offset - offsets[startChunk];
  // Check if entirely within this chunk
  if (localOffset + size <= chunks[startChunk].length) {
    // Ensure 4-byte alignment for typed array views (Float32Array, Uint16Array)
    const chunkBaseOffset = chunks[startChunk].byteOffset + localOffset;
    if (chunkBaseOffset % 4 !== 0) {
      const copy = new Uint8Array(size);
      copy.set(chunks[startChunk].subarray(localOffset, localOffset + size));
      return copy;
    }
    return chunks[startChunk].subarray(localOffset, localOffset + size);
  }

  // Spans multiple chunks — copy into aligned buffer
  const result = new Uint8Array(size);
  let written = 0;
  for (let i = startChunk; i < chunks.length && written < size; i++) {
    const chunkStart = Math.max(0, offset + written - offsets[i]);
    const avail = chunks[i].length - chunkStart;
    const take = Math.min(avail, size - written);
    result.set(chunks[i].subarray(chunkStart, chunkStart + take), written);
    written += take;
  }
  return result;
}

/**
 * Load SF3D weights and organize into component structure.
 */
export async function loadWeights(device, url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch weights: ${response.status}`);

  const contentLength = parseInt(response.headers.get('content-length') || '0');
  const reader = response.body.getReader();
  const chunks = [];
  const chunkOffsets = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunkOffsets.push(received);
    chunks.push(value);
    received += value.length;
    if (onProgress) onProgress(received, contentLength);
  }

  // Build a chunked buffer that avoids a single >2GB ArrayBuffer
  const chunkedBuffer = { chunks, offsets: chunkOffsets, totalSize: received };

  // Parse header from the first chunk(s) — header is always small (<1MB)
  // Always copy to get a clean ArrayBuffer for DataView
  const headerBytes = extractBytes(chunkedBuffer, 0, Math.min(received, 1024 * 1024));
  const headerBuf = headerBytes.slice().buffer;
  const { tensors } = parseHeader(headerBuf);

  const get = (name) => {
    const info = tensors.get(name);
    if (!info) throw new Error(`Missing weight: ${name}`);
    return extractTensor(device, chunkedBuffer, info);
  };

  const tryGet = (name) => {
    const info = tensors.get(name);
    if (!info) return null;
    return extractTensor(device, chunkedBuffer, info);
  };

  const getCPU = (name) => {
    const info = tensors.get(name);
    if (!info) throw new Error(`Missing weight: ${name}`);
    return extractTensorCPU(chunkedBuffer, info);
  };

  const getInfo = (name) => {
    const info = tensors.get(name);
    if (!info) throw new Error(`Missing weight info: ${name}`);
    return info;
  };

  // === Image Tokenizer (DINOv2 ViT-Large with modulation) ===
  const imageTokenizer = {
    imageMean: get('image_tokenizer.image_mean'),
    imageStd: get('image_tokenizer.image_std'),
    patchEmbed: {
      weight: get('image_tokenizer.model.embeddings.patch_embeddings.projection.weight'),
      bias: get('image_tokenizer.model.embeddings.patch_embeddings.projection.bias'),
    },
    clsToken: get('image_tokenizer.model.embeddings.cls_token'),
    posEmbed: get('image_tokenizer.model.embeddings.position_embeddings'),
    layernorm: {
      weight: get('image_tokenizer.model.layernorm.weight'),
      bias: get('image_tokenizer.model.layernorm.bias'),
    },
    blocks: [],
  };

  // 24 DINOv2 transformer blocks with AdaNorm modulation
  for (let l = 0; l < 24; l++) {
    const p = `image_tokenizer.model.encoder.layer.${l}`;
    imageTokenizer.blocks.push({
      norm1: { weight: get(`${p}.norm1.weight`), bias: get(`${p}.norm1.bias`) },
      attn: {
        q: { weight: get(`${p}.attention.attention.query.weight`), bias: get(`${p}.attention.attention.query.bias`) },
        k: { weight: get(`${p}.attention.attention.key.weight`), bias: get(`${p}.attention.attention.key.bias`) },
        v: { weight: get(`${p}.attention.attention.value.weight`), bias: get(`${p}.attention.attention.value.bias`) },
        proj: { weight: get(`${p}.attention.output.dense.weight`), bias: get(`${p}.attention.output.dense.bias`) },
      },
      layerScale1: get(`${p}.layer_scale1.lambda1`),
      norm2: { weight: get(`${p}.norm2.weight`), bias: get(`${p}.norm2.bias`) },
      mlp: {
        fc1: { weight: get(`${p}.mlp.fc1.weight`), bias: get(`${p}.mlp.fc1.bias`) },
        fc2: { weight: get(`${p}.mlp.fc2.weight`), bias: get(`${p}.mlp.fc2.bias`) },
      },
      layerScale2: get(`${p}.layer_scale2.lambda1`),
      // AdaNorm modulation from camera embeddings
      norm1Mod: { weight: get(`${p}.norm1_modulation.linear2.weight`), bias: get(`${p}.norm1_modulation.linear2.bias`) },
      norm2Mod: { weight: get(`${p}.norm2_modulation.linear2.weight`), bias: get(`${p}.norm2_modulation.linear2.bias`) },
    });
  }

  // === Camera Embedder ===
  const cameraEmbedder = {
    weight: get('camera_embedder.linear.weight'),
    bias: get('camera_embedder.linear.bias'),
  };

  // === Triplane Tokenizer ===
  const tokenizer = {
    embeddings: get('tokenizer.embeddings'), // [3, 1024, 96, 96]
  };

  // === Two-Stream Backbone ===
  const backbone = {
    latentInit: get('backbone.latent_init'),         // [1, 1792, 1024]
    normTriplane: { weight: get('backbone.norm_triplane.weight'), bias: get('backbone.norm_triplane.bias') },
    projTriplane: { weight: get('backbone.proj_triplane.weight'), bias: get('backbone.proj_triplane.bias') },
    normImage: { weight: get('backbone.norm_image.weight'), bias: get('backbone.norm_image.bias') },
    projImage: { weight: get('backbone.proj_image.weight'), bias: get('backbone.proj_image.bias') },
    normLatent: { weight: get('backbone.norm_latent.weight'), bias: get('backbone.norm_latent.bias') },
    projLatent: { weight: get('backbone.proj_latent.weight'), bias: get('backbone.proj_latent.bias') },
    projOut: { weight: get('backbone.proj_out.weight'), bias: get('backbone.proj_out.bias') },
    mainBlocks: [],
  };

  // 4 TwoStreamBlocks, each with fuse_block_in, 3 transformer_blocks, fuse_block_out
  for (let b = 0; b < 4; b++) {
    const bp = `backbone.main_blocks.${b}`;

    // Helper to load a FuseBlock
    function loadFuseBlock(prefix) {
      return {
        attn: {
          wq: get(`${prefix}.attn.wq.weight`),
          wk: get(`${prefix}.attn.wk.weight`),
          wv: get(`${prefix}.attn.wv.weight`),
          proj: { weight: get(`${prefix}.attn.proj.weight`), bias: get(`${prefix}.attn.proj.bias`) },
        },
        normZ1: { weight: get(`${prefix}.norm_z1.weight`), bias: get(`${prefix}.norm_z1.bias`) },
        normX: tryGet(`${prefix}.norm_x.weight`) ? {
          weight: get(`${prefix}.norm_x.weight`), bias: get(`${prefix}.norm_x.bias`),
        } : null,
        normZ2: { weight: get(`${prefix}.norm_z2.weight`), bias: get(`${prefix}.norm_z2.bias`) },
        ff: {
          geglu: { weight: get(`${prefix}.ff.net.0.proj.weight`), bias: get(`${prefix}.ff.net.0.proj.bias`) },
          proj: { weight: get(`${prefix}.ff.net.2.weight`), bias: get(`${prefix}.ff.net.2.bias`) },
        },
      };
    }

    // Helper to load a BasicBlock
    function loadBasicBlock(prefix) {
      return {
        norm1: { weight: get(`${prefix}.norm1.weight`), bias: get(`${prefix}.norm1.bias`) },
        attn1: {  // self-attention
          wq: get(`${prefix}.attn1.wq.weight`),
          wk: get(`${prefix}.attn1.wk.weight`),
          wv: get(`${prefix}.attn1.wv.weight`),
          proj: { weight: get(`${prefix}.attn1.proj.weight`), bias: get(`${prefix}.attn1.proj.bias`) },
        },
        norm2: { weight: get(`${prefix}.norm2.weight`), bias: get(`${prefix}.norm2.bias`) },
        attn2: {  // cross-attention (or self if no encoder_hidden_states)
          wq: get(`${prefix}.attn2.wq.weight`),
          wk: get(`${prefix}.attn2.wk.weight`),
          wv: get(`${prefix}.attn2.wv.weight`),
          proj: { weight: get(`${prefix}.attn2.proj.weight`), bias: get(`${prefix}.attn2.proj.bias`) },
        },
        norm3: { weight: get(`${prefix}.norm3.weight`), bias: get(`${prefix}.norm3.bias`) },
        ff: {
          geglu: { weight: get(`${prefix}.ff.net.0.proj.weight`), bias: get(`${prefix}.ff.net.0.proj.bias`) },
          proj: { weight: get(`${prefix}.ff.net.2.weight`), bias: get(`${prefix}.ff.net.2.bias`) },
        },
      };
    }

    backbone.mainBlocks.push({
      fuseBlockIn: loadFuseBlock(`${bp}.fuse_block_in`),
      transformerBlocks: [0, 1, 2].map(i => loadBasicBlock(`${bp}.transformer_block.${i}`)),
      fuseBlockOut: loadFuseBlock(`${bp}.fuse_block_out`),
    });
  }

  // === Post-Processor (PixelShuffle) ===
  const postProcessor = {
    convLayers: [
      { weight: get('post_processor.upsample.0.weight'), bias: get('post_processor.upsample.0.bias') },
      { weight: get('post_processor.upsample.2.weight'), bias: get('post_processor.upsample.2.bias') },
      { weight: get('post_processor.upsample.4.weight'), bias: get('post_processor.upsample.4.bias') },
      { weight: get('post_processor.upsample.6.weight'), bias: get('post_processor.upsample.6.bias') },
    ],
  };

  // === Decoder (MaterialMLP) ===
  const decoder = {
    heads: {},
  };
  for (const headName of ['density', 'features', 'perturb_normal', 'vertex_offset']) {
    const layers = [];
    for (let i = 0; ; i += 2) {
      const w = tryGet(`decoder.heads.${headName}.${i}.weight`);
      const b = tryGet(`decoder.heads.${headName}.${i}.bias`);
      if (!w) break;
      layers.push({ weight: w, bias: b });
    }
    decoder.heads[headName] = layers;
  }

  // === Image Estimator (CLIP for roughness/metallic) ===
  // For v1, we can run this on CPU or skip and use defaults.
  // Load the estimation heads at minimum.
  const imageEstimator = {
    // CLIP visual encoder weights would go here
    // For now, just load the roughness/metallic prediction heads
    heads: {},
  };
  for (const headName of ['roughness', 'metallic']) {
    const subLayers = [];
    for (let sub = 0; sub < 3; sub++) {
      const layers = [];
      for (let i = 0; ; i += 2) {
        const w = tryGet(`image_estimator.heads.${headName}.${sub}.${i}.weight`);
        const b = tryGet(`image_estimator.heads.${headName}.${sub}.${i}.bias`);
        if (!w) break;
        layers.push({ weight: w, bias: b });
      }
      if (layers.length > 0) subLayers.push(layers);
    }
    imageEstimator.heads[headName] = subLayers;
  }

  console.log(`Loaded ${tensors.size} SF3D tensors from weight file`);

  // Raw tensor access for modules that need direct weight lookup by name
  // (e.g., CLIP visual encoder in clip_estimator.js)
  const _rawGet = (name) => get(name);
  const _rawGetCPU = (name) => getCPU(name);
  const _rawTryGet = (name) => tryGet(name);
  const _rawHas = (name) => tensors.has(name);

  return {
    imageTokenizer,
    cameraEmbedder,
    tokenizer,
    backbone,
    postProcessor,
    decoder,
    imageEstimator,
    _rawGet,
    _rawGetCPU,
    _rawTryGet,
    _rawHas,
  };
}
