/**
 * backbone.js — DINOv2 ViT-Large backbone dispatch for MoGe-2.
 *
 * Architecture:
 *   1. Patch embedding: image → [N+1, 1024] tokens (14×14 patches + CLS)
 *   2. 24 transformer blocks, each:
 *      a. LayerNorm1 → Attention (QKV → scores → softmax → apply → proj) → LayerScale1 + residual
 *      b. LayerNorm2 → GELU MLP (fc1 → GELU → fc2) → LayerScale2 + residual
 *   3. Extract intermediate features at layers [5, 11, 17, 23]
 *   4. Project each intermediate feature with 1x1 conv and sum
 *
 * Produces: [1024, tokenH, tokenW] feature map + [1024] CLS token
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';

import patchEmbedWGSL from '../shaders/patch_embed_dinov2.wgsl?raw';
import layerNormWGSL from '../shaders/layernorm_vit.wgsl?raw';
import attentionWGSL from '../shaders/attention.wgsl?raw';
import linearWGSL from '../shaders/linear.wgsl?raw';
import linearGeluWGSL from '../shaders/linear_gelu.wgsl?raw';
import layerscaleWGSL from '../shaders/layerscale.wgsl?raw';
import transposeWGSL from '../shaders/transpose_nd.wgsl?raw';

const MAX_WG = 65535;
function splitWG(total) {
  if (total <= MAX_WG) return [total, 1];
  return [MAX_WG, Math.ceil(total / MAX_WG)];
}
function ceilDiv(a, b) { return Math.ceil(a / b); }

function makeUniform(device, data) {
  const buf = device.createBuffer({
    size: Math.max(data.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer || data));
  buf.unmap();
  return buf;
}

// Cache key from typed array contents
function uniformKey(data) {
  const bytes = new Uint8Array(data.buffer || data);
  let h = 0;
  for (let i = 0; i < bytes.length; i++) h = (h * 31 + bytes[i]) | 0;
  return `u_${bytes.length}_${h}`;
}

// DINOv2 ViT-Large config
const VIT_CONFIG = {
  dim: 1024,
  numHeads: 16,
  headDim: 64,
  numLayers: 24,
  patchSize: 14,
  channels: 3,
  intermediateLayers: [5, 11, 17, 23],
  // Standard GELU MLP (not SwiGLU — verified from checkpoint weights)
  mlpHiddenDim: 4096,
  scale: 1.0 / Math.sqrt(64),
  eps: 1e-6,
};

export class DINOv2Backbone {
  constructor(device) {
    this.device = device;
    this.pipelines = {};
    this._uniformCache = new Map();
    this._bindGroupCache = new Map();
  }

  init() {
    const device = this.device;
    const make = (code, entry) => device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: entry },
    });

    this.pipelines.patchEmbed = make(patchEmbedWGSL, 'main');
    this.pipelines.layerNorm = make(layerNormWGSL, 'main');
    this.pipelines.attnScores = make(attentionWGSL, 'computeScores');
    this.pipelines.attnSoftmax = make(attentionWGSL, 'softmax');
    this.pipelines.attnApply = make(attentionWGSL, 'applyAttn');
    this.pipelines.linear = make(linearWGSL, 'main');
    this.pipelines.linearGelu = make(linearGeluWGSL, 'main');
    this.pipelines.layerScale = make(layerscaleWGSL, 'main');
    this.pipelines.transpose = make(transposeWGSL, 'main');

    // QKV split shader
    const splitModule = device.createShaderModule({
      code: `
        struct P { N: u32, D: u32, numWgX: u32 }
        @group(0) @binding(0) var<uniform> p: P;
        @group(0) @binding(1) var<storage, read> qkv: array<f32>;
        @group(0) @binding(2) var<storage, read_write> q: array<f32>;
        @group(0) @binding(3) var<storage, read_write> k: array<f32>;
        @group(0) @binding(4) var<storage, read_write> v: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
          let idx = (wgid.x + wgid.y * p.numWgX) * 256u + lid.x;
          if (idx >= p.N * p.D) { return; }
          let row = idx / p.D;
          let col = idx % p.D;
          let D3 = p.D * 3u;
          q[idx] = qkv[row * D3 + col];
          k[idx] = qkv[row * D3 + p.D + col];
          v[idx] = qkv[row * D3 + 2u * p.D + col];
        }
      `,
    });
    this.pipelines.splitQKV = device.createComputePipeline({
      layout: 'auto',
      compute: { module: splitModule, entryPoint: 'main' },
    });

    // Element-wise add shader (for summing intermediate feature projections)
    const addModule = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read_write> dst: array<f32>;
        @group(0) @binding(1) var<storage, read> src: array<f32>;
        struct P { count: u32, numWgX: u32 }
        @group(0) @binding(2) var<uniform> p: P;

        @compute @workgroup_size(256)
        fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
          let idx = (wgid.x + wgid.y * p.numWgX) * 256u + lid.x;
          if (idx >= p.count) { return; }
          dst[idx] = dst[idx] + src[idx];
        }
      `,
    });
    this.pipelines.add = device.createComputePipeline({
      layout: 'auto',
      compute: { module: addModule, entryPoint: 'main' },
    });
  }

  /**
   * Run the DINOv2 backbone.
   * @param {GPUCommandEncoder} encoder
   * @param {GPUBuffer} imageBuf - [3, imgH, imgW] normalized CHW image
   * @param {Object} weights - encoder weights from weight loader
   * @param {number} tokenH
   * @param {number} tokenW
   * @returns {{ featureBuf: GPUBuffer, clsTokenBuf: GPUBuffer }}
   */
  _ensureWorkBuffers(tokenH, tokenW) {
    if (this._workBufs && this._workTokenH === tokenH && this._workTokenW === tokenW) return;
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const numPatches = tokenH * tokenW;
    const N = numPatches + 1;
    const T = N * D;

    // Destroy old buffers if grid size changed
    if (this._workBufs) {
      for (const buf of Object.values(this._workBufs)) buf.destroy();
      this._bindGroupCache.clear();
    }

    this._workBufs = {
      tokenBufA: createEmptyBuffer(device, T * 4),
      tokenBufB: createEmptyBuffer(device, T * 4),
      normBuf: createEmptyBuffer(device, T * 4),
      qBuf: createEmptyBuffer(device, T * 4),
      kBuf: createEmptyBuffer(device, T * 4),
      vBuf: createEmptyBuffer(device, T * 4),
      scoreBuf: createEmptyBuffer(device, VIT_CONFIG.numHeads * N * N * 4),
      attnOutBuf: createEmptyBuffer(device, T * 4),
      projOutBuf: createEmptyBuffer(device, T * 4),
      hiddenBuf: createEmptyBuffer(device, N * VIT_CONFIG.mlpHiddenDim * 4),
      ffnOutBuf: createEmptyBuffer(device, T * 4),
      qkvWorkBuf: createEmptyBuffer(device, N * 3 * D * 4),
    };
    this._workTokenH = tokenH;
    this._workTokenW = tokenW;
  }

  encode(encoder, imageBuf, weights, tokenH, tokenW) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const numPatches = tokenH * tokenW;
    const N = numPatches + 1; // +1 for CLS
    const T = N * D; // total token elements

    this._ensureWorkBuffers(tokenH, tokenW);
    const wb = this._workBufs;

    // --- Patch embedding ---
    // tokenBufA is the initial token buffer
    this._encodePatchEmbed(encoder, imageBuf, weights, wb.tokenBufA, tokenH, tokenW);

    // --- Transformer blocks ---
    const intermediateFeatures = [];
    let currentTokens = wb.tokenBufA;

    const { normBuf, qBuf, kBuf, vBuf, scoreBuf, attnOutBuf, projOutBuf, hiddenBuf, ffnOutBuf, qkvWorkBuf, tokenBufA, tokenBufB } = wb;

    for (let l = 0; l < VIT_CONFIG.numLayers; l++) {
      // LayerNorm1
      this._encodeLayerNorm(encoder, currentTokens, normBuf, weights, `encoder.backbone.blocks.${l}.norm1`, N);

      // Attention: QKV projections
      this._encodeQKV(encoder, normBuf, qBuf, kBuf, vBuf, weights, l, N, qkvWorkBuf);

      // Attention scores
      this._encodeAttnScores(encoder, qBuf, kBuf, scoreBuf, N);

      // Softmax
      this._encodeAttnSoftmax(encoder, scoreBuf, N);

      // Apply attention
      this._encodeAttnApply(encoder, scoreBuf, vBuf, attnOutBuf, N);

      // Output projection
      this._encodeLinear(encoder, attnOutBuf, projOutBuf, weights, `encoder.backbone.blocks.${l}.attn.proj`, N, D, D);

      // LayerScale1 + residual: output = currentTokens + ls1.gamma * projOutBuf
      // Write to the OTHER buffer to avoid read/write race
      const attnResidualOut = (currentTokens === tokenBufA) ? tokenBufB : tokenBufA;
      this._encodeLayerScaleResidual(encoder, projOutBuf, currentTokens, attnResidualOut, weights, `encoder.backbone.blocks.${l}.ls1`, T, D);
      currentTokens = attnResidualOut;

      // LayerNorm2
      this._encodeLayerNorm(encoder, currentTokens, normBuf, weights, `encoder.backbone.blocks.${l}.norm2`, N);

      // GELU MLP: fc1 (linear+GELU) then fc2 (linear)
      this._encodeLinearGelu(encoder, normBuf, hiddenBuf, weights, `encoder.backbone.blocks.${l}.mlp.fc1`, N, D, VIT_CONFIG.mlpHiddenDim);
      this._encodeLinear(encoder, hiddenBuf, ffnOutBuf, weights, `encoder.backbone.blocks.${l}.mlp.fc2`, N, VIT_CONFIG.mlpHiddenDim, D);

      // LayerScale2 + residual: write to the other buffer
      const ffnResidualOut = (currentTokens === tokenBufA) ? tokenBufB : tokenBufA;
      this._encodeLayerScaleResidual(encoder, ffnOutBuf, currentTokens, ffnResidualOut, weights, `encoder.backbone.blocks.${l}.ls2`, T, D);
      currentTokens = ffnResidualOut;

      // Capture intermediate features at specified layers
      if (VIT_CONFIG.intermediateLayers.includes(l)) {
        // Copy current token state for feature extraction
        const snapBuf = createEmptyBuffer(device, T * 4, GPUBufferUsage.COPY_DST);
        encoder.copyBufferToBuffer(currentTokens, 0, snapBuf, 0, T * 4);
        intermediateFeatures.push({ buffer: snapBuf, layerIdx: l });
      }
    }

    // --- Project and sum intermediate features ---
    // Each intermediate feature gets a 1x1 conv projection, then all are summed
    const featureBuf = createEmptyBuffer(device, D * numPatches * 4);
    let sumBuf = null;
    let normedClsBuf = null; // Will hold the normed CLS token from the last layer

    for (let i = 0; i < intermediateFeatures.length; i++) {
      const { buffer: snapBuf } = intermediateFeatures[i];

      // Upstream flow (from get_intermediate_layers):
      //   1. Apply backbone final LayerNorm to intermediate block output
      //   2. Strip CLS token → [numPatches, D]
      //   3. Permute → [D, numPatches]
      //   4. Unflatten → [D, tokenH, tokenW]
      //   5. 1x1 conv projection → [D, tokenH, tokenW]
      //
      // Step 0: Apply backbone final norm to snapshot
      const normedBuf = createEmptyBuffer(device, T * 4);
      this._encodeLayerNorm(encoder, snapBuf, normedBuf, weights, 'encoder.backbone.norm', N);

      // Capture normed CLS token from last intermediate layer (for scale head)
      if (i === intermediateFeatures.length - 1) {
        normedClsBuf = normedBuf; // CLS token is at offset 0 (first D floats)
      }

      // Step 1: Linear projection on [numPatches, D] → [numPatches, D] (skip CLS via offset)
      const projBuf = createEmptyBuffer(device, D * numPatches * 4);
      this._encodeOutputProjection(encoder, normedBuf, projBuf, weights, i, N, numPatches);

      // Step 2: Transpose [numPatches, D] → [D, numPatches] (= [D, tokenH, tokenW] in CHW)
      const transposedBuf = createEmptyBuffer(device, D * numPatches * 4);
      this._encodeTranspose(encoder, projBuf, transposedBuf, numPatches, D);

      if (sumBuf === null) {
        sumBuf = transposedBuf;
      } else {
        this._encodeAdd(encoder, sumBuf, transposedBuf, D * numPatches);
      }
    }

    // Don't submit here — caller is responsible for submitting the encoder.
    // Return debug buffers for post-submit readback.
    return {
      featureBuf: sumBuf || featureBuf,
      clsTokenBuf: normedClsBuf || currentTokens,
      tokenH,
      tokenW,
      _debugSnaps: intermediateFeatures,
    };
  }

  _cachedUniform(data) {
    const key = uniformKey(data);
    if (this._uniformCache.has(key)) return this._uniformCache.get(key);
    const buf = makeUniform(this.device, data);
    this._uniformCache.set(key, buf);
    return buf;
  }

  _cachedBindGroup(tag, layout, entries) {
    // Build cache key from tag + all bound buffer identities
    let key = tag;
    for (const e of entries) {
      const r = e.resource;
      const buf = r.buffer;
      if (!buf._bgId) buf._bgId = ++DINOv2Backbone._bgIdCounter;
      key += `_${buf._bgId}`;
      if (r.offset !== undefined) key += `@${r.offset}`;
    }
    if (this._bindGroupCache.has(key)) return this._bindGroupCache.get(key);
    const bg = this.device.createBindGroup({ layout, entries });
    this._bindGroupCache.set(key, bg);
    return bg;
  }

  // --- Private dispatch methods ---

  _encodePatchEmbed(encoder, imageBuf, weights, outputBuf, tokenH, tokenW) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const ps = VIT_CONFIG.patchSize;
    const numTokens = tokenH * tokenW + 1;
    const totalWG = ceilDiv(numTokens * D, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([tokenH * ps, tokenW * ps, ps, tokenH, tokenW, 3, D, numTokens, wgX]);
    const paramsBuf = this._cachedUniform(paramsData);

    const bg = device.createBindGroup({
      layout: this.pipelines.patchEmbed.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: imageBuf } },
        { binding: 2, resource: { buffer: weights.encoder.patchEmbed.weight } },
        { binding: 3, resource: { buffer: weights.encoder.patchEmbed.bias } },
        { binding: 4, resource: { buffer: weights.encoder.clsToken } },
        { binding: 5, resource: { buffer: weights.encoder.posEmbed } },
        { binding: 6, resource: { buffer: outputBuf } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.patchEmbed);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLayerNorm(enc, input, output, weights, prefix, N) {
    const device = this.device;
    const D = VIT_CONFIG.dim;

    const paramsData = new ArrayBuffer(16);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, D, true);
    v.setFloat32(8, VIT_CONFIG.eps, true);
    const paramsBuf = this._cachedUniform(new Uint8Array(paramsData));

    const gammaKey = `${prefix}.weight`;
    const betaKey = `${prefix}.bias`;
    let gamma = weights.encoder.blockWeights?.[gammaKey];
    let beta = weights.encoder.blockWeights?.[betaKey];
    if (!gamma && prefix === 'encoder.backbone.norm') {
      gamma = weights.encoder.norm?.weight;
      beta = weights.encoder.norm?.bias;
    }
    if (!gamma || !beta) { console.warn(`Missing LayerNorm weights: ${prefix}`); return; }

    const bg = this._cachedBindGroup(`ln_${prefix}`, this.pipelines.layerNorm.getBindGroupLayout(0), [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: gamma } },
      { binding: 3, resource: { buffer: beta } },
      { binding: 4, resource: { buffer: output } },
    ]);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.layerNorm);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(N);
    pass.end();
  }

  _encodeQKV(enc, input, qBuf, kBuf, vBuf, weights, layerIdx, N, qkvWorkBuf) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const D3 = 3 * D;
    const prefix = `encoder.backbone.blocks.${layerIdx}.attn.qkv`;
    const qkvWeight = weights.encoder.blockWeights?.[`${prefix}.weight`];
    const qkvBias = weights.encoder.blockWeights?.[`${prefix}.bias`];

    if (!qkvWeight || !qkvBias) {
      console.warn(`Missing QKV weights for layer ${layerIdx}`);
      return;
    }

    // Project to [N, 3*D] with one linear call, then split Q/K/V
    this._encodeLinearFull(enc, input, qkvWorkBuf, qkvWeight, qkvBias, N, D, D3);

    // Split [N, 3*D] → Q [N, D], K [N, D], V [N, D]
    this._encodeSplitQKV(enc, qkvWorkBuf, qBuf, kBuf, vBuf, N, D);
  }

  _encodeLinearFull(enc, input, output, weight, bias, numRows, inDim, outDim) {
    const device = this.device;
    const totalWG = ceilDiv(numRows * outDim, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([numRows, inDim, outDim, wgX, 1]);
    const paramsBuf = this._cachedUniform(paramsData);

    const entries = [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: bias } },
      { binding: 4, resource: { buffer: output } },
    ];
    const bg = this._cachedBindGroup('linFull', this.pipelines.linear.getBindGroupLayout(0), entries);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.linear);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeSplitQKV(enc, qkvBuf, qBuf, kBuf, vBuf, N, D) {
    const device = this.device;
    const total = N * D;
    const totalWG = ceilDiv(total, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsBuf = this._cachedUniform(new Uint32Array([N, D, wgX]));

    const bg = this._cachedBindGroup('splitQKV', this.pipelines.splitQKV.getBindGroupLayout(0), [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: qkvBuf } },
      { binding: 2, resource: { buffer: qBuf } },
      { binding: 3, resource: { buffer: kBuf } },
      { binding: 4, resource: { buffer: vBuf } },
    ]);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.splitQKV);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLinear(enc, input, output, weights, prefix, numRows, inDim, outDim) {
    const device = this.device;
    const totalWG = ceilDiv(numRows * outDim, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([numRows, inDim, outDim, wgX, 1]);
    const paramsBuf = this._cachedUniform(paramsData);

    const weight = weights.encoder.blockWeights?.[`${prefix}.weight`];
    const bias = weights.encoder.blockWeights?.[`${prefix}.bias`];

    if (!weight || !bias) {
      console.warn(`Missing linear weights: ${prefix}`);
      return;
    }

    const bg = this._cachedBindGroup('lin', this.pipelines.linear.getBindGroupLayout(0), [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: bias } },
      { binding: 4, resource: { buffer: output } },
    ]);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.linear);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLinearWithOffsets(enc, input, weight, wOffset, wSize, bias, bOffset, bSize, output, numRows, inDim, outDim) {
    const device = this.device;
    const totalWG = ceilDiv(numRows * outDim, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([numRows, inDim, outDim, wgX, 1]);
    const paramsBuf = this._cachedUniform(paramsData);

    const bg = this._cachedBindGroup('linOff', this.pipelines.linear.getBindGroupLayout(0), [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: weight, offset: wOffset, size: wSize } },
      { binding: 3, resource: { buffer: bias, offset: bOffset, size: bSize } },
      { binding: 4, resource: { buffer: output } },
    ]);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.linear);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeAttnScores(enc, qBuf, kBuf, scoreBuf, N) {
    const device = this.device;
    const { numHeads, dim, headDim, scale } = VIT_CONFIG;
    const total = numHeads * N * N;
    const totalWG = ceilDiv(total, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new ArrayBuffer(24);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, dim, true);
    v.setUint32(8, numHeads, true);
    v.setUint32(12, headDim, true);
    v.setFloat32(16, scale, true);
    v.setUint32(20, wgX, true);
    const paramsBuf = this._cachedUniform(new Uint8Array(paramsData));

    const bg = this._cachedBindGroup('attnS', this.pipelines.attnScores.getBindGroupLayout(0), [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: qBuf } },
      { binding: 2, resource: { buffer: kBuf } },
      { binding: 3, resource: { buffer: scoreBuf } },
    ]);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.attnScores);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeAttnSoftmax(enc, scoreBuf, N) {
    const device = this.device;
    const totalRows = VIT_CONFIG.numHeads * N;
    const totalWG = ceilDiv(totalRows, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([N, VIT_CONFIG.numHeads, wgX]);
    const paramsBuf = this._cachedUniform(paramsData);

    const bg = this._cachedBindGroup('smx', this.pipelines.attnSoftmax.getBindGroupLayout(0), [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: scoreBuf } },
    ]);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.attnSoftmax);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeAttnApply(enc, scoreBuf, vBuf, output, N) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const totalWG = ceilDiv(N * D, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([N, D, VIT_CONFIG.numHeads, VIT_CONFIG.headDim, wgX]);
    const paramsBuf = this._cachedUniform(paramsData);

    const bg = this._cachedBindGroup('attnA', this.pipelines.attnApply.getBindGroupLayout(0), [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: scoreBuf } },
      { binding: 2, resource: { buffer: vBuf } },
      { binding: 3, resource: { buffer: output } },
    ]);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.attnApply);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLayerScaleResidual(enc, input, residual, output, weights, prefix, count, D) {
    const device = this.device;
    const totalWG = ceilDiv(count, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([count, D, wgX]);
    const paramsBuf = this._cachedUniform(paramsData);

    const gamma = weights.encoder.blockWeights?.[`${prefix}.gamma`];
    if (!gamma) { console.warn(`Missing LayerScale gamma: ${prefix}`); return; }

    const bg = this._cachedBindGroup('ls', this.pipelines.layerScale.getBindGroupLayout(0), [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: gamma } },
      { binding: 3, resource: { buffer: residual } },
      { binding: 4, resource: { buffer: output } },
    ]);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.layerScale);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLinearGelu(enc, input, output, weights, prefix, numRows, inDim, outDim) {
    const device = this.device;
    const totalWG = ceilDiv(numRows * outDim, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([numRows, inDim, outDim, wgX, 1]);
    const paramsBuf = this._cachedUniform(paramsData);

    const weight = weights.encoder.blockWeights?.[`${prefix}.weight`];
    const bias = weights.encoder.blockWeights?.[`${prefix}.bias`];
    if (!weight || !bias) { console.warn(`Missing linear+GELU weights: ${prefix}`); return; }

    const bg = this._cachedBindGroup('linG', this.pipelines.linearGelu.getBindGroupLayout(0), [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: bias } },
      { binding: 4, resource: { buffer: output } },
    ]);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.linearGelu);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeOutputProjection(enc, tokensBuf, outputBuf, weights, projIdx, N, numPatches) {
    // Extract patch tokens (skip CLS), reshape to CHW, and project with 1x1 conv
    // This is a simplified version — the upstream code does permute + unflatten + conv
    // For now, we treat it as a linear: [numPatches, D] → [numPatches, D]
    // using the output_projections weight
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const totalWG = ceilDiv(numPatches * D, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([numPatches, D, D, wgX]);
    const paramsBuf = this._cachedUniform(paramsData);

    const proj = weights.encoder.outputProjections[projIdx];

    const bg = this._cachedBindGroup(`outProj${projIdx}`, this.pipelines.linear.getBindGroupLayout(0), [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: tokensBuf, offset: D * 4, size: numPatches * D * 4 } },
      { binding: 2, resource: { buffer: proj.weight } },
      { binding: 3, resource: { buffer: proj.bias } },
      { binding: 4, resource: { buffer: outputBuf } },
    ]);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.linear);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeTranspose(enc, input, output, rows, cols) {
    const device = this.device;
    const total = rows * cols;
    const totalWG = ceilDiv(total, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([rows, cols, wgX]);
    const paramsBuf = this._cachedUniform(paramsData);

    const bg = this._cachedBindGroup('trans', this.pipelines.transpose.getBindGroupLayout(0), [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ]);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.transpose);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  /**
   * Run layer-by-layer comparison against PyTorch reference tensors.
   * Call from browser console: await window.__mogeInference.backbone.debugCompare(...)
   */
  async debugCompare(imageBuf, weights, tokenH, tokenW) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const numPatches = tokenH * tokenW;
    const N = numPatches + 1;
    const T = N * D;

    // Load reference manifest
    const manifestResp = await fetch('/layer_dumps/manifest.json');
    const manifest = await manifestResp.json();

    async function loadRef(name) {
      const info = manifest[name];
      if (!info) { console.warn(`No reference for ${name}`); return null; }
      const resp = await fetch(`/layer_dumps/${info.file}`);
      return new Float32Array(await resp.arrayBuffer());
    }

    const results = {};

    function compareArrays(label, gpu, ref) {
      if (!ref) { console.log(`  ${label}: no reference`); return null; }
      const n = Math.min(gpu.length, ref.length);
      let maxErr = 0, sumErr = 0, sumSq = 0;
      let gpuMin = Infinity, gpuMax = -Infinity, refMin = Infinity, refMax = -Infinity;
      let gpuSum = 0, refSum = 0, gpuSqSum = 0, refSqSum = 0;
      let worstIdx = 0, nanCount = 0, infCount = 0;
      let firstNanIdx = -1;
      for (let i = 0; i < n; i++) {
        if (isNaN(gpu[i])) { nanCount++; if (firstNanIdx < 0) firstNanIdx = i; continue; }
        if (!isFinite(gpu[i])) { infCount++; continue; }
        const err = Math.abs(gpu[i] - ref[i]);
        sumErr += err;
        sumSq += err * err;
        if (err > maxErr) { maxErr = err; worstIdx = i; }
        if (gpu[i] < gpuMin) gpuMin = gpu[i];
        if (gpu[i] > gpuMax) gpuMax = gpu[i];
        if (ref[i] < refMin) refMin = ref[i];
        if (ref[i] > refMax) refMax = ref[i];
        gpuSum += gpu[i]; refSum += ref[i];
        gpuSqSum += gpu[i] * gpu[i]; refSqSum += ref[i] * ref[i];
      }
      const finiteN = n - nanCount - infCount;
      const gpuMean = gpuSum / finiteN, refMean = refSum / finiteN;
      const gpuStd = Math.sqrt(Math.max(0, gpuSqSum / finiteN - gpuMean * gpuMean));
      const refStd = Math.sqrt(Math.max(0, refSqSum / finiteN - refMean * refMean));
      const meanErr = sumErr / finiteN;
      const rmsErr = Math.sqrt(sumSq / finiteN);
      const relStd = refStd > 0 ? gpuStd / refStd : NaN;
      const worstRow = Math.floor(worstIdx / D);
      const worstCol = worstIdx % D;

      const result = {
        label, maxErr, meanErr, rmsErr, relStd,
        gpu: { min: gpuMin, max: gpuMax, mean: gpuMean, std: gpuStd },
        ref: { min: refMin, max: refMax, mean: refMean, std: refStd },
        worstIdx, worstRow, worstCol,
        worstGpu: gpu[worstIdx], worstRef: ref[worstIdx],
      };
      results[label] = result;

      const nanStr = nanCount > 0 ? ` NaN=${nanCount}${firstNanIdx >= 0 ? `@${firstNanIdx}` : ''}` : '';
      const infStr = infCount > 0 ? ` Inf=${infCount}` : '';
      console.log(`  ${label}: maxErr=${maxErr.toFixed(4)} rmsErr=${rmsErr.toFixed(4)} relStd=${relStd.toFixed(4)} | GPU std=${gpuStd.toFixed(4)} REF std=${refStd.toFixed(4)} | worst@[${worstRow},${worstCol}] gpu=${gpu[worstIdx]?.toFixed(4)} ref=${ref[worstIdx]?.toFixed(4)}${nanStr}${infStr}`);
      return result;
    }

    console.log('\n=== BACKBONE COMPARISON ===\n');

    // --- Stage 1: Patch embedding ---
    {
      const tokenBuf = createEmptyBuffer(device, T * 4);
      const enc = device.createCommandEncoder();
      this._encodePatchEmbed(enc, imageBuf, weights, tokenBuf, tokenH, tokenW);
      device.queue.submit([enc.finish()]);
      const gpu = await readBuffer(device, tokenBuf, T * 4);
      const ref = await loadRef('tokens_after_pos_embed');
      compareArrays('patch_embed', gpu, ref);
      tokenBuf.destroy();
    }

    // --- Stage 2: Run blocks, compare at checkpoints ---
    const checkpoints = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 23];
    let tokenBufA = createEmptyBuffer(device, T * 4);
    let tokenBufB = createEmptyBuffer(device, T * 4);
    {
      const enc = device.createCommandEncoder();
      this._encodePatchEmbed(enc, imageBuf, weights, tokenBufA, tokenH, tokenW);
      device.queue.submit([enc.finish()]);
    }
    let currentTokens = tokenBufA;

    const normBuf = createEmptyBuffer(device, T * 4);
    const qBuf = createEmptyBuffer(device, T * 4);
    const kBuf = createEmptyBuffer(device, T * 4);
    const vBuf = createEmptyBuffer(device, T * 4);
    const scoreBuf = createEmptyBuffer(device, VIT_CONFIG.numHeads * N * N * 4);
    const attnOutBuf = createEmptyBuffer(device, T * 4);
    const projOutBuf = createEmptyBuffer(device, T * 4);
    const hiddenBuf = createEmptyBuffer(device, N * VIT_CONFIG.mlpHiddenDim * 4);
    const ffnOutBuf = createEmptyBuffer(device, T * 4);
    const qkvWorkBuf = createEmptyBuffer(device, N * 3 * D * 4);

    const DO_BLOCK0_SUBSTEPS = true;
    for (let l = 0; l < VIT_CONFIG.numLayers; l++) {
      // For block 0, run sub-steps individually and compare each
      if (l === 0 && DO_BLOCK0_SUBSTEPS) {
        console.log('\n--- Block 0 sub-steps ---');

        // norm1
        let enc = device.createCommandEncoder();
        this._encodeLayerNorm(enc, currentTokens, normBuf, weights, `encoder.backbone.blocks.0.norm1`, N);
        device.queue.submit([enc.finish()]);
        const norm1Gpu = await readBuffer(device, normBuf, T * 4);
        const norm1Ref = await loadRef('block_0_norm1');
        compareArrays('b0_norm1', norm1Gpu, norm1Ref);

        // QKV
        enc = device.createCommandEncoder();
        this._encodeQKV(enc, normBuf, qBuf, kBuf, vBuf, weights, 0, N, qkvWorkBuf);
        device.queue.submit([enc.finish()]);
        const qGpu = await readBuffer(device, qBuf, T * 4);
        const kGpu = await readBuffer(device, kBuf, T * 4);
        const vGpu = await readBuffer(device, vBuf, T * 4);
        // QKV ref is [N, 3*D] — split into Q, K, V
        const qkvRef = await loadRef('block_0_qkv');
        if (qkvRef) {
          const qRef = new Float32Array(N * D);
          const kRef = new Float32Array(N * D);
          const vRef = new Float32Array(N * D);
          for (let i = 0; i < N; i++) {
            for (let d = 0; d < D; d++) {
              qRef[i * D + d] = qkvRef[i * 3 * D + d];
              kRef[i * D + d] = qkvRef[i * 3 * D + D + d];
              vRef[i * D + d] = qkvRef[i * 3 * D + 2 * D + d];
            }
          }
          compareArrays('b0_Q', qGpu, qRef);
          compareArrays('b0_K', kGpu, kRef);
          compareArrays('b0_V', vGpu, vRef);
        }

        // Attention scores + softmax + apply
        enc = device.createCommandEncoder();
        this._encodeAttnScores(enc, qBuf, kBuf, scoreBuf, N);
        this._encodeAttnSoftmax(enc, scoreBuf, N);
        this._encodeAttnApply(enc, scoreBuf, vBuf, attnOutBuf, N);
        device.queue.submit([enc.finish()]);

        // Attn output projection
        enc = device.createCommandEncoder();
        this._encodeLinear(enc, attnOutBuf, projOutBuf, weights, `encoder.backbone.blocks.0.attn.proj`, N, D, D);
        device.queue.submit([enc.finish()]);
        const projGpu = await readBuffer(device, projOutBuf, T * 4);
        const attnRef = await loadRef('block_0_attn_out');
        compareArrays('b0_attn_proj', projGpu, attnRef);

        // LayerScale1 + residual
        enc = device.createCommandEncoder();
        const attnResidualOut = (currentTokens === tokenBufA) ? tokenBufB : tokenBufA;
        this._encodeLayerScaleResidual(enc, projOutBuf, currentTokens, attnResidualOut, weights, `encoder.backbone.blocks.0.ls1`, T, D);
        device.queue.submit([enc.finish()]);
        currentTokens = attnResidualOut;
        const ls1Gpu = await readBuffer(device, currentTokens, T * 4);
        const ls1Ref = await loadRef('block_0_after_ls1');
        compareArrays('b0_after_ls1', ls1Gpu, ls1Ref);

        // norm2
        enc = device.createCommandEncoder();
        this._encodeLayerNorm(enc, currentTokens, normBuf, weights, `encoder.backbone.blocks.0.norm2`, N);
        device.queue.submit([enc.finish()]);
        const norm2Gpu = await readBuffer(device, normBuf, T * 4);
        const norm2Ref = await loadRef('block_0_norm2');
        compareArrays('b0_norm2', norm2Gpu, norm2Ref);

        // MLP fc1 (linear + GELU)
        enc = device.createCommandEncoder();
        this._encodeLinearGelu(enc, normBuf, hiddenBuf, weights, `encoder.backbone.blocks.0.mlp.fc1`, N, D, VIT_CONFIG.mlpHiddenDim);
        device.queue.submit([enc.finish()]);
        const fc1Gpu = await readBuffer(device, hiddenBuf, N * VIT_CONFIG.mlpHiddenDim * 4);
        // Compare against post-GELU reference (our shader fuses linear+GELU)
        const fc1Ref = await loadRef('block_0_fc1_post_gelu');
        compareArrays('b0_fc1_gelu', fc1Gpu, fc1Ref);

        // MLP fc2 (linear)
        enc = device.createCommandEncoder();
        this._encodeLinear(enc, hiddenBuf, ffnOutBuf, weights, `encoder.backbone.blocks.0.mlp.fc2`, N, VIT_CONFIG.mlpHiddenDim, D);
        device.queue.submit([enc.finish()]);
        const fc2Gpu = await readBuffer(device, ffnOutBuf, T * 4);
        const mlpRef = await loadRef('block_0_mlp_out');
        compareArrays('b0_mlp_out', fc2Gpu, mlpRef);

        // LayerScale2 + residual → final block 0 output
        enc = device.createCommandEncoder();
        const ffnResidualOut = (currentTokens === tokenBufA) ? tokenBufB : tokenBufA;
        this._encodeLayerScaleResidual(enc, ffnOutBuf, currentTokens, ffnResidualOut, weights, `encoder.backbone.blocks.0.ls2`, T, D);
        device.queue.submit([enc.finish()]);
        currentTokens = ffnResidualOut;

        const b0Gpu = await readBuffer(device, currentTokens, T * 4);
        const b0Ref = await loadRef('block_0_output');
        compareArrays('block_0', b0Gpu, b0Ref);
        continue;
      }

      // All other blocks: run as a batch
      const enc = device.createCommandEncoder();

      this._encodeLayerNorm(enc, currentTokens, normBuf, weights, `encoder.backbone.blocks.${l}.norm1`, N);
      this._encodeQKV(enc, normBuf, qBuf, kBuf, vBuf, weights, l, N, qkvWorkBuf);
      this._encodeAttnScores(enc, qBuf, kBuf, scoreBuf, N);
      this._encodeAttnSoftmax(enc, scoreBuf, N);
      this._encodeAttnApply(enc, scoreBuf, vBuf, attnOutBuf, N);
      this._encodeLinear(enc, attnOutBuf, projOutBuf, weights, `encoder.backbone.blocks.${l}.attn.proj`, N, D, D);

      const attnResidualOut = (currentTokens === tokenBufA) ? tokenBufB : tokenBufA;
      this._encodeLayerScaleResidual(enc, projOutBuf, currentTokens, attnResidualOut, weights, `encoder.backbone.blocks.${l}.ls1`, T, D);
      currentTokens = attnResidualOut;

      this._encodeLayerNorm(enc, currentTokens, normBuf, weights, `encoder.backbone.blocks.${l}.norm2`, N);
      this._encodeLinearGelu(enc, normBuf, hiddenBuf, weights, `encoder.backbone.blocks.${l}.mlp.fc1`, N, D, VIT_CONFIG.mlpHiddenDim);
      this._encodeLinear(enc, hiddenBuf, ffnOutBuf, weights, `encoder.backbone.blocks.${l}.mlp.fc2`, N, VIT_CONFIG.mlpHiddenDim, D);

      const ffnResidualOut = (currentTokens === tokenBufA) ? tokenBufB : tokenBufA;
      this._encodeLayerScaleResidual(enc, ffnOutBuf, currentTokens, ffnResidualOut, weights, `encoder.backbone.blocks.${l}.ls2`, T, D);
      currentTokens = ffnResidualOut;

      device.queue.submit([enc.finish()]);

      if (checkpoints.includes(l)) {
        const gpu = await readBuffer(device, currentTokens, T * 4);
        const ref = await loadRef(`block_${l}_output`);
        if (ref) {
          compareArrays(`block_${l}`, gpu, ref);
        } else {
          let gMin = Infinity, gMax = -Infinity, gSum = 0, gSqSum = 0;
          for (let i = 0; i < gpu.length; i++) {
            if (gpu[i] < gMin) gMin = gpu[i];
            if (gpu[i] > gMax) gMax = gpu[i];
            gSum += gpu[i]; gSqSum += gpu[i] * gpu[i];
          }
          const gMean = gSum / gpu.length;
          const gStd = Math.sqrt(Math.max(0, gSqSum / gpu.length - gMean * gMean));
          console.log(`  block_${l}: GPU only | [${gMin.toFixed(4)}, ${gMax.toFixed(4)}] std=${gStd.toFixed(4)} (no ref dump)`);
          results[`block_${l}`] = { label: `block_${l}`, gpu: { min: gMin, max: gMax, mean: gMean, std: gStd } };
        }
      }
    }

    normBuf.destroy(); qBuf.destroy(); kBuf.destroy(); vBuf.destroy();
    scoreBuf.destroy(); attnOutBuf.destroy(); projOutBuf.destroy();
    hiddenBuf.destroy(); ffnOutBuf.destroy();
    tokenBufA.destroy(); tokenBufB.destroy();

    console.log('\n=== COMPARISON COMPLETE ===');
    window.__backboneCompareResults = results;
    return results;
  }

  /**
   * Detailed sub-block comparison for a single transformer block.
   * Runs block l step-by-step and compares each intermediate against PyTorch.
   * Requires PyTorch sub-block dumps (block_X_norm1, block_X_attn_qkv, etc.)
   * For now, compares within WebGPU at block 0 to isolate the divergence stage.
   */
  async debugBlock0(imageBuf, weights, tokenH, tokenW) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const numPatches = tokenH * tokenW;
    const N = numPatches + 1;
    const T = N * D;
    const l = 0;

    function stats(label, arr, refArr) {
      let min = Infinity, max = -Infinity, sum = 0, sqSum = 0;
      for (let i = 0; i < arr.length; i++) {
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
        sum += arr[i]; sqSum += arr[i] * arr[i];
      }
      const mean = sum / arr.length;
      const std = Math.sqrt(sqSum / arr.length - mean * mean);
      let errStr = '';
      if (refArr) {
        let maxErr = 0, worstIdx = 0;
        for (let i = 0; i < Math.min(arr.length, refArr.length); i++) {
          const err = Math.abs(arr[i] - refArr[i]);
          if (err > maxErr) { maxErr = err; worstIdx = i; }
        }
        const row = worstIdx % D === worstIdx ? worstIdx : Math.floor(worstIdx / D);
        const col = worstIdx % D;
        errStr = ` maxErr=${maxErr.toFixed(6)}@[${row},${col}] gpu=${arr[worstIdx]?.toFixed(6)} ref=${refArr[worstIdx]?.toFixed(6)}`;
      }
      console.log(`  ${label}: [${min.toFixed(4)}, ${max.toFixed(4)}] mean=${mean.toFixed(6)} std=${std.toFixed(6)}${errStr}`);
    }

    // Initialize from patch embed
    let tokenBufA = createEmptyBuffer(device, T * 4);
    let tokenBufB = createEmptyBuffer(device, T * 4);
    {
      const enc = device.createCommandEncoder();
      this._encodePatchEmbed(enc, imageBuf, weights, tokenBufA, tokenH, tokenW);
      device.queue.submit([enc.finish()]);
    }
    let currentTokens = tokenBufA;

    const normBuf = createEmptyBuffer(device, T * 4);
    const qBuf = createEmptyBuffer(device, T * 4);
    const kBuf = createEmptyBuffer(device, T * 4);
    const vBuf = createEmptyBuffer(device, T * 4);
    const scoreBuf = createEmptyBuffer(device, VIT_CONFIG.numHeads * N * N * 4);
    const attnOutBuf = createEmptyBuffer(device, T * 4);
    const projOutBuf = createEmptyBuffer(device, T * 4);
    const hiddenBuf = createEmptyBuffer(device, N * VIT_CONFIG.mlpHiddenDim * 4);
    const ffnOutBuf = createEmptyBuffer(device, T * 4);
    const qkvWorkBuf = createEmptyBuffer(device, N * 3 * D * 4);

    console.log('\n=== BLOCK 0 SUB-STEP COMPARISON ===\n');

    // Step 1: LayerNorm1
    {
      const enc = device.createCommandEncoder();
      this._encodeLayerNorm(enc, currentTokens, normBuf, weights, `encoder.backbone.blocks.${l}.norm1`, N);
      device.queue.submit([enc.finish()]);
      const norm1 = await readBuffer(device, normBuf, T * 4);
      stats('norm1_output', norm1);
      // Check dim 538 specifically across a few tokens
      console.log(`    dim538 samples: token0=${norm1[0*D+538]?.toFixed(6)}, token276=${norm1[276*D+538]?.toFixed(6)}`);
    }

    // Step 2: QKV
    {
      const enc = device.createCommandEncoder();
      this._encodeQKV(enc, normBuf, qBuf, kBuf, vBuf, weights, l, N, qkvWorkBuf);
      device.queue.submit([enc.finish()]);
      const q = await readBuffer(device, qBuf, T * 4);
      const k = await readBuffer(device, kBuf, T * 4);
      const v = await readBuffer(device, vBuf, T * 4);
      stats('Q', q);
      stats('K', k);
      stats('V', v);
      // Check dim 538 = head 8, offset 26
      console.log(`    Q dim538: token0=${q[538]?.toFixed(6)}, token276=${q[276*D+538]?.toFixed(6)}`);
      console.log(`    K dim538: token0=${k[538]?.toFixed(6)}, token276=${k[276*D+538]?.toFixed(6)}`);
      console.log(`    V dim538: token0=${v[538]?.toFixed(6)}, token276=${v[276*D+538]?.toFixed(6)}`);
    }

    // Step 3: Attention scores (just stats, huge tensor)
    {
      const enc = device.createCommandEncoder();
      this._encodeAttnScores(enc, qBuf, kBuf, scoreBuf, N);
      device.queue.submit([enc.finish()]);
      const scores = await readBuffer(device, scoreBuf, VIT_CONFIG.numHeads * N * N * 4);
      // Just check head 8 scores for token 276
      const headBase = 8 * N * N;
      const rowBase = headBase + 276 * N;
      let sMin = Infinity, sMax = -Infinity;
      for (let j = 0; j < N; j++) {
        if (scores[rowBase + j] < sMin) sMin = scores[rowBase + j];
        if (scores[rowBase + j] > sMax) sMax = scores[rowBase + j];
      }
      console.log(`  attn_scores head8 token276: [${sMin.toFixed(4)}, ${sMax.toFixed(4)}]`);
    }

    // Step 4: Softmax
    {
      const enc = device.createCommandEncoder();
      this._encodeAttnSoftmax(enc, scoreBuf, N);
      device.queue.submit([enc.finish()]);
      const softmax = await readBuffer(device, scoreBuf, VIT_CONFIG.numHeads * N * N * 4);
      const headBase = 8 * N * N;
      const rowBase = headBase + 276 * N;
      let sSum = 0, sMax = -Infinity;
      for (let j = 0; j < N; j++) {
        sSum += softmax[rowBase + j];
        if (softmax[rowBase + j] > sMax) sMax = softmax[rowBase + j];
      }
      console.log(`  softmax head8 token276: sum=${sSum.toFixed(6)}, max=${sMax.toFixed(6)}`);
    }

    // Step 5: Apply attention
    {
      const enc = device.createCommandEncoder();
      this._encodeAttnApply(enc, scoreBuf, vBuf, attnOutBuf, N);
      device.queue.submit([enc.finish()]);
      const attnOut = await readBuffer(device, attnOutBuf, T * 4);
      stats('attn_apply_output', attnOut);
      console.log(`    attn_out dim538: token276=${attnOut[276*D+538]?.toFixed(6)}`);
    }

    // Step 6: Output projection
    {
      const enc = device.createCommandEncoder();
      this._encodeLinear(enc, attnOutBuf, projOutBuf, weights, `encoder.backbone.blocks.${l}.attn.proj`, N, D, D);
      device.queue.submit([enc.finish()]);
      const proj = await readBuffer(device, projOutBuf, T * 4);
      stats('proj_output', proj);
      console.log(`    proj dim538: token276=${proj[276*D+538]?.toFixed(6)}`);
    }

    // Step 7: LayerScale1 + residual
    {
      const enc = device.createCommandEncoder();
      const outBuf = (currentTokens === tokenBufA) ? tokenBufB : tokenBufA;
      this._encodeLayerScaleResidual(enc, projOutBuf, currentTokens, outBuf, weights, `encoder.backbone.blocks.${l}.ls1`, T, D);
      device.queue.submit([enc.finish()]);
      const ls1 = await readBuffer(device, outBuf, T * 4);
      stats('after_ls1_residual', ls1);
      console.log(`    ls1_res dim538: token276=${ls1[276*D+538]?.toFixed(6)}`);
      currentTokens = outBuf;
    }

    // Step 8: LayerNorm2
    {
      const enc = device.createCommandEncoder();
      this._encodeLayerNorm(enc, currentTokens, normBuf, weights, `encoder.backbone.blocks.${l}.norm2`, N);
      device.queue.submit([enc.finish()]);
      const norm2 = await readBuffer(device, normBuf, T * 4);
      stats('norm2_output', norm2);
    }

    // Step 9: MLP fc1 (linear + GELU)
    {
      const enc = device.createCommandEncoder();
      this._encodeLinearGelu(enc, normBuf, hiddenBuf, weights, `encoder.backbone.blocks.${l}.mlp.fc1`, N, D, VIT_CONFIG.mlpHiddenDim);
      device.queue.submit([enc.finish()]);
      const fc1 = await readBuffer(device, hiddenBuf, N * VIT_CONFIG.mlpHiddenDim * 4);
      let fMin = Infinity, fMax = -Infinity, fSum = 0, fSqSum = 0;
      for (let i = 0; i < fc1.length; i++) {
        if (fc1[i] < fMin) fMin = fc1[i];
        if (fc1[i] > fMax) fMax = fc1[i];
        fSum += fc1[i]; fSqSum += fc1[i] * fc1[i];
      }
      const fMean = fSum / fc1.length, fStd = Math.sqrt(fSqSum / fc1.length - fMean * fMean);
      console.log(`  fc1_output (GELU): [${fMin.toFixed(4)}, ${fMax.toFixed(4)}] mean=${fMean.toFixed(6)} std=${fStd.toFixed(6)}`);
    }

    // Step 10: MLP fc2 (linear)
    {
      const enc = device.createCommandEncoder();
      this._encodeLinear(enc, hiddenBuf, ffnOutBuf, weights, `encoder.backbone.blocks.${l}.mlp.fc2`, N, VIT_CONFIG.mlpHiddenDim, D);
      device.queue.submit([enc.finish()]);
      const fc2 = await readBuffer(device, ffnOutBuf, T * 4);
      stats('fc2_output', fc2);
      console.log(`    fc2 dim538: token276=${fc2[276*D+538]?.toFixed(6)}`);
    }

    // Step 11: LayerScale2 + residual → final block 0 output
    {
      const enc = device.createCommandEncoder();
      const outBuf = (currentTokens === tokenBufA) ? tokenBufB : tokenBufA;
      this._encodeLayerScaleResidual(enc, ffnOutBuf, currentTokens, outBuf, weights, `encoder.backbone.blocks.${l}.ls2`, T, D);
      device.queue.submit([enc.finish()]);
      const final = await readBuffer(device, outBuf, T * 4);
      stats('block0_final', final);
      console.log(`    final dim538: token276=${final[276*D+538]?.toFixed(6)}`);
    }

    // Load reference for comparison
    const refResp = await fetch('/layer_dumps/block_0_output.bin');
    const ref = new Float32Array(await refResp.arrayBuffer());
    console.log(`\n  Reference block0 dim538 token276: ${ref[276*D+538]?.toFixed(6)}`);

    // Clean up
    normBuf.destroy(); qBuf.destroy(); kBuf.destroy(); vBuf.destroy();
    scoreBuf.destroy(); attnOutBuf.destroy(); projOutBuf.destroy();
    hiddenBuf.destroy(); ffnOutBuf.destroy();
    tokenBufA.destroy(); tokenBufB.destroy();

    console.log('\n=== BLOCK 0 ANALYSIS COMPLETE ===\n');
  }

  _encodeAdd(enc, dst, src, count) {
    // Simple element-wise add using the activation shader
    // We reuse layerScale with gamma=1 and dst as residual... or just inline
    // For simplicity, use a copy + add pattern
    const device = this.device;
    const totalWG = ceilDiv(count, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsBuf = this._cachedUniform(new Uint32Array([count, wgX]));

    const bg = this._cachedBindGroup('add', this.pipelines.add.getBindGroupLayout(0), [
      { binding: 0, resource: { buffer: dst } },
      { binding: 1, resource: { buffer: src } },
      { binding: 2, resource: { buffer: paramsBuf } },
    ]);

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.add);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }
}

DINOv2Backbone._bgIdCounter = 0;

export { VIT_CONFIG };
