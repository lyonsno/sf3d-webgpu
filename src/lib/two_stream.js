/**
 * two_stream.js — TwoStreamInterleaveTransformer dispatch for SF3D.
 *
 * Architecture (from backbone.py):
 *   Input:
 *     - triplane tokens [3*96*96, 1024] from tokenizer (GroupNorm → proj)
 *     - image tokens [N_img, 1024] from DINOv2 (LayerNorm → proj)
 *     - latent_init [1792, 1024] (learned, LayerNorm → proj)
 *     - latent = concat(image_tokens, latent_init) [N_img+1792, 1024]
 *
 *   4 TwoStreamBlocks, each:
 *     - fuse_block_in: fuse(latent ← triplane) via cross-attention + GEGLU FFN
 *     - 3 BasicBlocks: self-attention + cross-attention(latent ← encoder) + GEGLU FFN
 *     - fuse_block_out: fuse(triplane ← latent) via cross-attention + GEGLU FFN
 *
 *   Output: proj_out(triplane_tokens) + residual → [3*96*96, 1024]
 *
 * All attention uses separate Q/K/V (not fused QKV).
 * FFN uses GEGLU: linear → chunk → gate*GELU(hidden) → linear.
 */

import { createStorageBuffer, createEmptyBuffer } from './gpu.js';

import linearWGSL from '../shaders/linear.wgsl?raw';
import layerNormWGSL from '../shaders/layernorm_vit.wgsl?raw';
import crossAttentionWGSL from '../shaders/cross_attention.wgsl?raw';
import gegluWGSL from '../shaders/geglu.wgsl?raw';
import groupnormWGSL from '../shaders/groupnorm.wgsl?raw';

const WG_SIZE = 256;
const MAX_WG = 65535;
function splitWG(total) {
  if (total <= MAX_WG) return [total, 1];
  return [MAX_WG, Math.ceil(total / MAX_WG)];
}
function ceilDiv(a, b) { return Math.ceil(a / b); }

const CONFIG = {
  dim: 1024,         // latent/triplane dim
  numHeads: 16,
  headDim: 64,
  numBlocks: 4,
  numBasicBlocks: 3,
  numLatents: 1792,
  planeSize: 96,
  triplaneTokens: 3 * 96 * 96, // 27648
  gegluInnerDim: 4096,  // GEGLU inner = dim * mult = 1024 * 4 = 4096
  eps: 1e-5,
};

export class TwoStreamBackbone {
  constructor(device) {
    this.device = device;
    this.pipelines = {};
    this._uniformCache = new Map();
  }

  init() {
    const device = this.device;
    const make = (code, entry) => device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: entry },
    });

    this.pipelines.linear = make(linearWGSL, 'main');
    this.pipelines.layerNorm = make(layerNormWGSL, 'main');
    // Cross-attention pipelines share an explicit layout so all 6 bindings are available
    // to all three entry points (auto-layout would omit unused bindings per entry point)
    const crossAttnLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const crossAttnPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [crossAttnLayout] });
    this._crossAttnLayout = crossAttnLayout;

    const crossAttnModule = device.createShaderModule({ code: crossAttentionWGSL });
    const makeCA = (entry) => device.createComputePipeline({
      layout: crossAttnPipelineLayout,
      compute: { module: crossAttnModule, entryPoint: entry },
    });
    this.pipelines.crossAttnScores = makeCA('computeCrossScores');
    this.pipelines.crossAttnSoftmax = makeCA('softmaxCross');
    this.pipelines.crossAttnApply = makeCA('applyCrossAttn');
    this.pipelines.geglu = make(gegluWGSL, 'geglu_main');

    // GroupNorm pipelines — explicit shared layout for both entry points
    const gnLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const gnPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [gnLayout] });
    this._gnLayout = gnLayout;

    const gnModule = device.createShaderModule({ code: groupnormWGSL });
    this.pipelines.gnStats = device.createComputePipeline({
      layout: gnPipelineLayout,
      compute: { module: gnModule, entryPoint: 'groupnorm_stats' },
    });
    this.pipelines.gnNorm = device.createComputePipeline({
      layout: gnPipelineLayout,
      compute: { module: gnModule, entryPoint: 'groupnorm_normalize' },
    });

    // Element-wise add
    this.pipelines.add = make(`
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
    `, 'main');

    // Concat two buffers
    this.pipelines.concat = make(`
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> out: array<f32>;
      struct P { sizeA: u32, sizeB: u32, numWgX: u32 }
      @group(0) @binding(3) var<uniform> p: P;
      @compute @workgroup_size(256)
      fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
        let idx = (wgid.x + wgid.y * p.numWgX) * 256u + lid.x;
        let total = p.sizeA + p.sizeB;
        if (idx >= total) { return; }
        if (idx < p.sizeA) { out[idx] = a[idx]; }
        else { out[idx] = b[idx - p.sizeA]; }
      }
    `, 'main');
  }

  _cachedUniform(data) {
    const bytes = new Uint8Array(data.buffer || data);
    let h = 0;
    for (let i = 0; i < bytes.length; i++) h = (h * 31 + bytes[i]) | 0;
    const key = `u_${bytes.length}_${h}`;
    if (this._uniformCache.has(key)) return this._uniformCache.get(key);
    const buf = this.device.createBuffer({
      size: Math.max(bytes.byteLength, 16),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange()).set(bytes);
    buf.unmap();
    this._uniformCache.set(key, buf);
    return buf;
  }

  /**
   * Run the two-stream interleave transformer.
   *
   * @param {GPUCommandEncoder} encoder
   * @param {GPUBuffer} imageTokensBuf - [N_img, 1024] from DINOv2
   * @param {GPUBuffer} triplaneEmbBuf - [3, 1024, 96, 96] triplane embeddings
   * @param {Object} weights - backbone weights from loadWeights
   * @param {number} N_img - number of image tokens
   * @returns {GPUBuffer} - [3*96*96, 1024] refined triplane tokens
   */
  forward(encoder, imageTokensBuf, N_img, weights) {
    const device = this.device;
    const D = CONFIG.dim;
    const N_tri = CONFIG.triplaneTokens; // 27648
    const N_latent_init = CONFIG.numLatents; // 1792

    // 1. Prepare triplane tokens: GroupNorm → permute → proj
    //    Input: tokenizer.embeddings [3, 1024, 96, 96] in CHW
    //    Permute to [3*96*96, 1024] (reshape)
    //    GroupNorm operates on [1024, 3*96*96] (channel-first)
    //    Then permute to [N_tri, 1024] and project
    const gnInputBuf = createEmptyBuffer(device, D * N_tri * 4); // [1024, 27648] for GN
    // The tokenizer embeddings are already [3, 1024, 96, 96] = [3*1024*96*96]
    // GroupNorm expects [C, H*W] where C=1024, spatial=3*96*96=27648
    // After GN, permute to [27648, 1024] then project

    // For now, treat the triplane embeddings as [1024, 27648] for GroupNorm
    // (the data layout is [3, 1024, 96, 96] which when viewed as [1024, 27648_spatial] per plane
    //  is actually interleaved... but GroupNorm with 32 groups over 1024 channels is channel-wise)
    //
    // Actually the backbone code does:
    //   hidden_states = self.norm_triplane(hidden_states)  # GroupNorm on [B, C, N]
    //   hidden_states = hidden_states.permute(0, 2, 1)    # [B, N, C]
    //   hidden_states = self.proj_triplane(hidden_states)  # Linear [C, C]
    //
    // So the triplane tokens come in as [C, N] = [1024, 27648] from tokenizer,
    // get GroupNormed, then permuted to [N, C] = [27648, 1024], then projected.

    // GroupNorm on tokenizer embeddings [1024, 27648]
    const gnOutBuf = this._dispatchGroupNorm(encoder, weights.tokenizer_embeddings_buf,
      weights.normTriplane, D, N_tri, 32);

    // Permute [1024, 27648] → [27648, 1024]
    const triPermBuf = createEmptyBuffer(device, N_tri * D * 4);
    this._dispatchTranspose(encoder, gnOutBuf, triPermBuf, D, N_tri);

    // Project triplane: [27648, 1024] → [27648, 1024]
    const triProjBuf = createEmptyBuffer(device, N_tri * D * 4);
    this._dispatchLinear(encoder, triPermBuf, triProjBuf,
      weights.projTriplane.weight, weights.projTriplane.bias, N_tri, D, D);

    // 2. Prepare image latents: LayerNorm → proj → concat with latent_init
    const imgNormBuf = createEmptyBuffer(device, N_img * D * 4);
    this._dispatchLayerNorm(encoder, imageTokensBuf, imgNormBuf, weights.normImage, N_img, D);

    const imgProjBuf = createEmptyBuffer(device, N_img * D * 4);
    this._dispatchLinear(encoder, imgNormBuf, imgProjBuf,
      weights.projImage.weight, weights.projImage.bias, N_img, D, D);

    // Prepare latent_init: LayerNorm → proj
    const latentNormBuf = createEmptyBuffer(device, N_latent_init * D * 4);
    this._dispatchLayerNorm(encoder, weights.latentInit, latentNormBuf,
      weights.normLatent, N_latent_init, D);

    const latentProjBuf = createEmptyBuffer(device, N_latent_init * D * 4);
    this._dispatchLinear(encoder, latentNormBuf, latentProjBuf,
      weights.projLatent.weight, weights.projLatent.bias, N_latent_init, D, D);

    // Concat: latent = [image_tokens, latent_init] → [N_img + 1792, 1024]
    const N_latent = N_img + N_latent_init;
    const latentBuf = createEmptyBuffer(device, N_latent * D * 4);
    this._dispatchConcat(encoder, imgProjBuf, latentProjBuf, latentBuf,
      N_img * D, N_latent_init * D);

    // 3. Run 4 TwoStreamBlocks
    let currentLatent = latentBuf;
    let currentTriplane = triProjBuf;

    // Track intermediate buffers for diagnostics
    this._diagnosticBuffers = {};

    for (let b = 0; b < CONFIG.numBlocks; b++) {
      const block = weights.mainBlocks[b];

      // fuse_block_in: fuse(latent ← triplane)
      currentLatent = this._dispatchFuseBlock(encoder, currentLatent, currentTriplane,
        block.fuseBlockIn, N_latent, N_tri, D);

      // 3 BasicBlocks: self-attn + cross-attn(latent ← encoder_hidden_states) + GEGLU FFN
      for (let i = 0; i < CONFIG.numBasicBlocks; i++) {
        currentLatent = this._dispatchBasicBlock(encoder, currentLatent,
          imageTokensBuf, block.transformerBlocks[i], N_latent, N_img, D);
      }

      // fuse_block_out: fuse(triplane ← latent)
      currentTriplane = this._dispatchFuseBlock(encoder, currentTriplane, currentLatent,
        block.fuseBlockOut, N_tri, N_latent, D);

      // Save refs for diagnostics
      this._diagnosticBuffers[`block${b}_latent`] = currentLatent;
      this._diagnosticBuffers[`block${b}_triplane`] = currentTriplane;

      // Also save the triplane projection and GroupNorm output from first block
      if (b === 0) {
        this._diagnosticBuffers['gnOutBuf'] = gnOutBuf;
        this._diagnosticBuffers['triPermBuf'] = triPermBuf;
        this._diagnosticBuffers['triProjBuf'] = triProjBuf;
        this._diagnosticBuffers['imgProjBuf'] = imgProjBuf;
        this._diagnosticBuffers['latentProjBuf'] = latentProjBuf;
        this._diagnosticBuffers['latentBuf'] = latentBuf;
      }
    }

    // 4. Project out and add residual
    const projOutBuf = createEmptyBuffer(device, N_tri * D * 4);
    this._dispatchLinear(encoder, currentTriplane, projOutBuf,
      weights.projOut.weight, weights.projOut.bias, N_tri, D, D);

    // Permute back: [27648, 1024] → [1024, 27648] for residual add
    const projOutPermBuf = createEmptyBuffer(device, D * N_tri * 4);
    this._dispatchTranspose(encoder, projOutBuf, projOutPermBuf, N_tri, D);

    // Add residual (original triplane embeddings)
    this._dispatchAdd(encoder, projOutPermBuf, weights.tokenizer_embeddings_buf, D * N_tri);

    // Result: [1024, 27648] = [1024, 3*96*96] triplane features
    return { buffer: projOutPermBuf, C: D, N: N_tri, planeSize: CONFIG.planeSize };
  }

  // --- FuseBlock: cross-attention fuse(z ← x) + GEGLU FFN ---
  _dispatchFuseBlock(encoder, zBuf, xBuf, weights, N_z, N_x, D) {
    const device = this.device;

    // norm_z1
    const zNormBuf = createEmptyBuffer(device, N_z * D * 4);
    this._dispatchLayerNorm(encoder, zBuf, zNormBuf, weights.normZ1, N_z, D);

    // norm_x (if present)
    let xNormBuf = xBuf;
    if (weights.normX) {
      xNormBuf = createEmptyBuffer(device, N_x * D * 4);
      this._dispatchLayerNorm(encoder, xBuf, xNormBuf, weights.normX, N_x, D);
    }

    // Cross-attention: Q from z, KV from x
    const attnOutBuf = this._dispatchCrossAttention(encoder, zNormBuf, xNormBuf,
      weights.attn, N_z, N_x, D);

    // z = z + attn_out
    const z1Buf = createEmptyBuffer(device, N_z * D * 4);
    encoder.copyBufferToBuffer(zBuf, 0, z1Buf, 0, N_z * D * 4);
    this._dispatchAdd(encoder, z1Buf, attnOutBuf, N_z * D);

    // norm_z2 + GEGLU FFN
    const z2NormBuf = createEmptyBuffer(device, N_z * D * 4);
    this._dispatchLayerNorm(encoder, z1Buf, z2NormBuf, weights.normZ2, N_z, D);

    const ffnOutBuf = this._dispatchGEGLUFFN(encoder, z2NormBuf, weights.ff, N_z, D);

    // z = z1 + ffn_out
    const zOutBuf = createEmptyBuffer(device, N_z * D * 4);
    encoder.copyBufferToBuffer(z1Buf, 0, zOutBuf, 0, N_z * D * 4);
    this._dispatchAdd(encoder, zOutBuf, ffnOutBuf, N_z * D);

    return zOutBuf;
  }

  // --- BasicBlock: self-attn + cross-attn + GEGLU FFN ---
  _dispatchBasicBlock(encoder, zBuf, xBuf, weights, N_z, N_x, D) {
    const device = this.device;

    // 1. Self-attention: norm1 → attn1(z, z)
    const norm1Buf = createEmptyBuffer(device, N_z * D * 4);
    this._dispatchLayerNorm(encoder, zBuf, norm1Buf, weights.norm1, N_z, D);

    const selfAttnBuf = this._dispatchSelfAttention(encoder, norm1Buf,
      weights.attn1, N_z, D);

    const z1Buf = createEmptyBuffer(device, N_z * D * 4);
    encoder.copyBufferToBuffer(zBuf, 0, z1Buf, 0, N_z * D * 4);
    this._dispatchAdd(encoder, z1Buf, selfAttnBuf, N_z * D);

    // 2. Cross-attention: norm2 → attn2(z, x)
    const norm2Buf = createEmptyBuffer(device, N_z * D * 4);
    this._dispatchLayerNorm(encoder, z1Buf, norm2Buf, weights.norm2, N_z, D);

    const crossAttnBuf = this._dispatchCrossAttention(encoder, norm2Buf,
      xBuf, weights.attn2, N_z, N_x, D);

    const z2Buf = createEmptyBuffer(device, N_z * D * 4);
    encoder.copyBufferToBuffer(z1Buf, 0, z2Buf, 0, N_z * D * 4);
    this._dispatchAdd(encoder, z2Buf, crossAttnBuf, N_z * D);

    // 3. GEGLU FFN: norm3 → geglu_ffn
    const norm3Buf = createEmptyBuffer(device, N_z * D * 4);
    this._dispatchLayerNorm(encoder, z2Buf, norm3Buf, weights.norm3, N_z, D);

    const ffnOutBuf = this._dispatchGEGLUFFN(encoder, norm3Buf, weights.ff, N_z, D);

    const zOutBuf = createEmptyBuffer(device, N_z * D * 4);
    encoder.copyBufferToBuffer(z2Buf, 0, zOutBuf, 0, N_z * D * 4);
    this._dispatchAdd(encoder, zOutBuf, ffnOutBuf, N_z * D);

    return zOutBuf;
  }

  // --- Cross-attention dispatch (tiled over Q to fit WebGPU buffer limits) ---
  _dispatchCrossAttention(encoder, qInputBuf, kvInputBuf, attnWeights, N_q, N_kv, D) {
    const device = this.device;
    const numHeads = CONFIG.numHeads;
    const headDim = CONFIG.headDim;

    // Project Q, K, V
    const qBuf = createEmptyBuffer(device, N_q * D * 4);
    const kBuf = createEmptyBuffer(device, N_kv * D * 4);
    const vBuf = createEmptyBuffer(device, N_kv * D * 4);

    this._dispatchLinearNoBias(encoder, qInputBuf, qBuf, attnWeights.wq, N_q, D, D);
    this._dispatchLinearNoBias(encoder, kvInputBuf, kBuf, attnWeights.wk, N_kv, D, D);
    this._dispatchLinearNoBias(encoder, kvInputBuf, vBuf, attnWeights.wv, N_kv, D, D);

    // Tile Q to keep score buffer under WebGPU maxBufferSize (~256MB safe target).
    // Score buffer per tile = numHeads × tileQ × N_kv × 4 bytes.
    // With TILE_Q=256, worst case (N_kv=27648): 16 × 256 × 27648 × 4 = ~440MB.
    // Use 128 for extra safety margin: 16 × 128 × 27648 × 4 = ~220MB.
    const TILE_Q = 128;
    const scoreBufSize = numHeads * TILE_Q * N_kv * 4;
    const scoreBuf = createEmptyBuffer(device, scoreBufSize);
    const tileAttnOutBuf = createEmptyBuffer(device, TILE_Q * D * 4);
    const attnOutBuf = createEmptyBuffer(device, N_q * D * 4);

    for (let qStart = 0; qStart < N_q; qStart += TILE_Q) {
      const tileQ = Math.min(TILE_Q, N_q - qStart);
      const qOffsetBytes = qStart * D * 4;

      // Scores: pass tileQ as N_q to the shader, offset Q buffer
      {
        const total = numHeads * tileQ * N_kv;
        const totalWG = ceilDiv(total, WG_SIZE);
        const [wgX, wgY] = splitWG(totalWG);
        const params = this._cachedUniform(new Uint32Array([tileQ, N_kv, headDim, numHeads, wgX]));
        const bg = device.createBindGroup({
          layout: this._crossAttnLayout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: qBuf, offset: qOffsetBytes, size: tileQ * D * 4 } },
            { binding: 2, resource: { buffer: kBuf } },
            { binding: 3, resource: { buffer: vBuf } },
            { binding: 4, resource: { buffer: scoreBuf, size: numHeads * tileQ * N_kv * 4 } },
            { binding: 5, resource: { buffer: tileAttnOutBuf, size: tileQ * D * 4 } },
          ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.crossAttnScores);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(wgX, wgY);
        pass.end();
      }

      // Softmax
      {
        const totalRows = numHeads * tileQ;
        const totalWG = ceilDiv(totalRows, WG_SIZE);
        const [wgX, wgY] = splitWG(totalWG);
        const params = this._cachedUniform(new Uint32Array([tileQ, N_kv, headDim, numHeads, wgX]));
        const bg = device.createBindGroup({
          layout: this._crossAttnLayout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: qBuf, offset: qOffsetBytes, size: tileQ * D * 4 } },
            { binding: 2, resource: { buffer: kBuf } },
            { binding: 3, resource: { buffer: vBuf } },
            { binding: 4, resource: { buffer: scoreBuf, size: numHeads * tileQ * N_kv * 4 } },
            { binding: 5, resource: { buffer: tileAttnOutBuf, size: tileQ * D * 4 } },
          ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.crossAttnSoftmax);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(wgX, wgY);
        pass.end();
      }

      // Apply attention for this tile
      {
        const total = tileQ * numHeads * headDim;
        const totalWG = ceilDiv(total, WG_SIZE);
        const [wgX, wgY] = splitWG(totalWG);
        const params = this._cachedUniform(new Uint32Array([tileQ, N_kv, headDim, numHeads, wgX]));
        const bg = device.createBindGroup({
          layout: this._crossAttnLayout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: qBuf, offset: qOffsetBytes, size: tileQ * D * 4 } },
            { binding: 2, resource: { buffer: kBuf } },
            { binding: 3, resource: { buffer: vBuf } },
            { binding: 4, resource: { buffer: scoreBuf, size: numHeads * tileQ * N_kv * 4 } },
            { binding: 5, resource: { buffer: tileAttnOutBuf, size: tileQ * D * 4 } },
          ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.crossAttnApply);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(wgX, wgY);
        pass.end();
      }

      // Copy tile output to the right offset in the full output buffer
      encoder.copyBufferToBuffer(tileAttnOutBuf, 0, attnOutBuf, qOffsetBytes, tileQ * D * 4);
    }

    // Output projection
    const projOutBuf = createEmptyBuffer(device, N_q * D * 4);
    this._dispatchLinear(encoder, attnOutBuf, projOutBuf,
      attnWeights.proj.weight, attnWeights.proj.bias, N_q, D, D);

    return projOutBuf;
  }

  // --- Self-attention dispatch (tiled over Q to fit WebGPU buffer limits) ---
  _dispatchSelfAttention(encoder, inputBuf, attnWeights, N, D) {
    const device = this.device;
    const numHeads = CONFIG.numHeads;
    const headDim = CONFIG.headDim;
    const scale = 1.0 / Math.sqrt(headDim);

    const qBuf = createEmptyBuffer(device, N * D * 4);
    const kBuf = createEmptyBuffer(device, N * D * 4);
    const vBuf = createEmptyBuffer(device, N * D * 4);

    this._dispatchLinearNoBias(encoder, inputBuf, qBuf, attnWeights.wq, N, D, D);
    this._dispatchLinearNoBias(encoder, inputBuf, kBuf, attnWeights.wk, N, D, D);
    this._dispatchLinearNoBias(encoder, inputBuf, vBuf, attnWeights.wv, N, D, D);

    // Self-attention score buffer = numHeads × N × N × 4 bytes.
    // For N=3089: ~612MB, may exceed WebGPU maxBufferSize on some GPUs.
    // Tile over Q rows to stay within limits.
    const TILE_Q = 128;
    const scoreTileSize = numHeads * TILE_Q * N * 4;
    const scoreBuf = createEmptyBuffer(device, scoreTileSize);
    const attnOutBuf = createEmptyBuffer(device, N * D * 4);

    for (let qStart = 0; qStart < N; qStart += TILE_Q) {
      const tileQ = Math.min(TILE_Q, N - qStart);
      const qOffsetBytes = qStart * D * 4;

      // The self-attention shader uses a different param layout than cross-attention.
      // It expects: [N, D, numHeads, headDim, scale, numWgX] where N is the full
      // sequence length for K indexing. We need to use cross-attention shaders instead
      // for tiled self-attention, since they separate N_q and N_kv.

      // Scores
      {
        const total = numHeads * tileQ * N;
        const totalWG = ceilDiv(total, WG_SIZE);
        const [wgX, wgY] = splitWG(totalWG);
        const params = this._cachedUniform(new Uint32Array([tileQ, N, headDim, numHeads, wgX]));
        const bg = device.createBindGroup({
          layout: this._crossAttnLayout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: qBuf, offset: qOffsetBytes, size: tileQ * D * 4 } },
            { binding: 2, resource: { buffer: kBuf } },
            { binding: 3, resource: { buffer: vBuf } },
            { binding: 4, resource: { buffer: scoreBuf, size: numHeads * tileQ * N * 4 } },
            { binding: 5, resource: { buffer: attnOutBuf, offset: qOffsetBytes, size: tileQ * D * 4 } },
          ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.crossAttnScores);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(wgX, wgY);
        pass.end();
      }

      // Softmax
      {
        const totalRows = numHeads * tileQ;
        const totalWG = ceilDiv(totalRows, WG_SIZE);
        const [wgX, wgY] = splitWG(totalWG);
        const params = this._cachedUniform(new Uint32Array([tileQ, N, headDim, numHeads, wgX]));
        const bg = device.createBindGroup({
          layout: this._crossAttnLayout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: qBuf, offset: qOffsetBytes, size: tileQ * D * 4 } },
            { binding: 2, resource: { buffer: kBuf } },
            { binding: 3, resource: { buffer: vBuf } },
            { binding: 4, resource: { buffer: scoreBuf, size: numHeads * tileQ * N * 4 } },
            { binding: 5, resource: { buffer: attnOutBuf, offset: qOffsetBytes, size: tileQ * D * 4 } },
          ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.crossAttnSoftmax);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(wgX, wgY);
        pass.end();
      }

      // Apply
      {
        const total = tileQ * numHeads * headDim;
        const totalWG = ceilDiv(total, WG_SIZE);
        const [wgX, wgY] = splitWG(totalWG);
        const params = this._cachedUniform(new Uint32Array([tileQ, N, headDim, numHeads, wgX]));
        const bg = device.createBindGroup({
          layout: this._crossAttnLayout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: qBuf, offset: qOffsetBytes, size: tileQ * D * 4 } },
            { binding: 2, resource: { buffer: kBuf } },
            { binding: 3, resource: { buffer: vBuf } },
            { binding: 4, resource: { buffer: scoreBuf, size: numHeads * tileQ * N * 4 } },
            { binding: 5, resource: { buffer: attnOutBuf, offset: qOffsetBytes, size: tileQ * D * 4 } },
          ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.crossAttnApply);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(wgX, wgY);
        pass.end();
      }
    }

    // Output projection
    const projOutBuf = createEmptyBuffer(device, N * D * 4);
    this._dispatchLinear(encoder, attnOutBuf, projOutBuf,
      attnWeights.proj.weight, attnWeights.proj.bias, N, D, D);

    return projOutBuf;
  }

  // --- GEGLU FFN ---
  _dispatchGEGLUFFN(encoder, inputBuf, ffWeights, N, D) {
    const device = this.device;
    const innerDim = CONFIG.gegluInnerDim;

    // Linear: [N, D] → [N, 2*innerDim] (GEGLU projection)
    const geGluProjBuf = createEmptyBuffer(device, N * 2 * innerDim * 4);
    this._dispatchLinear(encoder, inputBuf, geGluProjBuf,
      ffWeights.geglu.weight, ffWeights.geglu.bias, N, D, 2 * innerDim);

    // GEGLU activation: [N, 2*innerDim] → [N, innerDim]
    const geGluOutBuf = createEmptyBuffer(device, N * innerDim * 4);
    {
      const totalWG = ceilDiv(N * innerDim, WG_SIZE);
      const [wgX, wgY] = splitWG(totalWG);
      const params = this._cachedUniform(new Uint32Array([N, innerDim, wgX]));
      const bg = device.createBindGroup({
        layout: this.pipelines.geglu.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: geGluProjBuf } },
          { binding: 2, resource: { buffer: geGluOutBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.geglu);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    }

    // Linear: [N, innerDim] → [N, D]
    const ffnOutBuf = createEmptyBuffer(device, N * D * 4);
    this._dispatchLinear(encoder, geGluOutBuf, ffnOutBuf,
      ffWeights.proj.weight, ffWeights.proj.bias, N, innerDim, D);

    return ffnOutBuf;
  }

  // --- Low-level dispatch helpers ---

  _dispatchLinear(encoder, input, output, weight, bias, rows, inDim, outDim) {
    const totalWG = ceilDiv(rows * outDim, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);
    const params = this._cachedUniform(new Uint32Array([rows, inDim, outDim, wgX]));
    const bg = this.device.createBindGroup({
      layout: this.pipelines.linear.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: weight } },
        { binding: 3, resource: { buffer: bias } },
        { binding: 4, resource: { buffer: output } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.linear);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _dispatchLinearNoBias(encoder, input, output, weight, rows, inDim, outDim) {
    // Use a zero bias buffer, reallocating if a larger outDim is needed
    if (!this._zeroBias || this._zeroBiasSize < outDim) {
      this._zeroBiasSize = outDim;
      this._zeroBias = createStorageBuffer(this.device, new Float32Array(outDim));
    }
    this._dispatchLinear(encoder, input, output, weight, this._zeroBias, rows, inDim, outDim);
  }

  _dispatchLayerNorm(encoder, input, output, norm, N, D) {
    const paramsData = new ArrayBuffer(16);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true); v.setUint32(4, D, true); v.setFloat32(8, CONFIG.eps, true);
    const params = this._cachedUniform(new Uint8Array(paramsData));
    const bg = this.device.createBindGroup({
      layout: this.pipelines.layerNorm.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: norm.weight } },
        { binding: 3, resource: { buffer: norm.bias } },
        { binding: 4, resource: { buffer: output } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.layerNorm);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(N);
    pass.end();
  }

  _dispatchGroupNorm(encoder, input, norm, C, spatialSize, numGroups) {
    const device = this.device;
    const total = C * spatialSize;

    const normTotalWG = ceilDiv(total, WG_SIZE);
    const [normWgX, normWgY] = splitWG(normTotalWG);
    const uniformArr = new ArrayBuffer(24);
    const u32View = new Uint32Array(uniformArr);
    const f32View = new Float32Array(uniformArr);
    u32View[0] = C; u32View[1] = 1; u32View[2] = spatialSize; u32View[3] = numGroups;
    f32View[4] = CONFIG.eps; u32View[5] = normWgX;
    const uniformBuf = this._cachedUniform(new Uint8Array(uniformArr));

    const statsBuf = createEmptyBuffer(device, numGroups * 2 * 4);
    const outputBuf = createEmptyBuffer(device, total * 4);

    // Stats pass — all bindings via shared explicit layout
    const statsBG = device.createBindGroup({
      layout: this._gnLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: norm.weight } },
        { binding: 3, resource: { buffer: norm.bias } },
        { binding: 4, resource: { buffer: outputBuf } },
        { binding: 5, resource: { buffer: statsBuf } },
      ],
    });
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.gnStats);
    pass.setBindGroup(0, statsBG);
    pass.dispatchWorkgroups(ceilDiv(numGroups, WG_SIZE));
    pass.end();

    // Normalize pass — all bindings via shared explicit layout
    const normBG = device.createBindGroup({
      layout: this._gnLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: norm.weight } },
        { binding: 3, resource: { buffer: norm.bias } },
        { binding: 4, resource: { buffer: outputBuf } },
        { binding: 5, resource: { buffer: statsBuf } },
      ],
    });
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.gnNorm);
    pass.setBindGroup(0, normBG);
    pass.dispatchWorkgroups(normWgX, normWgY);
    pass.end();

    return outputBuf;
  }

  _dispatchTranspose(encoder, input, output, rows, cols) {
    // Simple transpose: [rows, cols] → [cols, rows]
    // Use inline shader since transpose_nd.wgsl may have different binding layout
    if (!this.pipelines.transpose2d) {
      this.pipelines.transpose2d = this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: this.device.createShaderModule({
            code: `
              struct P { rows: u32, cols: u32, numWgX: u32 }
              @group(0) @binding(0) var<uniform> p: P;
              @group(0) @binding(1) var<storage, read> input: array<f32>;
              @group(0) @binding(2) var<storage, read_write> output: array<f32>;
              @compute @workgroup_size(256)
              fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
                let idx = (wgid.x + wgid.y * p.numWgX) * 256u + lid.x;
                if (idx >= p.rows * p.cols) { return; }
                let r = idx / p.cols;
                let c = idx % p.cols;
                output[c * p.rows + r] = input[r * p.cols + c];
              }
            `,
          }),
          entryPoint: 'main',
        },
      });
    }

    const total = rows * cols;
    const totalWG = ceilDiv(total, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);
    const params = this._cachedUniform(new Uint32Array([rows, cols, wgX]));

    const bg = this.device.createBindGroup({
      layout: this.pipelines.transpose2d.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: output } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.transpose2d);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _dispatchAdd(encoder, dst, src, count) {
    const totalWG = ceilDiv(count, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);
    const params = this._cachedUniform(new Uint32Array([count, wgX]));
    const bg = this.device.createBindGroup({
      layout: this.pipelines.add.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dst } },
        { binding: 1, resource: { buffer: src } },
        { binding: 2, resource: { buffer: params } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.add);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _dispatchConcat(encoder, aBuf, bBuf, outBuf, sizeA, sizeB) {
    const total = sizeA + sizeB;
    const totalWG = ceilDiv(total, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);
    const params = this._cachedUniform(new Uint32Array([sizeA, sizeB, wgX]));
    const bg = this.device.createBindGroup({
      layout: this.pipelines.concat.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.concat);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }
}

export { CONFIG as TWO_STREAM_CONFIG };
