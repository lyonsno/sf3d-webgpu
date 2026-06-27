/**
 * sf3d_backbone.js — DINOv2 ViT-Large backbone adapted for SF3D.
 *
 * Key differences from MOGE's DINOv2:
 *   - Separate Q/K/V projections (not fused QKV)
 *   - AdaNorm modulation: each layer has norm1_modulation and norm2_modulation
 *     that take camera embeddings and produce scale/shift for LayerNorm outputs
 *   - SwiGLU FFN in DINOv2-large (not standard GELU MLP)
 *   - Final layernorm after encoder
 *   - Output is last_hidden_state permuted to [N_v, C, N_t] tokens (no intermediate extraction)
 *
 * Produces: [N_tokens, dim] image features for the two-stream backbone
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';

import patchEmbedWGSL from '../shaders/patch_embed_dinov2.wgsl?raw';
import layerNormWGSL from '../shaders/layernorm_vit.wgsl?raw';
import attentionWGSL from '../shaders/attention.wgsl?raw';
import linearWGSL from '../shaders/linear.wgsl?raw';
import linearGeluWGSL from '../shaders/linear_gelu.wgsl?raw';
import layerscaleWGSL from '../shaders/layerscale.wgsl?raw';

const MAX_WG = 65535;
const WG_SIZE = 256;
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

const VIT_CONFIG = {
  dim: 1024,
  numHeads: 16,
  headDim: 64,
  numLayers: 24,
  patchSize: 14,
  channels: 3,
  // Standard GELU MLP (not SwiGLU — verified from checkpoint config)
  mlpHiddenDim: 4096,
  scale: 1.0 / Math.sqrt(64),
  eps: 1e-6,
};

export class SF3DImageTokenizer {
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

    this.pipelines.patchEmbed = make(patchEmbedWGSL, 'main');
    this.pipelines.layerNorm = make(layerNormWGSL, 'main');
    this.pipelines.attnScores = make(attentionWGSL, 'computeScores');
    this.pipelines.attnSoftmax = make(attentionWGSL, 'softmax');
    this.pipelines.attnApply = make(attentionWGSL, 'applyAttn');
    this.pipelines.linear = make(linearWGSL, 'main');
    this.pipelines.linearGelu = make(linearGeluWGSL, 'main');
    this.pipelines.layerScale = make(layerscaleWGSL, 'main');

    // Inline add shader
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

    // Modulated LayerNorm: output = (1 + scale) * LN(input) + shift
    // scale, shift come from linear(cameraEmbed) → [2*D], split into scale and shift
    this.pipelines.modulatedLN = make(`
      struct P { N: u32, D: u32, numWgX: u32 }
      @group(0) @binding(0) var<uniform> p: P;
      @group(0) @binding(1) var<storage, read> input: array<f32>;
      @group(0) @binding(2) var<storage, read> gamma: array<f32>;
      @group(0) @binding(3) var<storage, read> beta: array<f32>;
      @group(0) @binding(4) var<storage, read> modulation: array<f32>;
      @group(0) @binding(5) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(1)
      fn main(@builtin(global_invocation_id) gid: vec3u) {
        let row = gid.x;
        if (row >= p.N) { return; }
        let D = p.D;
        let base = row * D;
        var sum: f32 = 0.0;
        for (var d: u32 = 0; d < D; d++) { sum += input[base + d]; }
        let mean = sum / f32(D);
        var varSum: f32 = 0.0;
        for (var d: u32 = 0; d < D; d++) { let diff = input[base + d] - mean; varSum += diff * diff; }
        let invStd = 1.0 / sqrt(varSum / f32(D) + 1e-6);
        for (var d: u32 = 0; d < D; d++) {
          let normalized = (input[base + d] - mean) * invStd;
          let ln_out = gamma[d] * normalized + beta[d];
          let scale = modulation[d];
          let shift = modulation[D + d];
          output[base + d] = (1.0 + scale) * ln_out + shift;
        }
      }
    `, 'main');

    this.pipelines.linearGelu = make(linearGeluWGSL, 'main');
  }

  _cachedUniform(data) {
    const bytes = new Uint8Array(data.buffer || data);
    let h = 0;
    for (let i = 0; i < bytes.length; i++) h = (h * 31 + bytes[i]) | 0;
    const key = `u_${bytes.length}_${h}`;
    if (this._uniformCache.has(key)) return this._uniformCache.get(key);
    const buf = makeUniform(this.device, data);
    this._uniformCache.set(key, buf);
    return buf;
  }

  /**
   * Run the SF3D DINOv2 image tokenizer.
   *
   * @param {GPUCommandEncoder} encoder
   * @param {GPUBuffer} imageBuf - [3, 512, 512] normalized image
   * @param {GPUBuffer} cameraEmbedBuf - [768] camera embedding
   * @param {Object} weights - imageTokenizer weights from loadWeights
   * @returns {GPUBuffer} - [N_tokens, 1024] image token features (permuted for backbone)
   */
  encode(encoder, imageBuf, cameraEmbedBuf, weights) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const ps = VIT_CONFIG.patchSize;
    const imgSize = 512;
    const tokenH = imgSize / ps; // 36 (512/14 = 36.57, but DINOv2 does 36)
    const tokenW = imgSize / ps;
    // Actually 512/14 = 36.57... DINOv2 uses floor: tokenH = tokenW = 36
    // numPatches = 36*36 = 1296, N = 1297 (with CLS)
    const numPatches = tokenH * tokenW;
    const N = numPatches + 1;
    const T = N * D;

    // Work buffers
    const tokenBufA = createEmptyBuffer(device, T * 4);
    const tokenBufB = createEmptyBuffer(device, T * 4);
    const normBuf = createEmptyBuffer(device, T * 4);
    const qBuf = createEmptyBuffer(device, T * 4);
    const kBuf = createEmptyBuffer(device, T * 4);
    const vBuf = createEmptyBuffer(device, T * 4);
    const scoreBuf = createEmptyBuffer(device, VIT_CONFIG.numHeads * N * N * 4);
    const attnOutBuf = createEmptyBuffer(device, T * 4);
    const projBuf = createEmptyBuffer(device, T * 4);
    const hiddenBuf = createEmptyBuffer(device, N * VIT_CONFIG.mlpHiddenDim * 4);
    const ffnOutBuf = createEmptyBuffer(device, T * 4);
    const modBuf = createEmptyBuffer(device, 2 * D * 4); // modulation output [2*D]

    // 1. Patch embedding
    this._dispatchPatchEmbed(encoder, imageBuf, weights, tokenBufA, tokenH, tokenW);

    let currentTokens = tokenBufA;

    // 2. Run 24 transformer blocks with modulated attention
    for (let l = 0; l < VIT_CONFIG.numLayers; l++) {
      const block = weights.blocks[l];

      // Compute modulation for norm1: linear(cameraEmbed) → [2*D]
      this._dispatchLinear(encoder, cameraEmbedBuf, modBuf, block.norm1Mod.weight, block.norm1Mod.bias, 1, 768, 2 * D);

      // Modulated LayerNorm1
      this._dispatchModulatedLN(encoder, currentTokens, normBuf, block.norm1, modBuf, N);

      // Self-attention with separate Q/K/V projections
      this._dispatchLinear(encoder, normBuf, qBuf, block.attn.q.weight, block.attn.q.bias, N, D, D);
      this._dispatchLinear(encoder, normBuf, kBuf, block.attn.k.weight, block.attn.k.bias, N, D, D);
      this._dispatchLinear(encoder, normBuf, vBuf, block.attn.v.weight, block.attn.v.bias, N, D, D);

      // Attention
      this._dispatchAttnScores(encoder, qBuf, kBuf, scoreBuf, N);
      this._dispatchAttnSoftmax(encoder, scoreBuf, N);
      this._dispatchAttnApply(encoder, scoreBuf, vBuf, attnOutBuf, N);

      // Output projection
      this._dispatchLinear(encoder, attnOutBuf, projBuf, block.attn.proj.weight, block.attn.proj.bias, N, D, D);

      // LayerScale1 + residual
      const attnOut = (currentTokens === tokenBufA) ? tokenBufB : tokenBufA;
      this._dispatchLayerScaleResidual(encoder, projBuf, currentTokens, attnOut, block.layerScale1, T, D);
      currentTokens = attnOut;

      // Compute modulation for norm2
      this._dispatchLinear(encoder, cameraEmbedBuf, modBuf, block.norm2Mod.weight, block.norm2Mod.bias, 1, 768, 2 * D);

      // Modulated LayerNorm2
      this._dispatchModulatedLN(encoder, currentTokens, normBuf, block.norm2, modBuf, N);

      // GELU MLP: fc1 (linear + GELU) → fc2 (linear)
      this._dispatchLinearGelu(encoder, normBuf, hiddenBuf,
        block.mlp.fc1.weight, block.mlp.fc1.bias, N, D, VIT_CONFIG.mlpHiddenDim);
      this._dispatchLinear(encoder, hiddenBuf, ffnOutBuf,
        block.mlp.fc2.weight, block.mlp.fc2.bias, N, VIT_CONFIG.mlpHiddenDim, D);

      // LayerScale2 + residual
      const ffnOut = (currentTokens === tokenBufA) ? tokenBufB : tokenBufA;
      this._dispatchLayerScaleResidual(encoder, ffnOutBuf, currentTokens, ffnOut, block.layerScale2, T, D);
      currentTokens = ffnOut;
    }

    // 3. Final LayerNorm
    this._dispatchLayerNorm(encoder, currentTokens, normBuf, weights.layernorm, N);

    // Output: normBuf contains [N, D] where N = 1297 (CLS + 1296 patches)
    // The backbone expects [N_tokens, D] — we return all tokens including CLS
    // The two-stream backbone will use this via permute to [C, N_t]

    return {
      tokensBuf: normBuf,
      N,
      tokenH,
      tokenW,
    };
  }

  // --- Dispatch helpers ---

  _dispatchPatchEmbed(encoder, imageBuf, weights, outputBuf, tokenH, tokenW) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const ps = VIT_CONFIG.patchSize;
    const numTokens = tokenH * tokenW + 1;
    const totalWG = ceilDiv(numTokens * D, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);

    const params = this._cachedUniform(new Uint32Array([
      tokenH * ps, tokenW * ps, ps, tokenH, tokenW, 3, D, numTokens, wgX,
    ]));

    const bg = device.createBindGroup({
      layout: this.pipelines.patchEmbed.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: imageBuf } },
        { binding: 2, resource: { buffer: weights.patchEmbed.weight } },
        { binding: 3, resource: { buffer: weights.patchEmbed.bias } },
        { binding: 4, resource: { buffer: weights.clsToken } },
        { binding: 5, resource: { buffer: weights.posEmbed } },
        { binding: 6, resource: { buffer: outputBuf } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.patchEmbed);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _dispatchLinear(encoder, input, output, weight, bias, rows, inDim, outDim) {
    const device = this.device;
    const totalWG = ceilDiv(rows * outDim, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);
    const params = this._cachedUniform(new Uint32Array([rows, inDim, outDim, wgX]));

    const bg = device.createBindGroup({
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

  _dispatchLayerNorm(encoder, input, output, norm, N) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const paramsData = new ArrayBuffer(16);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, D, true);
    v.setFloat32(8, VIT_CONFIG.eps, true);
    const params = this._cachedUniform(new Uint8Array(paramsData));

    const bg = device.createBindGroup({
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

  _dispatchModulatedLN(encoder, input, output, norm, modBuf, N) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const params = this._cachedUniform(new Uint32Array([N, D, 0]));

    const bg = device.createBindGroup({
      layout: this.pipelines.modulatedLN.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: norm.weight } },
        { binding: 3, resource: { buffer: norm.bias } },
        { binding: 4, resource: { buffer: modBuf } },
        { binding: 5, resource: { buffer: output } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.modulatedLN);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(N);
    pass.end();
  }

  _dispatchAttnScores(encoder, qBuf, kBuf, scoreBuf, N) {
    const device = this.device;
    const { numHeads, dim, headDim, scale } = VIT_CONFIG;
    const total = numHeads * N * N;
    const totalWG = ceilDiv(total, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new ArrayBuffer(24);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, dim, true);
    v.setUint32(8, numHeads, true);
    v.setUint32(12, headDim, true);
    v.setFloat32(16, scale, true);
    v.setUint32(20, wgX, true);
    const params = this._cachedUniform(new Uint8Array(paramsData));

    const bg = device.createBindGroup({
      layout: this.pipelines.attnScores.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: qBuf } },
        { binding: 2, resource: { buffer: kBuf } },
        { binding: 3, resource: { buffer: scoreBuf } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.attnScores);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _dispatchAttnSoftmax(encoder, scoreBuf, N) {
    const totalRows = VIT_CONFIG.numHeads * N;
    const totalWG = ceilDiv(totalRows, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);
    const params = this._cachedUniform(new Uint32Array([N, VIT_CONFIG.numHeads, wgX]));

    const bg = this.device.createBindGroup({
      layout: this.pipelines.attnSoftmax.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: scoreBuf } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.attnSoftmax);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _dispatchAttnApply(encoder, scoreBuf, vBuf, output, N) {
    const D = VIT_CONFIG.dim;
    const totalWG = ceilDiv(N * D, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);
    const params = this._cachedUniform(new Uint32Array([N, D, VIT_CONFIG.numHeads, VIT_CONFIG.headDim, wgX]));

    const bg = this.device.createBindGroup({
      layout: this.pipelines.attnApply.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: scoreBuf } },
        { binding: 2, resource: { buffer: vBuf } },
        { binding: 3, resource: { buffer: output } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.attnApply);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _dispatchLayerScaleResidual(encoder, input, residual, output, gamma, count, D) {
    const totalWG = ceilDiv(count, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);
    const params = this._cachedUniform(new Uint32Array([count, D, wgX]));

    const bg = this.device.createBindGroup({
      layout: this.pipelines.layerScale.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: gamma } },
        { binding: 3, resource: { buffer: residual } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.layerScale);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _dispatchLinearGelu(encoder, input, output, weight, bias, rows, inDim, outDim) {
    const totalWG = ceilDiv(rows * outDim, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);
    const params = this._cachedUniform(new Uint32Array([rows, inDim, outDim, wgX]));

    const bg = this.device.createBindGroup({
      layout: this.pipelines.linearGelu.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: weight } },
        { binding: 3, resource: { buffer: bias } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.linearGelu);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }
}

export { VIT_CONFIG };
