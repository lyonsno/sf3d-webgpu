/**
 * clip_estimator.js — CLIP-based material property estimator for SF3D.
 *
 * Runs CLIP ViT-B/32 visual encoder on the input image, then two small MLP
 * heads to predict roughness and metallic as beta distribution modes.
 *
 * Architecture:
 *   - Patch embedding: 32×32 stride-32 conv → 49 patches of 768d
 *   - CLS token prepend → 50 tokens
 *   - 12 transformer blocks: LN → fused QKV attention (12 heads) → LN → MLP (GELU)
 *   - LN → CLS token → visual projection (768→512)
 *   - Two MLP heads: shared 3-layer (512→512, ReLU) → two branches → beta params → mode
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';

const HIDDEN_DIM = 768;
const NUM_HEADS = 12;
const HEAD_DIM = 64;
const MLP_DIM = 3072;
const NUM_BLOCKS = 12;
const PATCH_SIZE = 32;
const IMAGE_SIZE = 224;
const NUM_PATCHES = (IMAGE_SIZE / PATCH_SIZE) ** 2; // 49
const NUM_TOKENS = NUM_PATCHES + 1; // 50
const PROJ_DIM = 512;
const WG_SIZE = 256;

const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD = [0.26862954, 0.26130258, 0.27577711];

let _pipelines = null;
let _weightBuffers = null;

function ceilDiv(a, b) { return Math.ceil(a / b); }
function splitWG(total) {
  const maxX = 65535;
  if (total <= maxX) return [total, 1];
  return [maxX, ceilDiv(total, maxX)];
}

async function _ensurePipelines(device) {
  if (_pipelines) return;

  const [linearSrc, layernormSrc] = await Promise.all([
    fetch('/src/shaders/linear.wgsl').then(r => r.text()),
    fetch('/src/shaders/layernorm_vit.wgsl').then(r => r.text()),
  ]);

  _pipelines = {
    linear: device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: linearSrc }), entryPoint: 'main' },
    }),
    layernorm: device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: layernormSrc }), entryPoint: 'main' },
    }),
  };
}

function _ensureWeightBuffers(device, weights) {
  if (_weightBuffers) return;
  _weightBuffers = {};

  const makeGPU = (name) => weights._rawGet(name);

  // Visual encoder weights
  _weightBuffers.lnPre = { weight: makeGPU('image_estimator.model.visual.ln_pre.weight'),
                           bias: makeGPU('image_estimator.model.visual.ln_pre.bias') };
  _weightBuffers.lnPost = { weight: makeGPU('image_estimator.model.visual.ln_post.weight'),
                            bias: makeGPU('image_estimator.model.visual.ln_post.bias') };

  _weightBuffers.blocks = [];
  for (let i = 0; i < NUM_BLOCKS; i++) {
    const p = `image_estimator.model.visual.transformer.resblocks.${i}`;
    _weightBuffers.blocks.push({
      ln1: { weight: makeGPU(`${p}.ln_1.weight`), bias: makeGPU(`${p}.ln_1.bias`) },
      ln2: { weight: makeGPU(`${p}.ln_2.weight`), bias: makeGPU(`${p}.ln_2.bias`) },
      qkv: { weight: makeGPU(`${p}.attn.in_proj_weight`), bias: makeGPU(`${p}.attn.in_proj_bias`) },
      outProj: { weight: makeGPU(`${p}.attn.out_proj.weight`), bias: makeGPU(`${p}.attn.out_proj.bias`) },
      fc: { weight: makeGPU(`${p}.mlp.c_fc.weight`), bias: makeGPU(`${p}.mlp.c_fc.bias`) },
      proj: { weight: makeGPU(`${p}.mlp.c_proj.weight`), bias: makeGPU(`${p}.mlp.c_proj.bias`) },
    });
  }
}

/**
 * Run CLIP material estimation on an image.
 */
export async function estimateMaterials(device, imagePixels, imgWidth, imgHeight, weights) {
  await _ensurePipelines(device);
  _ensureWeightBuffers(device, weights);

  // Step 1: CPU preprocessing — resize to 224, normalize, patch embed
  const embeddings = _patchEmbed(
    _preprocessForCLIP(imagePixels, imgWidth, imgHeight), weights);

  // Step 2: GPU visual transformer
  const features = await _runVisualTransformer(device, embeddings, weights);

  // Step 3: CPU heads (tiny)
  const roughness = _runHead(features, weights, 'roughness');
  const metallic = _runHead(features, weights, 'metallic');

  // Diagnostic: compare features against PyTorch reference
  console.log(`CLIP features[0:5]: ${Array.from(features.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}`);
  console.log(`CLIP features range: [${Math.min(...features).toFixed(4)}, ${Math.max(...features).toFixed(4)}]`);
  console.log(`CLIP features mean: ${(features.reduce((a,b)=>a+b,0)/features.length).toFixed(4)}`);
  // PyTorch ref: features[0:5] = [-0.1376, 0.6448, -0.5184, -0.3570, 0.1465], range=[-4.27, 4.27], mean=0.0271
  console.log(`CLIP material estimation: roughness=${roughness.toFixed(3)}, metallic=${metallic.toFixed(3)}`);
  return { roughness, metallic };
}

function _preprocessForCLIP(pixels, width, height) {
  const out = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE);
  for (let y = 0; y < IMAGE_SIZE; y++) {
    for (let x = 0; x < IMAGE_SIZE; x++) {
      const srcX = x * (width - 1) / (IMAGE_SIZE - 1);
      const srcY = y * (height - 1) / (IMAGE_SIZE - 1);
      const x0 = Math.floor(srcX), y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, width - 1), y1 = Math.min(y0 + 1, height - 1);
      const fx = srcX - x0, fy = srcY - y0;
      for (let c = 0; c < 3; c++) {
        const v = pixels[(y0 * width + x0) * 4 + c] * (1-fx)*(1-fy) +
                  pixels[(y0 * width + x1) * 4 + c] * fx*(1-fy) +
                  pixels[(y1 * width + x0) * 4 + c] * (1-fx)*fy +
                  pixels[(y1 * width + x1) * 4 + c] * fx*fy;
        out[c * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] = (v - CLIP_MEAN[c]) / CLIP_STD[c];
      }
    }
  }
  return out;
}

function _patchEmbed(image, weights) {
  const conv1W = weights._rawGetCPU('image_estimator.model.visual.conv1.weight'); // [768, 3, 32, 32]
  const classEmb = weights._rawGetCPU('image_estimator.model.visual.class_embedding'); // [768]
  const posEmb = weights._rawGetCPU('image_estimator.model.visual.positional_embedding'); // [50, 768]

  const patchDim = 3 * PATCH_SIZE * PATCH_SIZE; // 3072
  const result = new Float32Array(NUM_TOKENS * HIDDEN_DIM);

  // CLS token
  for (let d = 0; d < HIDDEN_DIM; d++) result[d] = classEmb[d];

  // Patch embeddings: patches[49, 3072] × conv1W[768, 3072]^T → [49, 768]
  for (let py = 0; py < 7; py++) {
    for (let px = 0; px < 7; px++) {
      const patchIdx = py * 7 + px;
      for (let d = 0; d < HIDDEN_DIM; d++) {
        let sum = 0;
        for (let c = 0; c < 3; c++) {
          for (let dy = 0; dy < PATCH_SIZE; dy++) {
            for (let dx = 0; dx < PATCH_SIZE; dx++) {
              sum += image[c * IMAGE_SIZE * IMAGE_SIZE + (py*PATCH_SIZE+dy) * IMAGE_SIZE + (px*PATCH_SIZE+dx)]
                   * conv1W[d * patchDim + c * PATCH_SIZE * PATCH_SIZE + dy * PATCH_SIZE + dx];
            }
          }
        }
        result[(patchIdx + 1) * HIDDEN_DIM + d] = sum;
      }
    }
  }

  // Add positional embedding
  for (let i = 0; i < NUM_TOKENS * HIDDEN_DIM; i++) result[i] += posEmb[i];
  return result;
}

async function _runVisualTransformer(device, embeddings, weights) {
  const N = NUM_TOKENS, D = HIDDEN_DIM;

  let xBuf = createStorageBuffer(device, new Float32Array(embeddings));
  const encoder = device.createCommandEncoder();

  // Pre-LN
  xBuf = _dispatchLN(encoder, device, xBuf, N, D, _weightBuffers.lnPre);

  for (let b = 0; b < NUM_BLOCKS; b++) {
    const blk = _weightBuffers.blocks[b];

    // LN1 → fused QKV → attention → out proj → residual
    const ln1 = _dispatchLN(encoder, device, xBuf, N, D, blk.ln1);
    const qkv = _dispatchLinear(encoder, device, ln1, N, D, 3*D, blk.qkv);
    const attn = _dispatchFusedAttn(encoder, device, qkv, N);
    const proj = _dispatchLinear(encoder, device, attn, N, D, D, blk.outProj);
    xBuf = _dispatchAdd(encoder, device, xBuf, proj, N * D);

    // LN2 → MLP (fc → GELU → proj) → residual
    const ln2 = _dispatchLN(encoder, device, xBuf, N, D, blk.ln2);
    const fc = _dispatchLinear(encoder, device, ln2, N, D, MLP_DIM, blk.fc);
    const gelu = _dispatchGelu(encoder, device, fc, N * MLP_DIM);
    const mlp = _dispatchLinear(encoder, device, gelu, N, MLP_DIM, D, blk.proj);
    xBuf = _dispatchAdd(encoder, device, xBuf, mlp, N * D);
  }

  // Post-LN
  xBuf = _dispatchLN(encoder, device, xBuf, N, D, _weightBuffers.lnPost);

  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  // Read CLS token, project to 512d on CPU
  const allTokens = await readBuffer(device, xBuf, N * D * 4);
  const cls = allTokens.slice(0, D);

  // visual.proj: PyTorch [768, 512], converter transposed to [512, 768]
  // output[d] = sum_k cls[k] * proj_transposed[d * 768 + k]
  const projW = weights._rawGetCPU('image_estimator.model.visual.proj');
  const features = new Float32Array(PROJ_DIM);
  for (let d = 0; d < PROJ_DIM; d++) {
    let sum = 0;
    for (let k = 0; k < D; k++) sum += cls[k] * projW[d * D + k];
    features[d] = sum;
  }
  return features;
}

// --- GPU dispatch helpers ---

function _dispatchLinear(encoder, device, input, rows, inDim, outDim, w) {
  const totalWG = ceilDiv(rows * outDim, WG_SIZE);
  const [wgX, wgY] = splitWG(totalWG);
  const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([rows, inDim, outDim, wgX]));
  const output = createEmptyBuffer(device, rows * outDim * 4);
  const bg = device.createBindGroup({
    layout: _pipelines.linear.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: w.weight } },
      { binding: 3, resource: { buffer: w.bias } },
      { binding: 4, resource: { buffer: output } },
    ],
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(_pipelines.linear);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();
  return output;
}

function _dispatchLN(encoder, device, input, N, D, norm) {
  const paramsData = new ArrayBuffer(16);
  const v = new DataView(paramsData);
  v.setUint32(0, N, true);
  v.setUint32(4, D, true);
  v.setFloat32(8, 1e-5, true);
  const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint8Array(paramsData));
  const output = createEmptyBuffer(device, N * D * 4);
  const bg = device.createBindGroup({
    layout: _pipelines.layernorm.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: norm.weight } },
      { binding: 3, resource: { buffer: norm.bias } },
      { binding: 4, resource: { buffer: output } },
    ],
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(_pipelines.layernorm);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(N);
  pass.end();
  return output;
}

function _dispatchAdd(encoder, device, a, b, count) {
  const shaderCode = `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i >= ${count}u) { return; }
      out[i] = a[i] + b[i];
    }
  `;
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaderCode }), entryPoint: 'main' },
  });
  const output = createEmptyBuffer(device, count * 4);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: a } },
      { binding: 1, resource: { buffer: b } },
      { binding: 2, resource: { buffer: output } },
    ],
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(ceilDiv(count, 256));
  pass.end();
  return output;
}

function _dispatchGelu(encoder, device, input, count) {
  const shaderCode = `
    @group(0) @binding(0) var<storage, read> inp: array<f32>;
    @group(0) @binding(1) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i >= ${count}u) { return; }
      let x = inp[i];
      // Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
      // WGSL doesn't have erf, use Abramowitz & Stegun approximation
      let t = 1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476));
      let erf_approx = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x * 0.5);
      let erf_val = select(-erf_approx, erf_approx, x >= 0.0);
      out[i] = x * 0.5 * (1.0 + erf_val);
    }
  `;
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaderCode }), entryPoint: 'main' },
  });
  const output = createEmptyBuffer(device, count * 4);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
    ],
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(ceilDiv(count, 256));
  pass.end();
  return output;
}

function _dispatchFusedAttn(encoder, device, qkvBuf, N) {
  const D = HIDDEN_DIM;
  const scale = 1.0 / Math.sqrt(HEAD_DIM);
  const shaderCode = `
    @group(0) @binding(0) var<storage, read> qkv: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      if (idx >= ${N * NUM_HEADS}u) { return; }
      let qi = idx / ${NUM_HEADS}u;
      let h = idx % ${NUM_HEADS}u;
      var scores: array<f32, ${N}>;
      var maxS: f32 = -1e30;
      for (var ki: u32 = 0u; ki < ${N}u; ki++) {
        var s: f32 = 0.0;
        for (var d: u32 = 0u; d < ${HEAD_DIM}u; d++) {
          s += qkv[qi * ${3*D}u + h * ${HEAD_DIM}u + d]
             * qkv[ki * ${3*D}u + ${D}u + h * ${HEAD_DIM}u + d];
        }
        s *= ${scale};
        scores[ki] = s;
        maxS = max(maxS, s);
      }
      var sumE: f32 = 0.0;
      for (var ki: u32 = 0u; ki < ${N}u; ki++) {
        scores[ki] = exp(scores[ki] - maxS);
        sumE += scores[ki];
      }
      let inv = 1.0 / sumE;
      for (var d: u32 = 0u; d < ${HEAD_DIM}u; d++) {
        var val: f32 = 0.0;
        for (var ki: u32 = 0u; ki < ${N}u; ki++) {
          val += scores[ki] * inv * qkv[ki * ${3*D}u + ${2*D}u + h * ${HEAD_DIM}u + d];
        }
        output[qi * ${D}u + h * ${HEAD_DIM}u + d] = val;
      }
    }
  `;
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaderCode }), entryPoint: 'main' },
  });
  const output = createEmptyBuffer(device, N * D * 4);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: qkvBuf } },
      { binding: 1, resource: { buffer: output } },
    ],
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(ceilDiv(N * NUM_HEADS, 64));
  pass.end();
  return output;
}

// --- CPU head computation ---

function _runHead(features, weights, headName) {
  const p = `image_estimator.heads.${headName}`;
  const bias = 1.0;
  const g = (n) => weights._rawGetCPU(n);
  let x = new Float32Array(features);
  for (const l of ['0.0', '0.2', '0.4']) x = _cpuLinearReLU(x, g(`${p}.${l}.weight`), g(`${p}.${l}.bias`));
  let d1 = _cpuLinear(_cpuLinearReLU(x, g(`${p}.1.0.weight`), g(`${p}.1.0.bias`)),
                       g(`${p}.1.2.weight`), g(`${p}.1.2.bias`));
  let d2 = _cpuLinear(_cpuLinearReLU(x, g(`${p}.2.0.weight`), g(`${p}.2.0.bias`)),
                       g(`${p}.2.2.weight`), g(`${p}.2.2.bias`));
  const alpha = _softplus(d1[0] + bias), beta = _softplus(d2[0] + bias);
  if (alpha <= 1 || beta <= 1) return alpha / (alpha + beta);
  return Math.max(0, Math.min(1, (alpha - 1) / (alpha + beta - 2)));
}

function _cpuLinear(x, weight, bias) {
  const inDim = x.length, outDim = bias.length;
  const out = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let s = bias[o];
    for (let i = 0; i < inDim; i++) s += x[i] * weight[i * outDim + o];
    out[o] = s;
  }
  return out;
}

function _cpuLinearReLU(x, w, b) {
  const out = _cpuLinear(x, w, b);
  for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
  return out;
}

function _softplus(x) { return x > 20 ? x : Math.log(1 + Math.exp(x)); }
