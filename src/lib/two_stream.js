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
import linearRangeWGSL from '../shaders/linear_range.wgsl?raw';
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

export const TWO_STREAM_STAGE_IDS = Object.freeze([
  'setup',
  ...Array.from({ length: CONFIG.numBlocks }, (_, block) => [
    `block-${block}-fuse-in`,
    ...Array.from(
      { length: CONFIG.numBasicBlocks },
      (_, basic) => `block-${block}-basic-${basic}`,
    ),
    `block-${block}-fuse-out`,
  ]).flat(),
  'final',
]);

export function normalizeLinearRowRange(totalRows, rowStart, rowCount) {
  if (!Number.isSafeInteger(totalRows) || totalRows <= 0) {
    throw new TypeError('totalRows must be a positive safe integer');
  }
  if (!Number.isSafeInteger(rowStart) || rowStart < 0) {
    throw new TypeError('rowStart must be a non-negative safe integer');
  }
  if (!Number.isSafeInteger(rowCount) || rowCount <= 0) {
    throw new TypeError('rowCount must be a positive safe integer');
  }
  const rowEnd = rowStart + rowCount;
  if (!Number.isSafeInteger(rowEnd) || rowEnd > totalRows) {
    throw new RangeError(
      `linear row range ${rowStart}-${rowEnd} exceeds totalRows ${totalRows}`,
    );
  }
  return Object.freeze({ totalRows, rowStart, rowCount, rowEnd });
}

export function createLinearRowRanges(totalRows, rowsPerDuty) {
  if (!Number.isSafeInteger(rowsPerDuty) || rowsPerDuty <= 0) {
    throw new TypeError('rowsPerDuty must be a positive safe integer');
  }
  normalizeLinearRowRange(totalRows, 0, Math.min(totalRows, rowsPerDuty));
  const ranges = [];
  for (let rowStart = 0; rowStart < totalRows; rowStart += rowsPerDuty) {
    const range = normalizeLinearRowRange(
      totalRows,
      rowStart,
      Math.min(rowsPerDuty, totalRows - rowStart),
    );
    ranges.push(Object.freeze({ rangeIndex: ranges.length, ...range }));
  }
  return Object.freeze(ranges);
}

export function createTwoStreamAttentionDutyPlan(N_img, options = {}) {
  if (!Number.isSafeInteger(N_img) || N_img <= 0) {
    throw new TypeError('N_img must be a positive safe integer');
  }
  const linearRowsPerDuty = options.linearRowsPerDuty ?? 128;
  if (!Number.isSafeInteger(linearRowsPerDuty) || linearRowsPerDuty <= 0) {
    throw new TypeError('linearRowsPerDuty must be a positive safe integer');
  }
  const duties = [];
  const append = duty => {
    duties.push(Object.freeze({
      dutyIndex: duties.length,
      dutyId: duty.dutyId,
      ...duty,
    }));
  };
  const appendAttention = (ownerId, tileCount) => {
    for (let tileIndex = 0; tileIndex < tileCount; tileIndex++) {
      append({
        dutyId: `${ownerId}-tile-${tileIndex}`,
        kind: 'attention-tile',
        ownerId,
        tileIndex,
        tileCount,
      });
    }
  };
  const appendLinearRanges = (ownerId, kind, block, ranges) => {
    for (const range of ranges) {
      append({
        dutyId: `${ownerId}-range-${range.rangeIndex}`,
        kind,
        ownerId,
        block,
        direction: 'out',
        rangeIndex: range.rangeIndex,
        rangeCount: ranges.length,
        totalRows: range.totalRows,
        rowStart: range.rowStart,
        rowCount: range.rowCount,
        rowEnd: range.rowEnd,
      });
    }
  };
  const latentTiles = ceilDiv(N_img + CONFIG.numLatents, 128);
  const triplaneTiles = ceilDiv(CONFIG.triplaneTokens, 128);
  const triplaneRowRanges = createLinearRowRanges(
    CONFIG.triplaneTokens,
    linearRowsPerDuty,
  );

  append({ dutyId: 'setup', kind: 'setup' });
  for (let block = 0; block < CONFIG.numBlocks; block++) {
    const fuseIn = `block-${block}-fuse-in`;
    append({ dutyId: `${fuseIn}-prepare`, kind: 'fuse-prepare', block, direction: 'in' });
    appendAttention(fuseIn, latentTiles);
    append({ dutyId: `${fuseIn}-finish`, kind: 'fuse-finish', block, direction: 'in' });

    for (let basic = 0; basic < CONFIG.numBasicBlocks; basic++) {
      const ownerId = `block-${block}-basic-${basic}`;
      const selfOwner = `${ownerId}-self`;
      const crossOwner = `${ownerId}-cross`;
      append({
        dutyId: `${selfOwner}-prepare`,
        kind: 'basic-self-prepare',
        block,
        basic,
      });
      appendAttention(selfOwner, latentTiles);
      append({
        dutyId: `${crossOwner}-prepare`,
        kind: 'basic-cross-prepare',
        block,
        basic,
      });
      appendAttention(crossOwner, latentTiles);
      append({
        dutyId: `${ownerId}-finish`,
        kind: 'basic-finish',
        block,
        basic,
      });
    }

    const fuseOut = `block-${block}-fuse-out`;
    append({ dutyId: `${fuseOut}-prepare`, kind: 'fuse-prepare', block, direction: 'out' });
    appendAttention(fuseOut, triplaneTiles);
    appendLinearRanges(
      `${fuseOut}-attention-projection`,
      'fuse-attention-linear-range',
      block,
      triplaneRowRanges,
    );
    append({
      dutyId: `${fuseOut}-residual-norm`,
      kind: 'fuse-residual-norm',
      block,
      direction: 'out',
    });
    appendLinearRanges(
      `${fuseOut}-geglu-expansion`,
      'fuse-geglu-linear-range',
      block,
      triplaneRowRanges,
    );
    append({
      dutyId: `${fuseOut}-geglu-activate`,
      kind: 'fuse-geglu-activate',
      block,
      direction: 'out',
    });
    appendLinearRanges(
      `${fuseOut}-ffn-projection`,
      'fuse-ffn-linear-range',
      block,
      triplaneRowRanges,
    );
    append({
      dutyId: `${fuseOut}-final-residual`,
      kind: 'fuse-final-residual',
      block,
      direction: 'out',
    });
  }
  append({ dutyId: 'final', kind: 'final' });
  return Object.freeze(duties);
}

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
    this.pipelines.linearRange = make(linearRangeWGSL, 'main');
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
    const state = this.createForwardState(imageTokensBuf, N_img, weights);
    for (let stageIndex = 0; stageIndex < TWO_STREAM_STAGE_IDS.length; stageIndex++) {
      this.dispatchForwardStage(encoder, state, stageIndex);
    }
    return this.getForwardResult(state);
  }

  createForwardState(imageTokensBuf, N_img, weights) {
    if (!Number.isSafeInteger(N_img) || N_img <= 0) {
      throw new TypeError('N_img must be a positive safe integer');
    }
    this._diagnosticBuffers = {};
    return {
      imageTokensBuf,
      N_img,
      weights,
      N_latent: N_img + CONFIG.numLatents,
      currentLatent: null,
      currentTriplane: null,
      nextStageIndex: 0,
      finePlan: null,
      nextFineDutyIndex: 0,
      activeOperation: null,
      result: null,
    };
  }

  createAttentionForwardState(imageTokensBuf, N_img, weights, options = {}) {
    const state = this.createForwardState(imageTokensBuf, N_img, weights);
    state.finePlan = createTwoStreamAttentionDutyPlan(N_img, options);
    return state;
  }

  dispatchForwardStage(encoder, state, stageIndex) {
    if (!Number.isSafeInteger(stageIndex)
      || stageIndex < 0
      || stageIndex >= TWO_STREAM_STAGE_IDS.length) {
      throw new RangeError(`invalid two-stream stage index ${stageIndex}`);
    }
    if (stageIndex !== state.nextStageIndex) {
      throw new Error(
        `two-stream stage ${stageIndex} is out of order; expected ${state.nextStageIndex}`,
      );
    }

    if (stageIndex === 0) {
      this._dispatchForwardSetup(encoder, state);
    } else if (stageIndex === TWO_STREAM_STAGE_IDS.length - 1) {
      this._dispatchForwardFinal(encoder, state);
    } else {
      const blockStage = stageIndex - 1;
      const stagesPerBlock = CONFIG.numBasicBlocks + 2;
      const blockIndex = Math.floor(blockStage / stagesPerBlock);
      const stageInBlock = blockStage % stagesPerBlock;
      this._dispatchForwardBlockStage(
        encoder,
        state,
        blockIndex,
        stageInBlock,
      );
    }

    state.nextStageIndex++;
    return state;
  }

  getForwardResult(state) {
    const stageComplete = state.nextStageIndex === TWO_STREAM_STAGE_IDS.length;
    const fineComplete = state.finePlan != null
      && state.nextFineDutyIndex === state.finePlan.length;
    if ((!stageComplete && !fineComplete) || state.result == null) {
      const completed = state.finePlan == null
        ? `${state.nextStageIndex}/${TWO_STREAM_STAGE_IDS.length}`
        : `${state.nextFineDutyIndex}/${state.finePlan.length}`;
      throw new Error(
        `two-stream forward is incomplete at stage ${completed}`,
      );
    }
    return state.result;
  }

  dispatchAttentionForwardDuty(encoder, state, dutyIndex) {
    if (state.finePlan == null) {
      throw new Error('two-stream attention-duty state has no fine plan');
    }
    if (dutyIndex !== state.nextFineDutyIndex) {
      throw new Error(
        `two-stream attention duty ${dutyIndex} is out of order; `
        + `expected ${state.nextFineDutyIndex}`,
      );
    }
    const duty = state.finePlan[dutyIndex];
    if (!duty) {
      throw new RangeError(`invalid two-stream attention duty ${dutyIndex}`);
    }

    switch (duty.kind) {
      case 'setup':
        this._dispatchForwardSetup(encoder, state);
        break;
      case 'fuse-prepare':
        this._dispatchFineFusePrepare(encoder, state, duty);
        break;
      case 'attention-tile':
        this._dispatchFineAttentionTile(encoder, state, duty);
        break;
      case 'fuse-finish':
        this._dispatchFineFuseFinish(encoder, state, duty);
        break;
      case 'fuse-attention-linear-range':
        this._dispatchFineFuseAttentionLinearRange(encoder, state, duty);
        break;
      case 'fuse-residual-norm':
        this._dispatchFineFuseResidualNorm(encoder, state, duty);
        break;
      case 'fuse-geglu-linear-range':
        this._dispatchFineFuseGEGLULinearRange(encoder, state, duty);
        break;
      case 'fuse-geglu-activate':
        this._dispatchFineFuseGEGLUActivate(encoder, state, duty);
        break;
      case 'fuse-ffn-linear-range':
        this._dispatchFineFuseFFNLinearRange(encoder, state, duty);
        break;
      case 'fuse-final-residual':
        this._dispatchFineFuseFinalResidual(encoder, state, duty);
        break;
      case 'basic-self-prepare':
        this._dispatchFineBasicSelfPrepare(encoder, state, duty);
        break;
      case 'basic-cross-prepare':
        this._dispatchFineBasicCrossPrepare(encoder, state, duty);
        break;
      case 'basic-finish':
        this._dispatchFineBasicFinish(encoder, state, duty);
        break;
      case 'final':
        if (state.activeOperation != null) {
          throw new Error(`cannot finalize with active ${state.activeOperation.ownerId}`);
        }
        this._dispatchForwardFinal(encoder, state);
        break;
      default:
        throw new Error(`unknown two-stream attention duty kind ${duty.kind}`);
    }
    state.nextFineDutyIndex++;
    return duty;
  }

  _requireFineOperation(state, kind, ownerId) {
    const operation = state.activeOperation;
    if (operation?.kind !== kind || operation.ownerId !== ownerId) {
      throw new Error(
        `two-stream duty ${ownerId} requires active ${kind}; `
        + `got ${operation?.kind ?? 'none'}:${operation?.ownerId ?? 'none'}`,
      );
    }
    return operation;
  }

  _dispatchFineFusePrepare(encoder, state, duty) {
    if (state.activeOperation != null) {
      throw new Error(`cannot prepare ${duty.dutyId} with active ${state.activeOperation.ownerId}`);
    }
    const D = CONFIG.dim;
    const N_tri = CONFIG.triplaneTokens;
    const block = state.weights.mainBlocks[duty.block];
    const incoming = duty.direction === 'in';
    const zBuf = incoming ? state.currentLatent : state.currentTriplane;
    const xBuf = incoming ? state.currentTriplane : state.currentLatent;
    const N_z = incoming ? state.N_latent : N_tri;
    const N_x = incoming ? N_tri : state.N_latent;
    const weights = incoming ? block.fuseBlockIn : block.fuseBlockOut;
    const zNormBuf = createEmptyBuffer(this.device, N_z * D * 4);
    this._dispatchLayerNorm(encoder, zBuf, zNormBuf, weights.normZ1, N_z, D);
    let xNormBuf = xBuf;
    if (weights.normX) {
      xNormBuf = createEmptyBuffer(this.device, N_x * D * 4);
      this._dispatchLayerNorm(encoder, xBuf, xNormBuf, weights.normX, N_x, D);
    }
    const ownerId = `block-${duty.block}-fuse-${duty.direction}`;
    state.activeOperation = {
      kind: 'fuse',
      ownerId,
      block: duty.block,
      direction: duty.direction,
      zBuf,
      weights,
      N_z,
      D,
      attention: this._createAttentionState(
        encoder,
        zNormBuf,
        xNormBuf,
        weights.attn,
        N_z,
        N_x,
        D,
      ),
    };
  }

  _dispatchFineAttentionTile(encoder, state, duty) {
    const operation = state.activeOperation;
    if (!operation || operation.ownerId !== duty.ownerId) {
      throw new Error(
        `attention tile ${duty.dutyId} requires active ${duty.ownerId}; `
        + `got ${operation?.ownerId ?? 'none'}`,
      );
    }
    if (operation.attention.tileCount !== duty.tileCount) {
      throw new Error(
        `attention tile count changed for ${duty.ownerId}: `
        + `${operation.attention.tileCount} != ${duty.tileCount}`,
      );
    }
    this._dispatchAttentionTile(encoder, operation.attention, duty.tileIndex);
  }

  _dispatchFineFuseFinish(encoder, state, duty) {
    const ownerId = `block-${duty.block}-fuse-${duty.direction}`;
    const operation = this._requireFineOperation(state, 'fuse', ownerId);
    const attnOutBuf = this._finishAttention(encoder, operation.attention);
    const z1Buf = createEmptyBuffer(this.device, operation.N_z * operation.D * 4);
    encoder.copyBufferToBuffer(
      operation.zBuf,
      0,
      z1Buf,
      0,
      operation.N_z * operation.D * 4,
    );
    this._dispatchAdd(encoder, z1Buf, attnOutBuf, operation.N_z * operation.D);
    const z2NormBuf = createEmptyBuffer(this.device, operation.N_z * operation.D * 4);
    this._dispatchLayerNorm(
      encoder,
      z1Buf,
      z2NormBuf,
      operation.weights.normZ2,
      operation.N_z,
      operation.D,
    );
    const ffnOutBuf = this._dispatchGEGLUFFN(
      encoder,
      z2NormBuf,
      operation.weights.ff,
      operation.N_z,
      operation.D,
    );
    const zOutBuf = createEmptyBuffer(this.device, operation.N_z * operation.D * 4);
    encoder.copyBufferToBuffer(
      z1Buf,
      0,
      zOutBuf,
      0,
      operation.N_z * operation.D * 4,
    );
    this._dispatchAdd(encoder, zOutBuf, ffnOutBuf, operation.N_z * operation.D);
    if (duty.direction === 'in') {
      state.currentLatent = zOutBuf;
    } else {
      state.currentTriplane = zOutBuf;
      this._diagnosticBuffers[`block${duty.block}_latent`] = state.currentLatent;
      this._diagnosticBuffers[`block${duty.block}_triplane`] = state.currentTriplane;
    }
    state.activeOperation = null;
  }

  _requireFineFuseOut(state, duty) {
    const ownerId = `block-${duty.block}-fuse-out`;
    return this._requireFineOperation(state, 'fuse', ownerId);
  }

  _dispatchFineLinearRange(
    encoder,
    operation,
    duty,
    phase,
    input,
    output,
    weight,
    bias,
    inDim,
    outDim,
  ) {
    if (duty.ownerId !== phase.ownerId) {
      throw new Error(
        `linear duty ${duty.dutyId} owner changed: `
        + `${duty.ownerId} != ${phase.ownerId}`,
      );
    }
    if (duty.rangeIndex !== phase.nextRangeIndex
      || duty.rangeCount !== phase.rangeCount
      || duty.totalRows !== operation.N_z) {
      throw new Error(
        `linear duty ${duty.dutyId} is out of range order; `
        + `expected ${phase.nextRangeIndex}/${phase.rangeCount} `
        + `over ${operation.N_z} rows`,
      );
    }
    const range = normalizeLinearRowRange(
      operation.N_z,
      duty.rowStart,
      duty.rowCount,
    );
    if (duty.rowStart !== phase.nextRowStart || duty.rowEnd !== range.rowEnd) {
      throw new Error(
        `linear duty ${duty.dutyId} is not contiguous; `
        + `expected row ${phase.nextRowStart}, got `
        + `${duty.rowStart}-${duty.rowEnd}`,
      );
    }
    this._dispatchLinearRange(
      encoder,
      input,
      output,
      weight,
      bias,
      operation.N_z,
      inDim,
      outDim,
      duty.rowStart,
      duty.rowCount,
    );
    phase.nextRangeIndex++;
    phase.nextRowStart = range.rowEnd;
  }

  _dispatchFineFuseAttentionLinearRange(encoder, state, duty) {
    const operation = this._requireFineFuseOut(state, duty);
    if (operation.attention.nextTileIndex !== operation.attention.tileCount) {
      throw new Error(
        `attention projection started before all tiles completed for ${operation.ownerId}`,
      );
    }
    if (!operation.attentionProjection) {
      operation.attentionProjection = {
        ownerId: `${operation.ownerId}-attention-projection`,
        nextRangeIndex: 0,
        nextRowStart: 0,
        rangeCount: duty.rangeCount,
        output: createEmptyBuffer(this.device, operation.N_z * operation.D * 4),
      };
    }
    const phase = operation.attentionProjection;
    this._dispatchFineLinearRange(
      encoder,
      operation,
      duty,
      phase,
      operation.attention.attnOutBuf,
      phase.output,
      operation.attention.attnWeights.proj.weight,
      operation.attention.attnWeights.proj.bias,
      operation.D,
      operation.D,
    );
  }

  _dispatchFineFuseResidualNorm(encoder, state, duty) {
    const operation = this._requireFineFuseOut(state, duty);
    const phase = operation.attentionProjection;
    if (!phase
      || phase.nextRangeIndex !== phase.rangeCount
      || phase.nextRowStart !== operation.N_z) {
      throw new Error(`attention projection is incomplete for ${operation.ownerId}`);
    }
    const byteLength = operation.N_z * operation.D * 4;
    operation.z1Buf = createEmptyBuffer(this.device, byteLength);
    encoder.copyBufferToBuffer(operation.zBuf, 0, operation.z1Buf, 0, byteLength);
    this._dispatchAdd(
      encoder,
      operation.z1Buf,
      phase.output,
      operation.N_z * operation.D,
    );
    operation.z2NormBuf = createEmptyBuffer(this.device, byteLength);
    this._dispatchLayerNorm(
      encoder,
      operation.z1Buf,
      operation.z2NormBuf,
      operation.weights.normZ2,
      operation.N_z,
      operation.D,
    );
  }

  _dispatchFineFuseGEGLULinearRange(encoder, state, duty) {
    const operation = this._requireFineFuseOut(state, duty);
    if (!operation.z2NormBuf) {
      throw new Error(`GEGLU expansion started before residual norm for ${operation.ownerId}`);
    }
    const innerDim = CONFIG.gegluInnerDim;
    if (!operation.gegluExpansion) {
      operation.gegluExpansion = {
        ownerId: `${operation.ownerId}-geglu-expansion`,
        nextRangeIndex: 0,
        nextRowStart: 0,
        rangeCount: duty.rangeCount,
        output: createEmptyBuffer(this.device, operation.N_z * 2 * innerDim * 4),
      };
    }
    const phase = operation.gegluExpansion;
    this._dispatchFineLinearRange(
      encoder,
      operation,
      duty,
      phase,
      operation.z2NormBuf,
      phase.output,
      operation.weights.ff.geglu.weight,
      operation.weights.ff.geglu.bias,
      operation.D,
      2 * innerDim,
    );
  }

  _dispatchFineFuseGEGLUActivate(encoder, state, duty) {
    const operation = this._requireFineFuseOut(state, duty);
    const phase = operation.gegluExpansion;
    if (!phase
      || phase.nextRangeIndex !== phase.rangeCount
      || phase.nextRowStart !== operation.N_z) {
      throw new Error(`GEGLU expansion is incomplete for ${operation.ownerId}`);
    }
    operation.gegluOutput = createEmptyBuffer(
      this.device,
      operation.N_z * CONFIG.gegluInnerDim * 4,
    );
    this._dispatchGEGLUActivation(
      encoder,
      phase.output,
      operation.gegluOutput,
      operation.N_z,
      CONFIG.gegluInnerDim,
    );
  }

  _dispatchFineFuseFFNLinearRange(encoder, state, duty) {
    const operation = this._requireFineFuseOut(state, duty);
    if (!operation.gegluOutput) {
      throw new Error(`FFN projection started before GEGLU activation for ${operation.ownerId}`);
    }
    if (!operation.ffnProjection) {
      operation.ffnProjection = {
        ownerId: `${operation.ownerId}-ffn-projection`,
        nextRangeIndex: 0,
        nextRowStart: 0,
        rangeCount: duty.rangeCount,
        output: createEmptyBuffer(this.device, operation.N_z * operation.D * 4),
      };
    }
    const phase = operation.ffnProjection;
    this._dispatchFineLinearRange(
      encoder,
      operation,
      duty,
      phase,
      operation.gegluOutput,
      phase.output,
      operation.weights.ff.proj.weight,
      operation.weights.ff.proj.bias,
      CONFIG.gegluInnerDim,
      operation.D,
    );
  }

  _dispatchFineFuseFinalResidual(encoder, state, duty) {
    const operation = this._requireFineFuseOut(state, duty);
    const phase = operation.ffnProjection;
    if (!phase
      || phase.nextRangeIndex !== phase.rangeCount
      || phase.nextRowStart !== operation.N_z) {
      throw new Error(`FFN projection is incomplete for ${operation.ownerId}`);
    }
    const byteLength = operation.N_z * operation.D * 4;
    const zOutBuf = createEmptyBuffer(this.device, byteLength);
    encoder.copyBufferToBuffer(operation.z1Buf, 0, zOutBuf, 0, byteLength);
    this._dispatchAdd(
      encoder,
      zOutBuf,
      phase.output,
      operation.N_z * operation.D,
    );
    state.currentTriplane = zOutBuf;
    this._diagnosticBuffers[`block${duty.block}_latent`] = state.currentLatent;
    this._diagnosticBuffers[`block${duty.block}_triplane`] = state.currentTriplane;
    state.activeOperation = null;
  }

  _dispatchFineBasicSelfPrepare(encoder, state, duty) {
    if (state.activeOperation != null) {
      throw new Error(`cannot prepare ${duty.dutyId} with active ${state.activeOperation.ownerId}`);
    }
    const D = CONFIG.dim;
    const weights = state.weights.mainBlocks[duty.block].transformerBlocks[duty.basic];
    const norm1Buf = createEmptyBuffer(this.device, state.N_latent * D * 4);
    this._dispatchLayerNorm(
      encoder,
      state.currentLatent,
      norm1Buf,
      weights.norm1,
      state.N_latent,
      D,
    );
    state.activeOperation = {
      kind: 'basic-self',
      ownerId: `block-${duty.block}-basic-${duty.basic}-self`,
      block: duty.block,
      basic: duty.basic,
      zBuf: state.currentLatent,
      weights,
      D,
      attention: this._createAttentionState(
        encoder,
        norm1Buf,
        norm1Buf,
        weights.attn1,
        state.N_latent,
        state.N_latent,
        D,
      ),
    };
  }

  _dispatchFineBasicCrossPrepare(encoder, state, duty) {
    const selfOwner = `block-${duty.block}-basic-${duty.basic}-self`;
    const operation = this._requireFineOperation(state, 'basic-self', selfOwner);
    const selfAttnBuf = this._finishAttention(encoder, operation.attention);
    const z1Buf = createEmptyBuffer(this.device, state.N_latent * operation.D * 4);
    encoder.copyBufferToBuffer(
      operation.zBuf,
      0,
      z1Buf,
      0,
      state.N_latent * operation.D * 4,
    );
    this._dispatchAdd(encoder, z1Buf, selfAttnBuf, state.N_latent * operation.D);
    const norm2Buf = createEmptyBuffer(this.device, state.N_latent * operation.D * 4);
    this._dispatchLayerNorm(
      encoder,
      z1Buf,
      norm2Buf,
      operation.weights.norm2,
      state.N_latent,
      operation.D,
    );
    operation.kind = 'basic-cross';
    operation.ownerId = `block-${duty.block}-basic-${duty.basic}-cross`;
    operation.z1Buf = z1Buf;
    operation.attention = this._createAttentionState(
      encoder,
      norm2Buf,
      state.imageTokensBuf,
      operation.weights.attn2,
      state.N_latent,
      state.N_img,
      operation.D,
    );
  }

  _dispatchFineBasicFinish(encoder, state, duty) {
    const ownerId = `block-${duty.block}-basic-${duty.basic}-cross`;
    const operation = this._requireFineOperation(state, 'basic-cross', ownerId);
    const crossAttnBuf = this._finishAttention(encoder, operation.attention);
    const z2Buf = createEmptyBuffer(this.device, state.N_latent * operation.D * 4);
    encoder.copyBufferToBuffer(
      operation.z1Buf,
      0,
      z2Buf,
      0,
      state.N_latent * operation.D * 4,
    );
    this._dispatchAdd(encoder, z2Buf, crossAttnBuf, state.N_latent * operation.D);
    const norm3Buf = createEmptyBuffer(this.device, state.N_latent * operation.D * 4);
    this._dispatchLayerNorm(
      encoder,
      z2Buf,
      norm3Buf,
      operation.weights.norm3,
      state.N_latent,
      operation.D,
    );
    const ffnOutBuf = this._dispatchGEGLUFFN(
      encoder,
      norm3Buf,
      operation.weights.ff,
      state.N_latent,
      operation.D,
    );
    const zOutBuf = createEmptyBuffer(this.device, state.N_latent * operation.D * 4);
    encoder.copyBufferToBuffer(
      z2Buf,
      0,
      zOutBuf,
      0,
      state.N_latent * operation.D * 4,
    );
    this._dispatchAdd(encoder, zOutBuf, ffnOutBuf, state.N_latent * operation.D);
    state.currentLatent = zOutBuf;
    state.activeOperation = null;
  }

  _dispatchForwardSetup(encoder, state) {
    const device = this.device;
    const D = CONFIG.dim;
    const N_tri = CONFIG.triplaneTokens;
    const N_latent_init = CONFIG.numLatents;
    const { imageTokensBuf, N_img, weights } = state;

    const gnOutBuf = this._dispatchGroupNorm(encoder, weights.tokenizer_embeddings_buf,
      weights.normTriplane, D, N_tri, 32);
    const triPermBuf = createEmptyBuffer(device, N_tri * D * 4);
    this._dispatchTranspose(encoder, gnOutBuf, triPermBuf, D, N_tri);
    const triProjBuf = createEmptyBuffer(device, N_tri * D * 4);
    this._dispatchLinear(encoder, triPermBuf, triProjBuf,
      weights.projTriplane.weight, weights.projTriplane.bias, N_tri, D, D);

    const imgNormBuf = createEmptyBuffer(device, N_img * D * 4);
    this._dispatchLayerNorm(encoder, imageTokensBuf, imgNormBuf, weights.normImage, N_img, D);
    const imgProjBuf = createEmptyBuffer(device, N_img * D * 4);
    this._dispatchLinear(encoder, imgNormBuf, imgProjBuf,
      weights.projImage.weight, weights.projImage.bias, N_img, D, D);

    const latentNormBuf = createEmptyBuffer(device, N_latent_init * D * 4);
    this._dispatchLayerNorm(encoder, weights.latentInit, latentNormBuf,
      weights.normLatent, N_latent_init, D);
    const latentProjBuf = createEmptyBuffer(device, N_latent_init * D * 4);
    this._dispatchLinear(encoder, latentNormBuf, latentProjBuf,
      weights.projLatent.weight, weights.projLatent.bias, N_latent_init, D, D);

    const N_latent = state.N_latent;
    const latentBuf = createEmptyBuffer(device, N_latent * D * 4);
    this._dispatchConcat(encoder, imgProjBuf, latentProjBuf, latentBuf,
      N_img * D, N_latent_init * D);

    state.currentLatent = latentBuf;
    state.currentTriplane = triProjBuf;
    Object.assign(this._diagnosticBuffers, {
      gnOutBuf,
      triPermBuf,
      triProjBuf,
      imgProjBuf,
      latentProjBuf,
      latentBuf,
    });
  }

  _dispatchForwardBlockStage(encoder, state, blockIndex, stageInBlock) {
    const D = CONFIG.dim;
    const N_tri = CONFIG.triplaneTokens;
    const block = state.weights.mainBlocks[blockIndex];

    if (stageInBlock === 0) {
      state.currentLatent = this._dispatchFuseBlock(
        encoder,
        state.currentLatent,
        state.currentTriplane,
        block.fuseBlockIn,
        state.N_latent,
        N_tri,
        D,
      );
      return;
    }

    if (stageInBlock <= CONFIG.numBasicBlocks) {
      const basicIndex = stageInBlock - 1;
      state.currentLatent = this._dispatchBasicBlock(
        encoder,
        state.currentLatent,
        state.imageTokensBuf,
        block.transformerBlocks[basicIndex],
        state.N_latent,
        state.N_img,
        D,
      );
      return;
    }

    state.currentTriplane = this._dispatchFuseBlock(
      encoder,
      state.currentTriplane,
      state.currentLatent,
      block.fuseBlockOut,
      N_tri,
      state.N_latent,
      D,
    );
    this._diagnosticBuffers[`block${blockIndex}_latent`] = state.currentLatent;
    this._diagnosticBuffers[`block${blockIndex}_triplane`] = state.currentTriplane;
  }

  _dispatchForwardFinal(encoder, state) {
    const device = this.device;
    const D = CONFIG.dim;
    const N_tri = CONFIG.triplaneTokens;
    const projOutBuf = createEmptyBuffer(device, N_tri * D * 4);
    this._dispatchLinear(encoder, state.currentTriplane, projOutBuf,
      state.weights.projOut.weight, state.weights.projOut.bias, N_tri, D, D);
    const projOutPermBuf = createEmptyBuffer(device, D * N_tri * 4);
    this._dispatchTranspose(encoder, projOutBuf, projOutPermBuf, N_tri, D);
    this._dispatchAdd(
      encoder,
      projOutPermBuf,
      state.weights.tokenizer_embeddings_buf,
      D * N_tri,
    );

    this._diagnosticBuffers['projOutBuf'] = projOutBuf;
    this._diagnosticBuffers['projOutPermBuf'] = projOutPermBuf;
    this._diagnosticBuffers['rearrangedEmb'] = state.weights.tokenizer_embeddings_buf;
    state.result = {
      buffer: projOutPermBuf,
      C: D,
      N: N_tri,
      planeSize: CONFIG.planeSize,
    };
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

  _createAttentionState(encoder, qInputBuf, kvInputBuf, attnWeights, N_q, N_kv, D) {
    const device = this.device;
    const numHeads = CONFIG.numHeads;
    const qBuf = createEmptyBuffer(device, N_q * D * 4);
    const kBuf = createEmptyBuffer(device, N_kv * D * 4);
    const vBuf = createEmptyBuffer(device, N_kv * D * 4);
    this._dispatchLinearNoBias(encoder, qInputBuf, qBuf, attnWeights.wq, N_q, D, D);
    this._dispatchLinearNoBias(encoder, kvInputBuf, kBuf, attnWeights.wk, N_kv, D, D);
    this._dispatchLinearNoBias(encoder, kvInputBuf, vBuf, attnWeights.wv, N_kv, D, D);

    const tileQCapacity = 128;
    const scoreBufSize = numHeads * tileQCapacity * N_kv * 4;
    const scoreBuf = createEmptyBuffer(device, scoreBufSize);
    const tileAttnOutBuf = createEmptyBuffer(device, tileQCapacity * D * 4);
    const attnOutBuf = createEmptyBuffer(device, N_q * D * 4);

    return {
      attnWeights,
      N_q,
      N_kv,
      D,
      numHeads,
      headDim: CONFIG.headDim,
      tileQCapacity,
      tileCount: ceilDiv(N_q, tileQCapacity),
      nextTileIndex: 0,
      qBuf,
      kBuf,
      vBuf,
      scoreBuf,
      tileAttnOutBuf,
      attnOutBuf,
    };
  }

  _dispatchAttentionTile(encoder, state, tileIndex) {
    if (tileIndex !== state.nextTileIndex) {
      throw new Error(
        `attention tile ${tileIndex} is out of order; expected ${state.nextTileIndex}`,
      );
    }
    if (tileIndex < 0 || tileIndex >= state.tileCount) {
      throw new RangeError(`invalid attention tile ${tileIndex}/${state.tileCount}`);
    }

    const device = this.device;
    const {
      N_q,
      N_kv,
      D,
      numHeads,
      headDim,
      tileQCapacity,
      qBuf,
      kBuf,
      vBuf,
      scoreBuf,
      tileAttnOutBuf,
      attnOutBuf,
    } = state;
    const qStart = tileIndex * tileQCapacity;
    const tileQ = Math.min(tileQCapacity, N_q - qStart);
    const qOffsetBytes = qStart * D * 4;
    const scoreSize = numHeads * tileQ * N_kv * 4;
    const outputSize = tileQ * D * 4;

    {
      const totalWG = ceilDiv(numHeads * tileQ * N_kv, WG_SIZE);
      const [wgX, wgY] = splitWG(totalWG);
      const params = this._cachedUniform(
        new Uint32Array([tileQ, N_kv, headDim, numHeads, wgX]),
      );
      const bg = device.createBindGroup({
        layout: this._crossAttnLayout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: qBuf, offset: qOffsetBytes, size: outputSize } },
          { binding: 2, resource: { buffer: kBuf } },
          { binding: 3, resource: { buffer: vBuf } },
          { binding: 4, resource: { buffer: scoreBuf, size: scoreSize } },
          { binding: 5, resource: { buffer: tileAttnOutBuf, size: outputSize } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.crossAttnScores);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    }

    {
      const totalWG = ceilDiv(numHeads * tileQ, WG_SIZE);
      const [wgX, wgY] = splitWG(totalWG);
      const params = this._cachedUniform(
        new Uint32Array([tileQ, N_kv, headDim, numHeads, wgX]),
      );
      const bg = device.createBindGroup({
        layout: this._crossAttnLayout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: qBuf, offset: qOffsetBytes, size: outputSize } },
          { binding: 2, resource: { buffer: kBuf } },
          { binding: 3, resource: { buffer: vBuf } },
          { binding: 4, resource: { buffer: scoreBuf, size: scoreSize } },
          { binding: 5, resource: { buffer: tileAttnOutBuf, size: outputSize } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.crossAttnSoftmax);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    }

    {
      const totalWG = ceilDiv(tileQ * numHeads * headDim, WG_SIZE);
      const [wgX, wgY] = splitWG(totalWG);
      const params = this._cachedUniform(
        new Uint32Array([tileQ, N_kv, headDim, numHeads, wgX]),
      );
      const bg = device.createBindGroup({
        layout: this._crossAttnLayout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: qBuf, offset: qOffsetBytes, size: outputSize } },
          { binding: 2, resource: { buffer: kBuf } },
          { binding: 3, resource: { buffer: vBuf } },
          { binding: 4, resource: { buffer: scoreBuf, size: scoreSize } },
          { binding: 5, resource: { buffer: tileAttnOutBuf, size: outputSize } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.crossAttnApply);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    }

    encoder.copyBufferToBuffer(tileAttnOutBuf, 0, attnOutBuf, qOffsetBytes, outputSize);
    state.nextTileIndex++;
  }

  _finishAttention(encoder, state) {
    if (state.nextTileIndex !== state.tileCount) {
      throw new Error(
        `attention is incomplete at tile ${state.nextTileIndex}/${state.tileCount}`,
      );
    }
    const projOutBuf = createEmptyBuffer(this.device, state.N_q * state.D * 4);
    this._dispatchLinear(
      encoder,
      state.attnOutBuf,
      projOutBuf,
      state.attnWeights.proj.weight,
      state.attnWeights.proj.bias,
      state.N_q,
      state.D,
      state.D,
    );
    return projOutBuf;
  }

  // --- Cross-attention dispatch (tiled over Q to fit WebGPU buffer limits) ---
  _dispatchCrossAttention(encoder, qInputBuf, kvInputBuf, attnWeights, N_q, N_kv, D) {
    const state = this._createAttentionState(
      encoder,
      qInputBuf,
      kvInputBuf,
      attnWeights,
      N_q,
      N_kv,
      D,
    );
    for (let tileIndex = 0; tileIndex < state.tileCount; tileIndex++) {
      this._dispatchAttentionTile(encoder, state, tileIndex);
    }
    return this._finishAttention(encoder, state);
  }

  _dispatchSelfAttention(encoder, inputBuf, attnWeights, N, D) {
    return this._dispatchCrossAttention(
      encoder,
      inputBuf,
      inputBuf,
      attnWeights,
      N,
      N,
      D,
    );
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
    this._dispatchGEGLUActivation(encoder, geGluProjBuf, geGluOutBuf, N, innerDim);

    // Linear: [N, innerDim] → [N, D]
    const ffnOutBuf = createEmptyBuffer(device, N * D * 4);
    this._dispatchLinear(encoder, geGluOutBuf, ffnOutBuf,
      ffWeights.proj.weight, ffWeights.proj.bias, N, innerDim, D);

    return ffnOutBuf;
  }

  _dispatchGEGLUActivation(encoder, input, output, rows, innerDim) {
    const totalWG = ceilDiv(rows * innerDim, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);
    const params = this._cachedUniform(new Uint32Array([rows, innerDim, wgX]));
    const bg = this.device.createBindGroup({
      layout: this.pipelines.geglu.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: output } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.geglu);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  // --- Low-level dispatch helpers ---

  _dispatchLinear(encoder, input, output, weight, bias, rows, inDim, outDim) {
    const totalWG = ceilDiv(rows * outDim, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);
    const params = this._cachedUniform(
      new Uint32Array([rows, inDim, outDim, wgX, 1]),
    );
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

  _dispatchLinearRange(
    encoder,
    input,
    output,
    weight,
    bias,
    totalRows,
    inDim,
    outDim,
    rowStart,
    rowCount,
  ) {
    const range = normalizeLinearRowRange(totalRows, rowStart, rowCount);
    const totalWG = ceilDiv(range.rowCount * outDim, WG_SIZE);
    const [wgX, wgY] = splitWG(totalWG);
    const params = this._cachedUniform(new Uint32Array([
      range.totalRows,
      inDim,
      outDim,
      range.rowStart,
      range.rowCount,
      wgX,
      1,
      0,
    ]));
    const bg = this.device.createBindGroup({
      layout: this.pipelines.linearRange.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: weight } },
        { binding: 3, resource: { buffer: bias } },
        { binding: 4, resource: { buffer: output } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.linearRange);
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
