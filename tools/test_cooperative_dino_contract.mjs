/**
 * Contract test: cooperative DINO boundary preserves the exact legacy dispatch
 * sequence, and drives it through the Kaminos cooperative execution facade with
 * correct boundary/range/progress/fence semantics.
 *
 * This does NOT require a real GPU. The invariant that determines SF3D numerical
 * parity is: the ordered sequence of neural dispatches (and their source/target
 * buffers via the ping-pong) must be identical between the legacy single-submit
 * encode() and the cooperative per-chunk encodeCooperative(). We capture that
 * sequence by stubbing the tokenizer's _dispatch* helpers to record calls, and
 * assert legacy === cooperative regardless of chunk size. We also assert the
 * command-buffer CUT points land exactly on block boundaries (never mid-block).
 *
 * Run: node tools/test_cooperative_dino_contract.mjs
 */
import assert from 'node:assert/strict';

import { SF3DImageTokenizer } from '../src/lib/sf3d_backbone.js';
import {
  runCooperativeDino,
  defineDinoEncoderManifest,
  DINO_BOUNDARY_ID,
  SF3D_ROUTE_ID,
} from '../src/lib/cooperative_dino.js';

const NUM_BLOCKS = 24;
const D = 1024;

// --- Recording stubs -------------------------------------------------------
// A tokenizer instance whose GPU-touching helpers are replaced by recorders.
// We tag each recorded op with the encoder it was recorded into so we can also
// verify command-buffer cut points.
function createRecordingTokenizer(log, currentEncoderRef) {
  const tok = Object.create(SF3DImageTokenizer.prototype);
  // Buffers are opaque tokens; identity is all that matters for ping-pong checks.
  let bufSeq = 0;
  const mkBuf = (tag) => ({ __buf: tag, id: bufSeq++ });
  // createEmptyBuffer is imported inside sf3d_backbone.js from gpu.js, which
  // touches GPUBufferUsage. We can't call the real one in Node, so we stub the
  // device with a createBuffer that returns opaque tokens and shim the global.
  globalThis.GPUBufferUsage = globalThis.GPUBufferUsage || {
    STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8, UNIFORM: 16,
  };
  tok.device = {
    createBuffer: () => mkBuf('empty'),
  };
  tok._uniformCache = new Map();

  const rec = (op) => (...args) => {
    // Record op name + the source/dest buffer identities that carry data flow.
    // For _dispatch* helpers the signature is (encoder, input, output, ...).
    const enc = args[0];
    assert.equal(enc, currentEncoderRef.value, `${op} recorded into unexpected encoder`);
    const input = args[1]?.__buf ? args[1].id : null;
    const output = args[2]?.__buf ? args[2].id : null;
    log.push({ op, input, output, encoder: enc.__id });
  };

  for (const name of [
    '_dispatchPatchEmbed', '_dispatchSiLU', '_dispatchLinear', '_dispatchModulatedLN',
    '_dispatchAttnScores', '_dispatchAttnSoftmax', '_dispatchAttnApply',
    '_dispatchLayerScaleResidual', '_dispatchLinearGelu', '_dispatchLayerNorm',
  ]) {
    tok[name] = rec(name);
  }
  return tok;
}

function fakeEncoder(id) {
  return {
    __id: id,
    copyBufferToBuffer() { /* diagnostics only; not data-flow relevant */ },
    finish() { return { __commandBuffer: id }; },
  };
}

// --- 1. Manifest shape -----------------------------------------------------
{
  const manifest = defineDinoEncoderManifest(NUM_BLOCKS, 1);
  assert.equal(manifest.routeId, SF3D_ROUTE_ID);
  assert.equal(manifest.phases.length, 1);
  const boundary = manifest.phases[0].boundaries[0];
  assert.equal(boundary.boundaryId, DINO_BOUNDARY_ID);
  assert.equal(boundary.kind, 'gpu-command');
  assert.equal(boundary.totalItems, NUM_BLOCKS);
  assert.equal(boundary.chunking.mode, 'fixed');
  assert.equal(boundary.chunking.chunkItems, 1);
  // No hidden cap: totalItems is exactly the declared block count.
  assert.throws(() => defineDinoEncoderManifest(0, 1), /positive safe integer/);
  assert.throws(() => defineDinoEncoderManifest(NUM_BLOCKS, 0), /positive safe integer/);
  console.log('ok  manifest shape');
}

// --- Legacy dispatch sequence (reference) ----------------------------------
function recordLegacy() {
  const log = [];
  const encRef = { value: null };
  const tok = createRecordingTokenizer(log, encRef);
  const weights = makeFakeWeights();
  const encoder = fakeEncoder('legacy');
  encRef.value = encoder;
  const ctx = tok._setupEncode(encoder, { __buf: 'image', id: -1 }, { __buf: 'cam', id: -2 }, weights);
  for (let l = 0; l < NUM_BLOCKS; l++) tok._encodeBlock(encoder, l, ctx, weights);
  tok._finalizeEncode(encoder, ctx, weights);
  return log;
}

function makeFakeWeights() {
  const lin = () => ({ weight: { __buf: 'w', id: 900 }, bias: { __buf: 'b', id: 901 } });
  const block = {
    norm1Mod: lin(), norm2Mod: lin(), norm1: lin(), norm2: lin(),
    attn: { q: lin(), k: lin(), v: lin(), proj: lin() },
    mlp: { fc1: lin(), fc2: lin() },
    layerScale1: { __buf: 'ls', id: 902 }, layerScale2: { __buf: 'ls', id: 903 },
  };
  return { blocks: Array.from({ length: NUM_BLOCKS }, () => block), layernorm: lin() };
}

// --- 2. Cooperative dispatch sequence === legacy, for several chunk sizes ---
async function recordCooperative(chunkBlocks, schedulingMode) {
  const log = [];
  const encRef = { value: null };
  const tok = createRecordingTokenizer(log, encRef);
  const weights = makeFakeWeights();

  const calls = [];
  let encSeq = 0;
  const device = {
    createCommandEncoder() {
      const e = fakeEncoder(`coop-${encSeq++}`);
      encRef.value = e;
      return e;
    },
    queue: {
      submit(buffers) { calls.push(`submit:${buffers[0].__commandBuffer}`); },
      onSubmittedWorkDone() { calls.push('fence'); return Promise.resolve(); },
    },
  };

  const progressEvents = [];
  const { result, report } = await runCooperativeDino({
    device,
    tokenizer: tok,
    imageBuf: { __buf: 'image', id: -1 },
    cameraEmbedBuf: { __buf: 'cam', id: -2 },
    weights,
    numBlocks: NUM_BLOCKS,
    chunkBlocks,
    schedulingMode,
    onProgress: (p) => progressEvents.push(p),
    invocationId: `test:${schedulingMode}:${chunkBlocks}`,
  });
  return { log, calls, report, progressEvents, result };
}

const legacyLog = recordLegacy();
assert.ok(legacyLog.length > 24 * 10, 'legacy log should have many dispatches');
console.log(`ok  legacy dispatch sequence recorded (${legacyLog.length} ops)`);

for (const chunkBlocks of [1, 2, 3, 4, 8, 24]) {
  const { log, calls, report, progressEvents } = await recordCooperative(chunkBlocks, 'cooperative');

  // (a) EXACT dispatch-sequence parity (op name + input/output buffer identity).
  const strip = (entries) => entries.map(({ op, input, output }) => `${op}:${input}->${output}`);
  assert.deepEqual(strip(log), strip(legacyLog),
    `cooperative(chunk=${chunkBlocks}) dispatch sequence must equal legacy`);

  // (b) Command-buffer cut points land on block boundaries. Setup ops share the
  //     first chunk's encoder; each subsequent encoder starts with a block's
  //     first op (norm1Mod linear -> _dispatchLinear). Count distinct encoders.
  const encoders = [...new Set(log.map(e => e.encoder))];
  const expectedChunks = Math.ceil(NUM_BLOCKS / chunkBlocks);
  assert.equal(encoders.length, expectedChunks,
    `chunk=${chunkBlocks} should cut into ${expectedChunks} command buffers, got ${encoders.length}`);

  // (c) Facade report: 24 items over 1 boundary, correct fence authority.
  assert.equal(report.schedulingMode, 'cooperative');
  assert.equal(report.status, 'succeeded');
  assert.equal(report.queueCompletionAuthority, 'per-gpu-duty-prefix-fence');
  const b = report.boundaries[0];
  assert.equal(b.boundaryId, DINO_BOUNDARY_ID);
  assert.equal(b.completedItems, NUM_BLOCKS);
  assert.equal(b.totalItems, NUM_BLOCKS);
  assert.equal(b.actualRangeCount, expectedChunks);
  assert.equal(report.progress.progress, 1);
  assert.equal(report.progress.percent, 100);

  // (d) Denominator-bearing progress with no null totals.
  assert.ok(progressEvents.length > 0);
  for (const p of progressEvents) {
    assert.equal(p.totalItems, NUM_BLOCKS, 'progress must always carry the denominator');
    assert.ok(p.completedItems <= NUM_BLOCKS);
  }
  const last = progressEvents[progressEvents.length - 1];
  assert.equal(last.completedItems, NUM_BLOCKS);

  // (e) Cooperative arm: one submit + one per-duty fence per chunk.
  assert.equal(calls.filter(c => c.startsWith('submit:')).length, expectedChunks);
  assert.equal(calls.filter(c => c === 'fence').length, expectedChunks);

  console.log(`ok  cooperative chunk=${chunkBlocks}: parity + ${expectedChunks} cuts + fences`);
}

// --- 3. Disabled A/B: identical coverage, ONE terminal fence, no per-duty ---
{
  const { log, calls, report } = await recordCooperative(1, 'disabled');
  const strip = (entries) => entries.map(({ op, input, output }) => `${op}:${input}->${output}`);
  assert.deepEqual(strip(log), strip(legacyLog),
    'disabled arm must run identical declared work to legacy');
  assert.equal(report.schedulingMode, 'disabled');
  assert.equal(report.status, 'succeeded');
  assert.equal(report.queueCompletionAuthority, 'one-terminal-prefix-fence');
  // Disabled: 24 submits (still per-range) but exactly ONE terminal fence.
  assert.equal(calls.filter(c => c.startsWith('submit:')).length, NUM_BLOCKS);
  assert.equal(calls.filter(c => c === 'fence').length, 1,
    'disabled A/B must take exactly one terminal queue fence');
  console.log('ok  disabled A/B: identical work, single terminal fence');
}

console.log('\nALL COOPERATIVE DINO CONTRACT CHECKS PASSED');
