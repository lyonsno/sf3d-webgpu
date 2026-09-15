#!/usr/bin/env node
/**
 * Fail-first contract for the product-route witness report/acceptance
 * (tools/product_route_witness_report.mjs). A valid witness is accepted; each
 * falsifier is rejected with a specific error.
 */
import assert from 'node:assert/strict';
import {
  CANONICAL_DEMO_CHAIR_GLB_SHA256,
  PRODUCT_ROUTE_WITNESS_SCHEMA,
  acceptProductRouteWitness,
  assembleProductRouteWitness,
  attributeGapsToStages,
  percentile,
  summarizeFrameGaps,
} from './product_route_witness_report.mjs';

// --- Percentile: nearest rank ---
assert.equal(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50), 5);
assert.equal(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95), 10);
assert.equal(percentile([4], 99), 4);
assert.equal(percentile([], 50), null);
console.log('ok  nearest-rank percentile');

// --- Frame summary + attribution on a synthetic route ---
// Window [100, 1100). Frames every 8ms except one 120ms stall inside 'clip'.
const frames = [];
let t = 90;
while (t < 1200) {
  const gap = (t >= 500 && t < 620) ? 120 : 8;
  frames.push({ start: t, end: t + gap });
  t += gap;
}
const spans = [
  { name: 'dino', start: 100, end: 400 },
  { name: 'clip', start: 400, end: 700 },
  { name: 'bake', start: 700, end: 1100 },
];
const window = { startMs: 100, endMs: 1100 };
const summary = summarizeFrameGaps(frames, window);
assert.ok(summary.frameIntervalCount > 100);
assert.equal(summary.p50Ms, 8);
assert.equal(summary.maxMs, 120);
assert.equal(summary.over33_3, 1);
assert.equal(summary.over100, 1);
assert.equal(summary.worst[0].gapMs, 120);
const attribution = attributeGapsToStages(frames, spans, window);
assert.equal(attribution.byStage.clip.maxGapMs, 120);
assert.equal(attribution.byStage.clip.over33_3, 1);
assert.equal(attribution.byStage.dino.maxGapMs, 8);
assert.equal(attribution.rankedByMaxGap[0], 'clip');
assert.equal(attribution.byStage.bake.durationMs, 400);
// Frames outside the window are excluded (the 90→98 frame).
assert.equal(summarizeFrameGaps([{ start: 0, end: 50 }, { start: 50, end: 150 }], window).frameIntervalCount, 1);
assert.throws(() => summarizeFrameGaps([{ start: 1, end: NaN }], window), /finite ordered/);
assert.throws(() => summarizeFrameGaps(frames, { startMs: 5, endMs: 5 }), /endMs > startMs/);
console.log('ok  frame summary + stage attribution');

// --- A valid witness ---
function validInput(over = {}) {
  return {
    arm: 'product-default',
    source: { commit: 'abc123', dirty: false, kitVersion: '0.1.48' },
    requestedOptions: {
      cooperativeDino: true, cooperativeTwoStream: true, cooperativePostProcessor: true, cooperativeBake: true,
      decoderArena: true, postProcessorCompletionPolicy: 'bounded-prefix',
      workers: { preprocessWorker: true, clipPrepWorker: true, marchingTetWorker: true, uvUnwrapWorker: true, materializeWorker: true },
    },
    offloads: { preprocess: 'worker', clipPrep: 'worker', marchingTet: 'worker', uvUnwrap: 'worker', materialize: 'worker' },
    cooperativeReports: {
      'dinov2-tokenizer': { status: 'succeeded', schedulingMode: 'cooperative', inFlightGpuDutyCount: 0, progress: { completedItems: 24, totalItems: 24, percent: 100 } },
      'two-stream-backbone': { status: 'succeeded', schedulingMode: 'cooperative', inFlightGpuDutyCount: 0, progress: { completedItems: 2922, totalItems: 2922, percent: 100 } },
      'post-processor': { status: 'succeeded', schedulingMode: 'cooperative', completionPolicy: 'bounded-prefix', issuedGpuDutyCount: 702, retiredGpuDutyCount: 702, inFlightGpuDutyCount: 0, maxObservedInFlightGpuDuties: 2, progress: { completedItems: 702, totalItems: 702, percent: 100 } },
      'texture-bake': { status: 'succeeded', schedulingMode: 'cooperative', inFlightGpuDutyCount: 0, progress: { completedItems: 61, totalItems: 61, percent: 100 } },
    },
    cooperativeValidations: { 'post-processor': { ok: true, errors: [] } },
    materializationOffloaded: true,
    frames, stageSpans: spans, inferenceWindow: window, visibility: 'visible',
    output: { glbSha256: CANONICAL_DEMO_CHAIR_GLB_SHA256, glbBytes: 2511516, numVertices: 9988, numFaces: 19976 },
    contender: { enabled: true, submitted: 900, completed: 899, errors: [] },
    totalMs: 1000,
    ...over,
  };
}
const good = assembleProductRouteWitness(validInput());
assert.equal(good.schema, PRODUCT_ROUTE_WITNESS_SCHEMA);
assert.equal(good.effective.cooperative['post-processor'].maxObservedInFlightGpuDuties, 2);
const accepted = acceptProductRouteWitness(good, { requireContender: true });
assert.deepEqual([...accepted.errors], [], 'valid witness must be accepted');
assert.equal(accepted.ok, true);
// Budget gate: the 120ms stall breaches a 33ms budget but not a 200ms one.
assert.equal(acceptProductRouteWitness(good, { maxGapBudgetMs: 200 }).ok, true);
const budget = acceptProductRouteWitness(good, { maxGapBudgetMs: 33.3 });
assert.equal(budget.ok, false);
assert.match(budget.errors.join('\n'), /max frame gap 120ms exceeds budget 33.3ms/);
console.log('ok  valid witness accepted; budget gate works');

// --- Falsifiers ---
const falsifiers = [
  ['glb sha drift', validInput({ output: { glbSha256: 'deadbeef', glbBytes: 10, numVertices: 1, numFaces: 1 } }), /glb sha deadbeef != expected/],
  ['offload fell back to main', validInput({ offloads: { preprocess: 'worker', clipPrep: 'main', marchingTet: 'worker', uvUnwrap: 'worker', materialize: 'worker' } }), /clipPrepWorker requested but clipPrep ran on main/],
  ['materialize not offloaded per telemetry', validInput({ materializationOffloaded: false }), /materializationOffloaded/],
  ['missing cooperative report', validInput({ cooperativeReports: (() => { const r = validInput().cooperativeReports; delete r['two-stream-backbone']; return r; })() }), /cooperativeTwoStream requested but no cooperative report/],
  ['unsettled cooperative report', validInput({ cooperativeReports: { ...validInput().cooperativeReports, 'post-processor': { status: 'succeeded', schedulingMode: 'cooperative', issuedGpuDutyCount: 702, retiredGpuDutyCount: 700, inFlightGpuDutyCount: 2, progress: { completedItems: 702, totalItems: 702 } } } }), /post-processor left 2 GPU duties in flight/],
  ['incomplete cooperative progress', validInput({ cooperativeReports: { ...validInput().cooperativeReports, 'two-stream-backbone': { status: 'succeeded', schedulingMode: 'cooperative', inFlightGpuDutyCount: 0, progress: { completedItems: 2900, totalItems: 2922 } } } }), /two-stream-backbone progress 2900\/2922 incomplete/],
  ['no denominator-bearing progress', validInput({ cooperativeReports: { ...validInput().cooperativeReports, 'texture-bake': { status: 'succeeded', schedulingMode: 'cooperative', inFlightGpuDutyCount: 0 } } }), /texture-bake cooperative report carries no denominator-bearing progress/],
  ['disabled scheduling reported as cooperative', validInput({ cooperativeReports: { ...validInput().cooperativeReports, 'dinov2-tokenizer': { status: 'succeeded', schedulingMode: 'disabled', inFlightGpuDutyCount: 0, progress: { completedItems: 24, totalItems: 24 } } } }), /dinov2-tokenizer scheduling mode disabled != cooperative/],
  ['kit validation failed', validInput({ cooperativeValidations: { 'post-processor': { ok: false, errors: ['observedPrefixFenceCount 1 != submitted 702'] } } }), /post-processor kit validation failed: observedPrefixFenceCount/],
  ['no frames in window', validInput({ frames: [{ start: 0, end: 10 }] }), /only 0 frame intervals/],
  ['hidden page', validInput({ visibility: 'hidden' }), /page visibility was hidden/],
  ['dead contender', validInput({ contender: { enabled: true, submitted: 0, completed: 0, errors: [] } }), /contender enabled but completed zero/],
  ['contender errors', validInput({ contender: { enabled: true, submitted: 5, completed: 5, errors: ['No contender WebGPU adapter'] } }), /contender errors: No contender/],
];
for (const [name, input, re] of falsifiers) {
  const verdict = acceptProductRouteWitness(assembleProductRouteWitness(input), { requireContender: true });
  assert.equal(verdict.ok, false, `${name} must be rejected`);
  assert.match(verdict.errors.join('\n'), re, `${name} must name its reason`);
  console.log(`ok  rejects: ${name}`);
}
// Window never closed: assembly itself refuses.
assert.throws(() => assembleProductRouteWitness(validInput({ inferenceWindow: { startMs: 100, endMs: null } })), /inference window/);
// Contender not required → a baseline arm without one is fine.
assert.equal(acceptProductRouteWitness(assembleProductRouteWitness(validInput({ contender: { enabled: false } }))).ok, true);
console.log('ok  window-never-closed refused at assembly; baseline arm without contender accepted');

console.log('\nPRODUCT ROUTE WITNESS REPORT CONTRACT PASSED');
