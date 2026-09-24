#!/usr/bin/env node
/**
 * Fail-first contract for the product-route witness report/acceptance
 * (tools/product_route_witness_report.mjs). A valid witness is accepted; each
 * falsifier is rejected with a specific error.
 */
import assert from 'node:assert/strict';
import { SF3D_ROUTE_ID as SRC_ROUTE_ID, DINO_COOPERATIVE_MANIFEST_ID } from '../src/lib/cooperative_dino.js';
import { TWO_STREAM_MANIFEST_ID, TWO_STREAM_ATTENTION_MANIFEST_ID } from '../src/lib/cooperative_two_stream.js';
import { POST_PROCESSOR_MANIFEST_ID, POST_PROCESSOR_LAYER_MANIFEST_ID, POST_PROCESSOR_CHANNEL_MANIFEST_ID } from '../src/lib/cooperative_post_processor.js';
import { TEXTURE_BAKE_MANIFEST_ID } from '../src/lib/cooperative_texture_bake.js';
import {
  CANONICAL_DEMO_CHAIR_DUTY_COUNTS,
  CANONICAL_DEMO_CHAIR_GLB_SHA256,
  COOPERATIVE_MECHANISM_IDENTITY,
  PRODUCT_ROUTE_WITNESS_SCHEMA,
  SF3D_ROUTE_ID,
  expectedCooperativeIdentity,
  projectCooperativeReport,
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
    producerRoute: {
      invocation: 'direct-full-pipeline',
      deviceRelation: 'producer-device===window._sf3d_device',
      rendererDeviceRelationship: 'not-observed-by-this-witness',
      producer: {
        routeId: SF3D_ROUTE_ID, commit: 'abc123', kitVersion: '0.1.48', deviceInjected: true,
        deviceTopology: 'host-injected-device',
        backend: { kind: 'webgpu-local', runtime: 'browser', adapterName: 'Apple M4 Max', browser: 'Chrome', features: [], limits: { maxBufferSize: 1 }, timestampQuery: 'unavailable' },
      },
      runIdentity: null,
      browserKit: { packageName: '@kaminos/webgpu-inference-kit', exportedVersion: '0.1.48', exportFingerprint: 'a'.repeat(64), witnessModuleUrl: 'http://127.0.0.1:4173/tools/browser_kit_identity.js' },
    },
    requestedOptions: {
      cooperativeDino: true, cooperativeTwoStream: true, cooperativePostProcessor: true, cooperativeBake: true,
      decoderArena: true, twoStreamDutyGranularity: 'attention-tile', postProcessorDutyGranularity: 'channel-range', postProcessorCompletionPolicy: 'bounded-prefix',
      workers: { preprocessWorker: true, clipPrepWorker: true, marchingTetWorker: true, uvUnwrapWorker: true, materializeWorker: true },
    },
    offloads: { preprocess: 'worker', clipPrep: 'worker', marchingTet: 'worker', uvUnwrap: 'worker', materialize: 'worker' },
    cooperativeReports: {
      'dinov2-tokenizer': { status: 'succeeded', schedulingMode: 'cooperative', completionPolicy: 'strict-prefix', routeId: SF3D_ROUTE_ID, manifestId: 'sf3d.dino-encoder-cooperative-boundaries.v0', invocationId: 'sf3d:dino:cooperative', submittedGpuDutyCount: 24, inFlightGpuDutyCount: 0, progress: { completedItems: 24, totalItems: 24, percent: 100 } },
      'two-stream-backbone': { status: 'succeeded', schedulingMode: 'cooperative', completionPolicy: 'strict-prefix', routeId: SF3D_ROUTE_ID, manifestId: 'sf3d.two-stream-attention-cooperative-boundaries.v0', invocationId: 'sf3d:two-stream:cooperative', submittedGpuDutyCount: 2922, inFlightGpuDutyCount: 0, progress: { completedItems: 2922, totalItems: 2922, percent: 100 } },
      'post-processor': { status: 'succeeded', schedulingMode: 'cooperative', completionPolicy: 'bounded-prefix', routeId: SF3D_ROUTE_ID, manifestId: 'sf3d.post-processor-channel-cooperative-boundaries.v0', invocationId: 'sf3d:post-processor:cooperative', submittedGpuDutyCount: 702, issuedGpuDutyCount: 702, retiredGpuDutyCount: 702, inFlightGpuDutyCount: 0, maxObservedInFlightGpuDuties: 2, progress: { completedItems: 702, totalItems: 702, percent: 100 } },
      'texture-bake': { status: 'succeeded', schedulingMode: 'cooperative', completionPolicy: 'strict-prefix', routeId: SF3D_ROUTE_ID, manifestId: 'sf3d.texture-bake-cooperative-boundaries.v0', invocationId: 'sf3d:texture-bake:cooperative', submittedGpuDutyCount: 61, inFlightGpuDutyCount: 0, progress: { completedItems: 245837, totalItems: 245837, percent: 100 } },
    },
    cooperativeValidations: {
      'dinov2-tokenizer': { ok: true, errors: [] }, 'two-stream-backbone': { ok: true, errors: [] },
      'post-processor': { ok: true, errors: [] }, 'texture-bake': { ok: true, errors: [] },
    },
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
assert.ok(good.effective.producerRoute, 'assembly must preserve effective producer route identity');
assert.equal(good.effective.producerRoute.invocation, 'direct-full-pipeline');
assert.equal(good.effective.producerRoute.rendererDeviceRelationship, 'not-observed-by-this-witness');
const producerRunInput = validInput({ producerRoute: {
  ...validInput().producerRoute,
  invocation: 'producer.run',
  runIdentity: { schema: 'sf3d.producer-run-identity.v0', runId: 'witness:test', routeId: SF3D_ROUTE_ID, deviceTopology: 'host-injected-device', producerCommit: 'abc123', kitVersion: '0.1.48' },
} });
assert.equal(acceptProductRouteWitness(assembleProductRouteWitness(producerRunInput)).ok, true, 'producer.run identity must be accepted when it agrees with the producer');
const routeFalsifiers = [
  ['missing producer route identity', validInput({ producerRoute: null }), /producer route identity is missing/],
  ['wrong producer route', validInput({ producerRoute: { ...validInput().producerRoute, producer: { ...validInput().producerRoute.producer, routeId: 'other.route' } } }), /producer routeId other.route != expected/],
  ['fallback device mismatch', validInput({ producerRoute: { ...validInput().producerRoute, deviceRelation: 'mismatch' } }), /producer\/window device relation mismatch != producer-device===window\._sf3d_device/],
  ['renderer device overclaim', validInput({ producerRoute: { ...validInput().producerRoute, rendererDeviceRelationship: 'same-device' } }), /renderer device relationship same-device is outside this witness contract/],
  ['browser kit version mismatch', validInput({ producerRoute: { ...validInput().producerRoute, browserKit: { ...validInput().producerRoute.browserKit, exportedVersion: '0.1.47' } } }), /browser kit version 0.1.47 != installed kit version 0.1.48/],
  ['producer kit version mismatch', validInput({ producerRoute: { ...validInput().producerRoute, producer: { ...validInput().producerRoute.producer, kitVersion: '0.1.47' } } }), /producer kit version 0.1.47 != installed kit version 0.1.48/],
  ['missing served kit identity', validInput({ producerRoute: { ...validInput().producerRoute, browserKit: null } }), /browser-executed kit identity is missing/],
  ['producer-run route mismatch', validInput({ producerRoute: { ...producerRunInput.producerRoute, runIdentity: { ...producerRunInput.producerRoute.runIdentity, routeId: 'other.route' } } }), /producer run routeId other.route != producer routeId/],
  ['producer-run topology mismatch', validInput({ producerRoute: { ...producerRunInput.producerRoute, runIdentity: { ...producerRunInput.producerRoute.runIdentity, deviceTopology: 'producer-owned-device' } } }), /producer run deviceTopology producer-owned-device != producer deviceTopology/],
];
for (const [name, input, re] of routeFalsifiers) {
  const verdict = acceptProductRouteWitness(assembleProductRouteWitness(input));
  assert.equal(verdict.ok, false, `${name} must be rejected`);
  assert.match(verdict.errors.join('\n'), re, `${name} must name its reason`);
  console.log(`ok  rejects: ${name}`);
}
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

// --- Review 2026-09-16 HIGH: cooperative identity binding + mandatory kit validation ---
// The identity table must be the source modules' own constants, not copies that drift.
assert.equal(SF3D_ROUTE_ID, SRC_ROUTE_ID);
assert.equal(COOPERATIVE_MECHANISM_IDENTITY['dinov2-tokenizer'].manifestIds.default, DINO_COOPERATIVE_MANIFEST_ID);
assert.equal(COOPERATIVE_MECHANISM_IDENTITY['two-stream-backbone'].manifestIds['attention-tile'], TWO_STREAM_ATTENTION_MANIFEST_ID);
assert.equal(COOPERATIVE_MECHANISM_IDENTITY['two-stream-backbone'].manifestIds.stage, TWO_STREAM_MANIFEST_ID);
assert.equal(COOPERATIVE_MECHANISM_IDENTITY['post-processor'].manifestIds.plane, POST_PROCESSOR_MANIFEST_ID);
assert.equal(COOPERATIVE_MECHANISM_IDENTITY['post-processor'].manifestIds.layer, POST_PROCESSOR_LAYER_MANIFEST_ID);
assert.equal(COOPERATIVE_MECHANISM_IDENTITY['post-processor'].manifestIds['channel-range'], POST_PROCESSOR_CHANNEL_MANIFEST_ID);
assert.equal(COOPERATIVE_MECHANISM_IDENTITY['texture-bake'].manifestIds.default, TEXTURE_BAKE_MANIFEST_ID);
{
  const exp = expectedCooperativeIdentity(validInput().requestedOptions);
  assert.deepEqual(exp['two-stream-backbone'], { routeId: SF3D_ROUTE_ID, manifestId: TWO_STREAM_ATTENTION_MANIFEST_ID, invocationId: 'sf3d:two-stream:cooperative', schedulingMode: 'cooperative', completionPolicy: 'strict-prefix' });
  assert.deepEqual(exp['post-processor'], { routeId: SF3D_ROUTE_ID, manifestId: POST_PROCESSOR_CHANNEL_MANIFEST_ID, invocationId: 'sf3d:post-processor:cooperative', schedulingMode: 'cooperative', completionPolicy: 'bounded-prefix' });
  const stage = expectedCooperativeIdentity({ ...validInput().requestedOptions, twoStreamDutyGranularity: 'stage', postProcessorDutyGranularity: 'plane', postProcessorCompletionPolicy: 'strict-prefix' });
  assert.equal(stage['two-stream-backbone'].manifestId, TWO_STREAM_MANIFEST_ID);
  assert.equal(stage['post-processor'].manifestId, POST_PROCESSOR_MANIFEST_ID);
  assert.equal(stage['post-processor'].completionPolicy, 'strict-prefix');
  assert.equal(expectedCooperativeIdentity({ cooperativeDino: false, cooperativeTwoStream: false, cooperativePostProcessor: false, cooperativeBake: false })['dinov2-tokenizer'], undefined);
  console.log('ok  expected cooperative identity derives from requested granularity/policy');
}
const withReport = (key, patch) => validInput({ cooperativeReports: { ...validInput().cooperativeReports, [key]: { ...validInput().cooperativeReports[key], ...patch } } });
const identityFalsifiers = [
  ['wrong two-stream manifest', withReport('two-stream-backbone', { manifestId: 'wrong.manifest' }), /two-stream-backbone manifestId wrong.manifest != expected sf3d.two-stream-attention-cooperative-boundaries.v0/],
  ['wrong route id', withReport('dinov2-tokenizer', { routeId: 'other.route.v9' }), /dinov2-tokenizer routeId other.route.v9 != expected sf3d.image-to-mesh.webgpu-local.v0/],
  ['wrong invocation id', withReport('texture-bake', { invocationId: 'sf3d:texture-bake:disabled' }), /texture-bake invocationId sf3d:texture-bake:disabled != expected sf3d:texture-bake:cooperative/],
  ['missing manifest id', withReport('post-processor', { manifestId: undefined }), /post-processor manifestId missing != expected sf3d.post-processor-channel-cooperative-boundaries.v0/],
  ['wrong completion policy', withReport('post-processor', { completionPolicy: 'strict-prefix' }), /post-processor completionPolicy strict-prefix != expected bounded-prefix/],
  ['granularity mismatch', validInput({ requestedOptions: { ...validInput().requestedOptions, twoStreamDutyGranularity: 'stage' } }), /two-stream-backbone manifestId sf3d.two-stream-attention-cooperative-boundaries.v0 != expected sf3d.two-stream-cooperative-boundaries.v0/],
  ['all validations removed', validInput({ cooperativeValidations: {} }), /dinov2-tokenizer requested but no kit validation record/],
  ['two-stream validation missing', validInput({ cooperativeValidations: (() => { const v = { ...validInput().cooperativeValidations }; delete v['two-stream-backbone']; return v; })() }), /two-stream-backbone requested but no kit validation record/],
  ['texture-bake validation missing', validInput({ cooperativeValidations: (() => { const v = { ...validInput().cooperativeValidations }; delete v['texture-bake']; return v; })() }), /texture-bake requested but no kit validation record/],
  ['validation ok not boolean true', validInput({ cooperativeValidations: { ...validInput().cooperativeValidations, 'dinov2-tokenizer': { ok: 'true', errors: [] } } }), /dinov2-tokenizer kit validation failed/],
];
for (const [name, input, re] of identityFalsifiers) {
  const verdict = acceptProductRouteWitness(assembleProductRouteWitness(input), { requireContender: true });
  assert.equal(verdict.ok, false, `${name} must be rejected`);
  assert.match(verdict.errors.join('\n'), re, `${name} must name its reason`);
  console.log(`ok  rejects: ${name}`);
}
{
  // The reviewer's exact falsifier over a preserved receipt shape: wrong manifest + validations emptied.
  const tampered = validInput({ cooperativeValidations: {} });
  tampered.cooperativeReports = { ...tampered.cooperativeReports, 'two-stream-backbone': { ...tampered.cooperativeReports['two-stream-backbone'], manifestId: 'wrong.manifest' } };
  const verdict = acceptProductRouteWitness(assembleProductRouteWitness(tampered));
  assert.equal(verdict.ok, false);
  assert.ok(verdict.errors.length >= 5, 'names the manifest mismatch and every missing validation');
  console.log('ok  rejects: reviewer tamper (wrong manifest + no validations)');
}

// --- Same-device contender through the common foreground service ---
// The witness must carry the producer's foreground report and the contender's
// receipt breakdown, and refuse a run whose foreground demand was left
// pending/active or whose host frames failed/canceled.
const fgReport = {
  status: 'succeeded', requestCount: 120, receiptCount: 120, pendingRequestCount: 0, activeRequestCount: 0,
  noDemandBoundaryCount: 3000,
};
const sameDeviceContender = {
  enabled: true, mode: 'same-device-foreground-opportunity', submitted: 120, completed: 120, errors: [],
  receipts: { completed: 120, failed: 0, canceled: 0, outsideRun: 0, schedulerBoundary: 100, foregroundWindow: 18, runFinish: 2 },
};
const withFg = assembleProductRouteWitness(validInput({ foregroundOpportunities: fgReport, contender: sameDeviceContender }));
assert.equal(withFg.foregroundOpportunities.status, 'succeeded');
assert.equal(withFg.contender.mode, 'same-device-foreground-opportunity');
assert.equal(withFg.contender.receipts.schedulerBoundary, 100);
assert.deepEqual([...acceptProductRouteWitness(withFg, { requireContender: true }).errors], []);
// Without a foreground report the field is absent and a second-device contender still passes.
assert.equal(good.foregroundOpportunities, null);
const fgFalsifiers = [
  ['foreground demand left unsettled', validInput({ foregroundOpportunities: { ...fgReport, status: 'incomplete', pendingRequestCount: 2 }, contender: sameDeviceContender }), /foreground opportunity report status incomplete/],
  ['same-device contender without foreground report', validInput({ contender: sameDeviceContender }), /same-device contender requires a foreground opportunity report/],
  ['host frames failed', validInput({ foregroundOpportunities: fgReport, contender: { ...sameDeviceContender, receipts: { ...sameDeviceContender.receipts, failed: 3 } } }), /contender receipts: 3 failed/],
  ['host frames canceled', validInput({ foregroundOpportunities: fgReport, contender: { ...sameDeviceContender, receipts: { ...sameDeviceContender.receipts, canceled: 1 } } }), /contender receipts: 1 canceled/],
];
for (const [name, input, re] of fgFalsifiers) {
  const verdict = acceptProductRouteWitness(assembleProductRouteWitness(input), { requireContender: true });
  assert.equal(verdict.ok, false, `${name} must be rejected`);
  assert.match(verdict.errors.join('\n'), re, `${name} must name its reason`);
  console.log(`ok  rejects: ${name}`);
}

// --- Review 2026-09-16 P1: same-device accounting equalities ---
// The witness must refuse a same-device arm whose counts do not add up: every
// submission accounted for, every completion located, in-run receipts equal to
// the producer's foreground request/receipt counts, and at least one host frame
// actually serviced inside the run.
{
  const adversarial = validInput({
    foregroundOpportunities: { ...fgReport, requestCount: 0, receiptCount: 0 },
    contender: { ...sameDeviceContender, submitted: 100, completed: 1, receipts: { completed: 1, failed: 0, canceled: 0, outsideRun: 1, schedulerBoundary: 0, foregroundWindow: 0, runFinish: 0 } },
  });
  const verdict = acceptProductRouteWitness(assembleProductRouteWitness(adversarial), { requireContender: true });
  assert.equal(verdict.ok, false, 'malformed same-device accounting must be rejected');
  assert.match(verdict.errors.join('\n'), /contender completed 1 != submitted 100/);
  assert.match(verdict.errors.join('\n'), /serviced zero host frames inside the run/);
  console.log('ok  rejects: same-device arm with 99 vanished submissions and only outside-run service');
}
const accountingFalsifiers = [
  ['completed != submitted', { ...sameDeviceContender, completed: 119 }, /contender completed 119 != submitted 120/],
  ['receipt statuses do not sum to submitted', { ...sameDeviceContender, receipts: { ...sameDeviceContender.receipts, completed: 119 } }, /contender receipts 119\+0\+0 != submitted 120/],
  ['service locations do not sum to completed', { ...sameDeviceContender, receipts: { ...sameDeviceContender.receipts, foregroundWindow: 17 } }, /service locations 119 != completed 120/],
];
for (const [name, contender, re] of accountingFalsifiers) {
  const verdict = acceptProductRouteWitness(assembleProductRouteWitness(validInput({ foregroundOpportunities: fgReport, contender })), { requireContender: true });
  assert.equal(verdict.ok, false, `${name} must be rejected`);
  assert.match(verdict.errors.join('\n'), re, `${name} must name its reason`);
  console.log(`ok  rejects: ${name}`);
}
{
  // In-run receipts (100+18+2 = 120) must equal the common service request/receipt counts.
  const fg = { ...fgReport, requestCount: 119, receiptCount: 119 };
  const verdict = acceptProductRouteWitness(assembleProductRouteWitness(validInput({ foregroundOpportunities: fg, contender: sameDeviceContender })), { requireContender: true });
  assert.match(verdict.errors.join('\n'), /foreground requestCount 119 != in-run contender receipts 120/);
  console.log('ok  rejects: in-run receipts disagree with the producer foreground report');
}

// --- Review 2026-09-16 P2: measured duty counts preserved and pinned ---
{
  const raw = { status: 'succeeded', schedulingMode: 'cooperative', submittedGpuDutyCount: 61, gpuDuties: new Array(61).fill({}), inFlightGpuDutyCount: 0, progress: { completedItems: 245837, totalItems: 245837 } };
  const once = projectCooperativeReport(raw);
  const twice = projectCooperativeReport(once);
  assert.equal(once.submittedGpuDutyCount, 61, 'kit submitted duty count preserved');
  assert.equal(once.gpuDutyCount, 61);
  assert.deepEqual(twice, once, 'projection is idempotent (the browser projects once, assembly projects again)');
  console.log('ok  projection preserves submittedGpuDutyCount and is idempotent');

  assert.deepEqual(CANONICAL_DEMO_CHAIR_DUTY_COUNTS, { 'dinov2-tokenizer': 24, 'two-stream-backbone': 2922, 'post-processor': 702, 'texture-bake': 61 });
  const pinned = acceptProductRouteWitness(good, { expectedDutyCounts: CANONICAL_DEMO_CHAIR_DUTY_COUNTS });
  assert.deepEqual([...pinned.errors], [], 'canonical duty counts accepted');
  const drifted = assembleProductRouteWitness(validInput({ cooperativeReports: { ...validInput().cooperativeReports, 'texture-bake': { ...validInput().cooperativeReports['texture-bake'], submittedGpuDutyCount: 60 } } }));
  assert.match(acceptProductRouteWitness(drifted, { expectedDutyCounts: CANONICAL_DEMO_CHAIR_DUTY_COUNTS }).errors.join('\n'), /texture-bake submitted 60 GPU duties, expected 61/);
  const missing = assembleProductRouteWitness(validInput({ cooperativeReports: { ...validInput().cooperativeReports, 'texture-bake': { status: 'succeeded', schedulingMode: 'cooperative', inFlightGpuDutyCount: 0, progress: { completedItems: 245837, totalItems: 245837 } } } }));
  assert.match(acceptProductRouteWitness(missing, { expectedDutyCounts: CANONICAL_DEMO_CHAIR_DUTY_COUNTS }).errors.join('\n'), /texture-bake preserves no submitted GPU duty count/);
  console.log('ok  pinned duty counts: canonical accepted, drift and missing count rejected');
}

console.log('\nPRODUCT ROUTE WITNESS REPORT CONTRACT PASSED');
