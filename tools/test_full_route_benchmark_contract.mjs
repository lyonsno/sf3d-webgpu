#!/usr/bin/env node

import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  buildBenchmarkFailureReport,
  buildFullRouteArmReceipt,
  compareFullRouteArms,
  parsePositiveSafeIntegerOption,
  summarizeCounterbalancedPair,
  validateGlbPayload,
} from './full_route_benchmark_contract.mjs';

const frames = [
  { start: 0, end: 8, gap: 8 },
  { start: 8, end: 18, gap: 10 },
  { start: 18, end: 40, gap: 22 },
  { start: 40, end: 60, gap: 20 },
  { start: 60, end: 100, gap: 40 },
  { start: 100, end: 120, gap: 20 },
  { start: 120, end: 130, gap: 10 },
];

assert.deepEqual(
  parsePositiveSafeIntegerOption(undefined, { label: '--batch', defaultValue: 4096 }),
  { requested: null, effective: 4096, defaulted: true },
);
assert.deepEqual(
  parsePositiveSafeIntegerOption('16384', { label: '--batch', defaultValue: 4096 }),
  { requested: '16384', effective: 16384, defaulted: false },
);
for (const invalid of ['0', '-1', '1.5', 'NaN', 'nope', '', null, true]) {
  assert.throws(
    () => parsePositiveSafeIntegerOption(invalid, { label: '--batch', defaultValue: 4096 }),
    /positive safe integer/,
    `invalid requested batch ${String(invalid)} must fail loud`,
  );
}

const jsonText = JSON.stringify({ asset: { version: '2.0' } });
const jsonSource = new TextEncoder().encode(jsonText);
const jsonLength = Math.ceil(jsonSource.byteLength / 4) * 4;
const validGlb = new Uint8Array(12 + 8 + jsonLength);
const validGlbView = new DataView(validGlb.buffer);
validGlbView.setUint32(0, 0x46546c67, true);
validGlbView.setUint32(4, 2, true);
validGlbView.setUint32(8, validGlb.byteLength, true);
validGlbView.setUint32(12, jsonLength, true);
validGlbView.setUint32(16, 0x4e4f534a, true);
validGlb.fill(0x20, 20);
validGlb.set(jsonSource, 20);
assert.deepEqual(
  validateGlbPayload(validGlb),
  { byteLength: validGlb.byteLength, version: 2, chunkCount: 1 },
);
assert.throws(() => validateGlbPayload(new Uint8Array()), /complete 12-byte header/);
const headerOnlyGlb = validGlb.slice(0, 12);
new DataView(headerOnlyGlb.buffer).setUint32(8, 12, true);
assert.throws(
  () => validateGlbPayload(headerOnlyGlb),
  /JSON chunk/,
  'a header-only GLB must not be accepted as a complete artifact',
);
const partialGlb = validGlb.slice();
new DataView(partialGlb.buffer).setUint32(8, 1024, true);
assert.throws(() => validateGlbPayload(partialGlb), /declares 1024 bytes/);
const wrongFirstChunk = validGlb.slice();
new DataView(wrongFirstChunk.buffer).setUint32(16, 0x004e4942, true);
assert.throws(() => validateGlbPayload(wrongFirstChunk), /first chunk.*JSON/i);

const control = buildFullRouteArmReceipt({
  name: 'monolithic',
  ordinal: 1,
  options: { cooperativeDino: false },
  output: {
    totalMs: 100,
    glb: validGlb.buffer.slice(0),
    stageSpans: [
      { name: 'texture-bake', start: 10, end: 50 },
      { name: 'glb-export', start: 50, end: 110 },
    ],
  },
  frames,
  pipelineStartMs: 10,
  pipelineEndMs: 110,
});

assert.deepEqual(control.execution, {
  cooperativeDino: false,
  cooperativeBake: false,
  bakeSchedulingMode: 'monolithic',
  bakeBatchTexels: null,
  decoderArena: false,
  materializeWorker: false,
});
assert.equal(control.totalMs, 100);
assert.equal(control.pipelineIntervalMs, 100);
assert.equal(control.glbBytes, validGlb.byteLength);
assert.deepEqual(control.stageDurationsMs, {
  'texture-bake': 40,
  'glb-export': 60,
});
assert.deepEqual(control.cadence.wholeRoute, {
  frameIntervalCount: 5,
  p50Ms: 20,
  p95Ms: 40,
  p99Ms: 40,
  maxMs: 40,
  over16_7Count: 4,
  over33_3Count: 1,
  over100Count: 0,
});
assert.equal(control.cadence.byStage['texture-bake'].maxGapMs, 22);
assert.equal(control.cadence.byStage['texture-bake'].overlapMs, 40);
assert.equal(control.cadence.byStage['glb-export'].maxGapMs, 40);
assert.equal(control.cadence.byStage['glb-export'].overlapMs, 60);

const candidate = buildFullRouteArmReceipt({
  name: 'arena-plus-worker',
  ordinal: 2,
  options: {
    cooperativeDino: false,
    cooperativeBake: true,
    bakeSchedulingMode: 'cooperative',
    bakeBatchTexels: 4096,
    decoderArena: true,
    materializeWorker: {},
  },
  output: {
    totalMs: 110,
    glb: validGlb.buffer.slice(0),
    stageSpans: [
      { name: 'texture-bake', start: 10, end: 55 },
      { name: 'glb-export', start: 55, end: 120 },
    ],
  },
  frames,
  pipelineStartMs: 10,
  pipelineEndMs: 120,
});

assert.deepEqual(candidate.execution, {
  cooperativeDino: false,
  cooperativeBake: true,
  bakeSchedulingMode: 'cooperative',
  bakeBatchTexels: 4096,
  decoderArena: true,
  materializeWorker: true,
});

assert.deepEqual(
  compareFullRouteArms(control, candidate, { mechanismStage: 'texture-bake' }),
  {
    controlArm: 'monolithic',
    candidateArm: 'arena-plus-worker',
    mechanismStage: 'texture-bake',
    observedFullRouteDeltaMs: 10,
    observedFullRouteRatio: 1.1,
    mechanismStageDeltaMs: 5,
    mechanismStageRatio: 1.125,
    outsideMechanismObservedDeltaMs: 5,
    causalAuthority: 'mechanism-stage-only',
  },
);

assert.throws(
  () => buildFullRouteArmReceipt({
    name: 'missing-total',
    ordinal: 3,
    options: {},
    output: { glb: { byteLength: 1 }, stageSpans: [] },
    frames: [],
    pipelineStartMs: 0,
    pipelineEndMs: 1,
  }),
  /totalMs/,
);
assert.throws(
  () => buildFullRouteArmReceipt({
    name: 'blank-output',
    ordinal: 3,
    options: {},
    output: {
      totalMs: 10,
      glb: { byteLength: 0 },
      stageSpans: [{ name: 'texture-bake', start: 0, end: 10 }],
    },
    frames,
    pipelineStartMs: 0,
    pipelineEndMs: 10,
  }),
  /glb.*positive/i,
  'blank output must not enter a successful arm receipt',
);
assert.throws(
  () => buildFullRouteArmReceipt({
    name: 'cached-zero-wall',
    ordinal: 3,
    options: {},
    output: {
      totalMs: 0,
      glb: validGlb.buffer.slice(0),
      stageSpans: [{ name: 'texture-bake', start: 0, end: 1 }],
    },
    frames,
    pipelineStartMs: 0,
    pipelineEndMs: 1,
  }),
  /totalMs.*positive/i,
  'zero-duration cached output must not enter a successful arm receipt',
);
assert.throws(
  () => buildFullRouteArmReceipt({
    name: 'empty-route-interval',
    ordinal: 3,
    options: {},
    output: {
      totalMs: 1,
      glb: validGlb.buffer.slice(0),
      stageSpans: [{ name: 'texture-bake', start: 0, end: 1 }],
    },
    frames,
    pipelineStartMs: 1,
    pipelineEndMs: 1,
  }),
  /pipeline.*positive/i,
  'an empty measured route interval must not enter a successful arm receipt',
);
assert.throws(
  () => compareFullRouteArms(
    control,
    { ...candidate, stageDurationsMs: {} },
    { mechanismStage: 'texture-bake' },
  ),
  /texture-bake/,
);

const pairedArms = [
  {
    ...control,
    name: 'monolithic',
    ordinal: 1,
    totalMs: 100,
    stageDurationsMs: { ...control.stageDurationsMs, 'texture-bake': 40 },
    cadence: {
      ...control.cadence,
      byStage: {
        ...control.cadence.byStage,
        'texture-bake': {
          ...control.cadence.byStage['texture-bake'],
          maxAttributedOverlapMs: 20,
        },
      },
    },
  },
  {
    ...candidate,
    name: 'arena-plus-worker',
    ordinal: 2,
    totalMs: 112,
    stageDurationsMs: { ...candidate.stageDurationsMs, 'texture-bake': 45 },
    cadence: {
      ...candidate.cadence,
      byStage: {
        ...candidate.cadence.byStage,
        'texture-bake': {
          ...candidate.cadence.byStage['texture-bake'],
          maxAttributedOverlapMs: 8,
        },
      },
    },
  },
  {
    ...candidate,
    name: 'arena-plus-worker',
    ordinal: 3,
    totalMs: 108,
    stageDurationsMs: { ...candidate.stageDurationsMs, 'texture-bake': 43 },
    cadence: {
      ...candidate.cadence,
      byStage: {
        ...candidate.cadence.byStage,
        'texture-bake': {
          ...candidate.cadence.byStage['texture-bake'],
          maxAttributedOverlapMs: 10,
        },
      },
    },
  },
  {
    ...control,
    name: 'monolithic',
    ordinal: 4,
    totalMs: 104,
    stageDurationsMs: { ...control.stageDurationsMs, 'texture-bake': 42 },
    cadence: {
      ...control.cadence,
      byStage: {
        ...control.cadence.byStage,
        'texture-bake': {
          ...control.cadence.byStage['texture-bake'],
          maxAttributedOverlapMs: 22,
        },
      },
    },
  },
];

assert.deepEqual(
  summarizeCounterbalancedPair(pairedArms, {
    controlArm: 'monolithic',
    candidateArm: 'arena-plus-worker',
    mechanismStage: 'texture-bake',
  }),
  {
    design: 'A-B-B-A',
    episodeOrder: [
      'monolithic',
      'arena-plus-worker',
      'arena-plus-worker',
      'monolithic',
    ],
    controlArm: 'monolithic',
    candidateArm: 'arena-plus-worker',
    mechanismStage: 'texture-bake',
    controlEpisodes: {
      totalMs: [100, 104],
      mechanismStageMs: [40, 42],
      mechanismMaxAttributedGapMs: [20, 22],
    },
    candidateEpisodes: {
      totalMs: [112, 108],
      mechanismStageMs: [45, 43],
      mechanismMaxAttributedGapMs: [8, 10],
    },
    medians: {
      controlTotalMs: 102,
      candidateTotalMs: 110,
      observedFullRouteDeltaMs: 8,
      observedFullRouteRatio: 1.078431,
      controlMechanismStageMs: 41,
      candidateMechanismStageMs: 44,
      mechanismStageDeltaMs: 3,
      mechanismStageRatio: 1.073171,
      outsideMechanismObservedDeltaMs: 5,
      controlMechanismMaxAttributedGapMs: 21,
      candidateMechanismMaxAttributedGapMs: 9,
      mechanismCadenceRatio: 0.428571,
    },
    orderDrift: {
      controlLastMinusFirstMs: 4,
      candidateSecondMinusFirstMs: -4,
    },
    cadenceHypothesisSatisfied: true,
    causalAuthority: 'counterbalanced-mechanism-stage',
  },
);

const trustworthy = { schema: 'example.v1', arms: [{ totalMs: 123 }] };
assert.deepEqual(
  buildBenchmarkFailureReport('cadence did not collapse', 'acceptance', trustworthy),
  {
    ok: false,
    failurePhase: 'acceptance',
    error: 'cadence did not collapse',
    lastTrustworthyEvidence: trustworthy,
  },
);

const smokeSource = fs.readFileSync(
  new URL('./smoke_five_arm_ab.mjs', import.meta.url),
  'utf8',
);
assert.match(smokeSource, /buildFullRouteArmReceipt/, 'browser smoke must use the receipt contract');
assert.match(smokeSource, /compareFullRouteArms/, 'browser smoke must emit an authority-bounded comparison');
assert.match(smokeSource, /routeWall/, 'console headline must expose full-route wall time');
assert.match(smokeSource, /outside-texture-bake observed delta/, 'console must separate unrelated stage variance');
assert.match(smokeSource, /allArmsShareOneResidentModel:\s*true/, 'timing scope must state resident-model reuse');
assert.match(smokeSource, /visibilityStart !== 'visible'/, 'each episode must require visible-page cadence authority');
assert.match(smokeSource, /requestAnimationFrame\(t => \{\s*last = t;/, 'cadence probe must be primed before the route');
assert.match(smokeSource, /schema: 'sf3d\.raw-full-route-scheduling-episodes\.v1'/, 'raw episodes must be preservable before summary');
assert.match(smokeSource, /requestedBatch/, 'report must preserve requested batch separately from effective batch');
assert.match(smokeSource, /effectiveKit/, 'report must preserve effective imported kit identity');
assert.match(
  smokeSource,
  /browserExecutedKit/,
  'the same browser page that runs the route must witness its imported kit module identity',
);
assert.match(
  smokeSource,
  /__sf3dBenchmarkCheckpoint/,
  'each completed browser episode must checkpoint trustworthy evidence to Node',
);
assert.match(
  smokeSource,
  /expected-output-sha/,
  'portable paired execution must bind the known canonical output hash',
);
assert.ok(
  smokeSource.indexOf("schema: 'sf3d.raw-full-route-scheduling-episodes.v1'")
    < smokeSource.indexOf('summarizeCounterbalancedPair(withSha'),
  'raw episodes must enter last trustworthy evidence before paired summarization',
);

console.log('full-route benchmark contract passed');
