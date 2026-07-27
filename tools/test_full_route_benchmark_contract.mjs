#!/usr/bin/env node

import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  buildBenchmarkFailureReport,
  buildFullRouteArmReceipt,
  compareFullRouteArms,
  summarizeCounterbalancedPair,
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

const control = buildFullRouteArmReceipt({
  name: 'monolithic',
  ordinal: 1,
  options: { cooperativeDino: false },
  output: {
    totalMs: 100,
    glb: { byteLength: 64 },
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
assert.equal(control.glbBytes, 64);
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
    glb: { byteLength: 64 },
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
assert.ok(
  smokeSource.indexOf("schema: 'sf3d.raw-full-route-scheduling-episodes.v1'")
    < smokeSource.indexOf('summarizeCounterbalancedPair(withSha'),
  'raw episodes must enter last trustworthy evidence before paired summarization',
);

console.log('full-route benchmark contract passed');
