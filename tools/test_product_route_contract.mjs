#!/usr/bin/env node
/**
 * Contract: the product route composes EVERY proven foreground-liveness
 * mechanism by default, with the settings the A/B harnesses measured, and
 * fails loud on a misspelled worker role (which would silently fall back to
 * the main-thread path and reintroduce the gap the worker removes).
 */
import assert from 'node:assert/strict';
import {
  PRODUCT_ROUTE_DEFAULTS,
  PRODUCT_ROUTE_WORKER_ROLES,
  createProductRouteOptions,
  describeProductRouteOptions,
} from '../src/lib/product_route.js';

// 1. Every mechanism on, with the harness-measured settings.
assert.equal(PRODUCT_ROUTE_DEFAULTS.cooperativeDino, true);
assert.equal(PRODUCT_ROUTE_DEFAULTS.dinoSchedulingMode, 'cooperative');
assert.equal(PRODUCT_ROUTE_DEFAULTS.dinoChunkBlocks, 1);
assert.equal(PRODUCT_ROUTE_DEFAULTS.cooperativeTwoStream, true);
assert.equal(PRODUCT_ROUTE_DEFAULTS.twoStreamDutyGranularity, 'attention-tile');
assert.equal(PRODUCT_ROUTE_DEFAULTS.twoStreamLinearRowsPerDuty, 256, 'clean-assay Pareto profile');
assert.equal(PRODUCT_ROUTE_DEFAULTS.cooperativePostProcessor, true);
assert.equal(PRODUCT_ROUTE_DEFAULTS.postProcessorDutyGranularity, 'channel-range');
assert.equal(PRODUCT_ROUTE_DEFAULTS.postProcessorChannelsPerDuty, 16);
assert.equal(PRODUCT_ROUTE_DEFAULTS.postProcessorCompletionPolicy, 'bounded-prefix');
assert.equal(PRODUCT_ROUTE_DEFAULTS.postProcessorMaxInFlightGpuDuties, 2);
assert.equal(PRODUCT_ROUTE_DEFAULTS.cooperativeBake, true);
assert.equal(PRODUCT_ROUTE_DEFAULTS.bakeBatchTexels, 4096);
assert.equal(PRODUCT_ROUTE_DEFAULTS.decoderArena, true);
assert.ok(Object.isFrozen(PRODUCT_ROUTE_DEFAULTS), 'defaults must be frozen');
console.log('ok  defaults enable every mechanism with the measured settings');

// 2. All five CPU offload roles are declared, each pointing at a worker module.
assert.deepEqual(
  Object.keys(PRODUCT_ROUTE_WORKER_ROLES).sort(),
  ['clipPrepWorker', 'marchingTetWorker', 'materializeWorker', 'preprocessWorker', 'uvUnwrapWorker'],
);
for (const [role, rel] of Object.entries(PRODUCT_ROUTE_WORKER_ROLES)) {
  assert.match(rel, /_worker\.js$/, `${role} must point at a *_worker.js module`);
}
console.log('ok  five worker roles declared');

// 3. Composition: workers fill their option names; overrides win; result frozen.
const fake = { preprocessWorker: { fake: 'pp' }, uvUnwrapWorker: { fake: 'uv' } };
const options = createProductRouteOptions({ workers: fake, overrides: { bakeBatchTexels: 8192 } });
assert.equal(options.preprocessWorker, fake.preprocessWorker);
assert.equal(options.uvUnwrapWorker, fake.uvUnwrapWorker);
assert.equal(options.materializeWorker, undefined);
assert.equal(options.bakeBatchTexels, 8192);
assert.equal(options.cooperativeDino, true);
assert.ok(Object.isFrozen(options));
console.log('ok  workers and overrides compose');

// 4. Fail-loud on unknown role (would silently degrade to main thread).
assert.throws(
  () => createProductRouteOptions({ workers: { preprocesWorker: {} } }),
  /unknown product route worker role: preprocesWorker/,
);
console.log('ok  unknown worker role rejected');

// 5. Effective description reflects what will actually run, not the request.
const described = describeProductRouteOptions(options);
assert.equal(described.cooperativeDino, true);
assert.equal(described.cooperativeTwoStream, true);
assert.equal(described.twoStreamDutyGranularity, 'attention-tile');
assert.equal(described.postProcessorCompletionPolicy, 'bounded-prefix');
assert.equal(described.decoderArena, true);
assert.equal(described.bakeBatchTexels, 8192);
assert.deepEqual(described.workers, {
  preprocessWorker: true, clipPrepWorker: false, marchingTetWorker: false,
  uvUnwrapWorker: true, materializeWorker: false,
});
const disabled = describeProductRouteOptions(createProductRouteOptions({
  overrides: { dinoSchedulingMode: 'disabled', cooperativeBake: false },
}));
assert.equal(disabled.cooperativeDino, false, 'disabled scheduling mode is not cooperative');
assert.equal(disabled.cooperativeBake, false);
assert.equal(disabled.decoderArena, false, 'arena requires cooperative bake');
assert.equal(disabled.bakeBatchTexels, null);
console.log('ok  effective description distinguishes requested from effective');

console.log('\nPRODUCT ROUTE CONTRACT PASSED');
