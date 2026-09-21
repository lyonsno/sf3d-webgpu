#!/usr/bin/env node
/**
 * Consumer contract for the common persistent foreground service.
 *
 * `beginRun` is asynchronous: it reserves run admission before draining
 * already-admitted outside-run frames.  The producer must await that boundary
 * before it exposes its next GPU encode, retain the kit's boundary adapter,
 * and forward the explicit CPU/worker window without inventing a timer drain.
 */
import assert from 'node:assert/strict';
import { createProducerLifecycle, prepareProducerRun } from '../src/lib/producer_lifecycle.js';
import { withForegroundScope } from '../src/lib/foreground_scope.js';

const events = [];
const lifecycle = createProducerLifecycle({ release() {} });
const foreground = {
  async beginRun(runId) {
    events.push(`begin:${runId}`);
    await Promise.resolve();
    events.push(`admitted:${runId}`);
    return {
      runId,
      foregroundOpportunities: Object.freeze({ runId, boundary: 'kit' }),
      async withForeground(phase, callback) {
        events.push(`scope:${phase}`);
        return await callback();
      },
      async finish() { events.push(`finish:${runId}`); return { submissionCount: 3 }; },
    };
  },
};

const prepared = await prepareProducerRun({
  lifecycle,
  foreground,
  runId: 'foreground-service-uptake',
  buildOptions: () => Object.freeze({ route: 'product' }),
});

assert.deepEqual(events, ['begin:foreground-service-uptake', 'admitted:foreground-service-uptake']);
assert.equal(prepared.options.foregroundOpportunities.boundary, 'kit');
assert.equal(await prepared.options.withForeground('uv-unwrap', async () => 'worker-value'), 'worker-value');
assert.deepEqual(events, [
  'begin:foreground-service-uptake',
  'admitted:foreground-service-uptake',
  'scope:uv-unwrap',
]);
assert.deepEqual(await prepared.release(), { submissionCount: 3 });
assert.equal(lifecycle.activeRunId, null);
assert.deepEqual(events.at(-1), 'finish:foreground-service-uptake');

assert.equal(await withForegroundScope({}, 'no-service', async () => 'plain-value'), 'plain-value');
assert.equal(await withForegroundScope(
  { withForeground: async (phase, work) => `scoped:${phase}:${await work()}` },
  'marching-tet-worker',
  async () => 'worker-value',
), 'scoped:marching-tet-worker:worker-value');

console.log('FOREGROUND SERVICE UPTAKE CONTRACT PASSED');
