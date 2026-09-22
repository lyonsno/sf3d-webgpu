#!/usr/bin/env node
/**
 * Producer disposal must retain the kit foreground service's asynchronous
 * drain authority.  Declaring release before an admitted idle callback settles
 * lets host teardown race producer-owned workers and weights.
 */
import assert from 'node:assert/strict';
import { createWebGpuForegroundService } from '@kaminos/webgpu-inference-kit';
import { createProducerLifecycle, prepareProducerRun } from '../src/lib/producer_lifecycle.js';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

async function remainsPending(promise) {
  const marker = Symbol('pending');
  assert.equal(await Promise.race([promise.then(() => false, () => false), Promise.resolve(marker)]), marker);
}

function foregroundService(routeId) {
  const queue = { submit() {} };
  return createWebGpuForegroundService({ routeId, device: { queue }, queue });
}

// An already-admitted outside-run callback must settle before producer-owned
// resources are released or disposal completion resolves.
{
  const callbackGate = deferred();
  const callbackStarted = deferred();
  const foreground = foregroundService('sf3d.test.dispose-idle');
  let releasedResources = 0;
  const lifecycle = createProducerLifecycle({
    async release() {
      await foreground.dispose();
      releasedResources += 1;
    },
  });
  const handle = foreground.request({
    requestId: 'idle-frame',
    async run() {
      callbackStarted.resolve();
      await callbackGate.promise;
      return { frame: 'complete' };
    },
  });
  await callbackStarted.promise;

  const first = lifecycle.dispose();
  const second = lifecycle.dispose();
  assert.equal(first.status, 'releasing');
  assert.equal(second.status, 'already-disposed');
  assert.equal(first.completion, second.completion, 'repeated disposal returns one completion authority');
  assert.throws(() => lifecycle.beginRun('too-late'), /disposed/);
  assert.throws(() => lifecycle.assertAcceptingRequests(), /disposed/);
  await remainsPending(first.completion);
  assert.equal(releasedResources, 0, 'resources stay owned while the idle callback drains');

  callbackGate.resolve();
  assert.equal((await handle.completion).status, 'completed');
  assert.deepEqual(await first.completion, { status: 'released' });
  assert.equal(releasedResources, 1);
  assert.equal((await lifecycle.dispose().completion).status, 'released');
  assert.equal(releasedResources, 1, 'resources release exactly once');
}

// Callback failure still settles through the service and cannot strand the
// lifecycle drain or duplicate producer resource release.
{
  const foreground = foregroundService('sf3d.test.dispose-callback-failure');
  const callbackStarted = deferred();
  const callbackGate = deferred();
  let releasedResources = 0;
  const lifecycle = createProducerLifecycle({
    async release() {
      await foreground.dispose();
      releasedResources += 1;
    },
  });
  const handle = foreground.request({
    requestId: 'failing-idle-frame',
    async run() {
      callbackStarted.resolve();
      await callbackGate.promise;
      throw new Error('frame failed');
    },
  });
  await callbackStarted.promise;
  const disposal = lifecycle.dispose();
  await remainsPending(disposal.completion);
  callbackGate.resolve();
  const receipt = await handle.completion;
  assert.equal(receipt.status, 'failed-before-submission');
  assert.equal(receipt.failure.error.message, 'frame failed');
  assert.deepEqual(await disposal.completion, { status: 'released' });
  assert.equal(releasedResources, 1);
}

// Disposal during a run refuses new work immediately but cannot start the
// foreground-service/resource drain until that run's finish boundary settles.
{
  const finishGate = deferred();
  const events = [];
  const lifecycle = createProducerLifecycle({ async release() { events.push('resources-released'); } });
  const prepared = await prepareProducerRun({
    lifecycle,
    foreground: {
      async beginRun(runId) {
        return {
          runId,
          foregroundOpportunities: Object.freeze({ runId }),
          async withForeground(_phase, work) { return await work(); },
          async finish() {
            events.push('finish-started');
            await finishGate.promise;
            events.push('finish-settled');
            return { status: 'succeeded' };
          },
        };
      },
    },
    runId: 'active-run',
    buildOptions: () => Object.freeze({ route: 'product' }),
  });
  const first = lifecycle.dispose();
  const second = lifecycle.dispose();
  assert.equal(first.status, 'deferred-until-run-ends');
  assert.equal(first.completion, second.completion);
  assert.throws(() => lifecycle.beginRun('too-late'), /disposed/);
  const runRelease = prepared.release();
  await Promise.resolve();
  assert.deepEqual(events, ['finish-started']);
  await remainsPending(first.completion);
  finishGate.resolve();
  assert.deepEqual(await runRelease, { status: 'succeeded' });
  assert.deepEqual(await first.completion, { status: 'released' });
  assert.deepEqual(events, ['finish-started', 'finish-settled', 'resources-released']);
}

console.log('PRODUCER DISPOSAL COMPLETION CONTRACT PASSED');
