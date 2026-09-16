#!/usr/bin/env node
/**
 * Producer-side foreground-opportunity bridge contract
 * (src/lib/foreground_opportunity_bridge.js).
 *
 * The SF3D producer shares one GPUDevice with a host (the Kaminos kiln). Host
 * frames are submitted through the kit's foreground-opportunity interlock so
 * they are serviced before the next SF3D GPU duty encodes. Two holes a plain
 * interlock leaves open, and the bridge must close:
 *   - no run is active → there is never a boundary; the request must execute
 *     immediately on the device (receipt marked servicedOutsideRun);
 *   - a run is active but SF3D is in a CPU-only stretch (UV unwrap, UV
 *     rasterize, GLB export) → no scheduler boundary arrives; the bridge must
 *     service pending demand itself within a frame budget (idle drain).
 * Fail-first invariants:
 *   1. outside a run: executed immediately, submissions reach queue.submit;
 *   2. during a run with prompt scheduler boundaries: serviced at the
 *      scheduler boundary, zero idle drains;
 *   3. during a run with no scheduler boundaries: serviced by an idle drain
 *      within the budget, receipt names the drain boundary;
 *   4. finishRun with pending demand: serviced (not canceled) by the finish
 *      drain; report status succeeded;
 *   5. a second beginRun while one is active throws;
 *   6. a failing foreground callback yields a failed receipt and does not
 *      break the bridge (next request still serviced).
 */
import assert from 'node:assert/strict';
import { createForegroundOpportunityBridge } from '../src/lib/foreground_opportunity_bridge.js';

function makeFakeGpu() {
  const submits = [];
  const queue = { submit(buffers) { submits.push(buffers.length); }, onSubmittedWorkDone: async () => {} };
  return { device: { queue }, queue, submits };
}
const tick = (ms) => new Promise(r => setTimeout(r, ms));
const frameRequest = (id, submits = 1) => ({
  requestId: id,
  run(ctx) {
    for (let i = 0; i < submits; i += 1) ctx.submit([{ fake: `cb-${id}-${i}` }], { submissionId: `${id}:s${i}` });
    return { frame: id };
  },
  metadata: { kind: 'test-frame' },
});

// 1. Outside a run → immediate execution.
{
  const gpu = makeFakeGpu();
  const bridge = createForegroundOpportunityBridge({ routeId: 'sf3d.test', device: gpu.device, queue: gpu.queue });
  const handle = bridge.request(frameRequest('f1', 2));
  const receipt = await handle.completion;
  assert.equal(receipt.status, 'completed');
  assert.equal(receipt.servicedOutsideRun, true);
  assert.equal(receipt.submissions.length, 2);
  assert.deepEqual(receipt.result, { frame: 'f1' });
  assert.deepEqual(gpu.submits, [1, 1]);
  assert.equal(bridge.snapshot().activeRun, null);
  assert.equal(bridge.snapshot().outsideRunReceiptCount, 1);
  console.log('ok  outside a run: executed immediately on the device');
}

// 2. During a run with a prompt scheduler boundary → serviced there, no drain.
{
  const gpu = makeFakeGpu();
  const bridge = createForegroundOpportunityBridge({ routeId: 'sf3d.test', device: gpu.device, queue: gpu.queue, drainIntervalMs: 2, drainAfterMs: 50 });
  const run = bridge.beginRun('run-A');
  const fo = run.foregroundOpportunities;
  assert.equal(typeof fo.serviceAtBoundary, 'function');
  assert.equal(typeof fo.pressureSnapshot, 'function');
  const handle = bridge.request(frameRequest('f2'));
  assert.equal(fo.pressureSnapshot().pendingRequestCount, 1);
  const service = await fo.serviceAtBoundary({ invocationId: 'sched:1', boundaryId: 'sched:1:b1', dutyId: 'duty-1', phase: 'dinov2-tokenizer', position: 'before-encode' });
  assert.equal(service.status, 'serviced');
  const receipt = await handle.completion;
  assert.equal(receipt.status, 'completed');
  assert.equal(receipt.boundary.invocationId, 'sched:1');
  assert.equal(receipt.servicedOutsideRun, undefined);
  const report = await run.finish();
  assert.equal(report.status, 'succeeded');
  assert.equal(report.producer.schedulerBoundaryServiceCount, 1);
  assert.equal(report.producer.idleDrainBoundaryCount, 0);
  assert.deepEqual(gpu.submits, [1]);
  console.log('ok  during a run: serviced at the scheduler boundary, no idle drain');
}

// 3. During a run with no scheduler boundaries → idle drain within budget.
{
  const gpu = makeFakeGpu();
  const bridge = createForegroundOpportunityBridge({ routeId: 'sf3d.test', device: gpu.device, queue: gpu.queue, drainIntervalMs: 2, drainAfterMs: 5 });
  const run = bridge.beginRun('run-B');
  const t0 = performance.now();
  const handle = bridge.request(frameRequest('f3'));
  const receipt = await Promise.race([handle.completion, tick(500).then(() => { throw new Error('idle drain never serviced the request'); })]);
  assert.equal(receipt.status, 'completed');
  assert.equal(receipt.boundary.phase, 'sf3d-producer-idle-drain');
  assert.ok(performance.now() - t0 < 200, 'drained within budget');
  const report = await run.finish();
  assert.ok(report.producer.idleDrainBoundaryCount >= 1);
  assert.equal(report.producer.schedulerBoundaryServiceCount, 0);
  assert.equal(report.status, 'succeeded');
  console.log('ok  during a CPU-only stretch: idle drain services within budget');
}

// 4. finishRun with pending demand → serviced by the finish drain.
{
  const gpu = makeFakeGpu();
  const bridge = createForegroundOpportunityBridge({ routeId: 'sf3d.test', device: gpu.device, queue: gpu.queue, drainIntervalMs: 1000, drainAfterMs: 1000 });
  const run = bridge.beginRun('run-C');
  const handle = bridge.request(frameRequest('f4'));
  const report = await run.finish();
  const receipt = await handle.completion;
  assert.equal(receipt.status, 'completed', 'finish drains pending demand instead of canceling it');
  assert.equal(receipt.boundary.phase, 'sf3d-producer-run-finish');
  assert.equal(report.status, 'succeeded');
  assert.equal(report.producer.finishDrainServicedCount, 1);
  assert.equal(bridge.snapshot().activeRun, null);
  // After finish, requests go back to immediate execution.
  const after = await bridge.request(frameRequest('f5')).completion;
  assert.equal(after.servicedOutsideRun, true);
  console.log('ok  finishRun drains pending demand; requests after finish run immediately');
}

// 5. Concurrent runs refused.
{
  const gpu = makeFakeGpu();
  const bridge = createForegroundOpportunityBridge({ routeId: 'sf3d.test', device: gpu.device, queue: gpu.queue });
  const run = bridge.beginRun('run-D');
  assert.throws(() => bridge.beginRun('run-E'), /already has an active run/);
  await run.finish();
  console.log('ok  second beginRun while active refused');
}

// 6. Failing callback → failed receipt; bridge keeps working.
{
  const gpu = makeFakeGpu();
  const bridge = createForegroundOpportunityBridge({ routeId: 'sf3d.test', device: gpu.device, queue: gpu.queue });
  const bad = await bridge.request({ requestId: 'bad', run() { throw new Error('kiln encode threw'); } }).completion;
  assert.equal(bad.status, 'failed-before-submission');
  assert.match(bad.failure.error.message, /kiln encode threw/);
  const good = await bridge.request(frameRequest('f6')).completion;
  assert.equal(good.status, 'completed');
  // Duplicate ids refused while the first is still in flight (outside a run)
  // and for the whole run (inside a run, kit policy).
  const slow = bridge.request({ requestId: 'dup', async run() { await tick(20); return 1; } });
  assert.throws(() => bridge.request(frameRequest('dup')), /duplicate/);
  await slow.completion;
  const run = bridge.beginRun('run-F');
  bridge.request(frameRequest('in-run'));
  assert.throws(() => bridge.request(frameRequest('in-run')), /duplicate/);
  await run.finish();
  console.log('ok  failing callback → failed receipt; bridge still services; duplicates refused');
}

console.log('\nFOREGROUND OPPORTUNITY BRIDGE CONTRACT PASSED');
