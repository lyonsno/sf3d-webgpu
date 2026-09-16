#!/usr/bin/env node
/**
 * Foreground-opportunity seam contract for the SF3D cooperative runtime
 * (cooperative_dino.js createSf3dCooperativeRuntime).
 *
 * The kit facade awaits runtime.prepareCommandDutyAtBoundary(descriptor,
 * schedulerInvocation) before every cooperative GPU encode. When the runtime is
 * given a foreground-opportunity interlock (the kit's
 * createWebGpuForegroundOpportunityInterlock shape), pending foreground demand
 * must be serviced at that boundary exactly as the kit's own runtime does:
 *   1. with pending demand: serviceAtBoundary is called once with
 *      position 'before-encode', the invocation id, the duty id and phase; the
 *      service record is attached to descriptor.metadata.foregroundOpportunityService;
 *   2. with no demand: serviceAtBoundary is NOT called (no synthetic service);
 *   3. a failed service throws (the duty is not encoded on top of a failed
 *      foreground turn);
 *   4. demand without an invocation identity throws;
 *   5. without an interlock the runtime stays a pass-through (unchanged A/B
 *      behavior).
 */
import assert from 'node:assert/strict';
import { createSf3dCooperativeRuntime } from '../src/lib/cooperative_dino.js';

const fakeDevice = { queue: { submit() {}, onSubmittedWorkDone: async () => {} } };

function makeInterlock({ pending, serviceStatus = 'serviced' }) {
  const calls = [];
  return {
    calls,
    schema: 'kaminos.webgpu-foreground-opportunity-interlock.v0',
    request() { throw new Error('not used'); },
    snapshot() { return { pendingRequestCount: pending, activeServiceCount: 0, queuedServiceCount: 0 }; },
    pressureSnapshot() { return { pendingRequestCount: pending, activeRequestCount: 0, activeServiceCount: 0, queuedServiceCount: 0 }; },
    async serviceAtBoundary(boundary) {
      calls.push(boundary);
      return { schema: 'kaminos.webgpu-foreground-opportunity-service.v0', status: serviceStatus, boundary,
        capturedRequestCount: pending, servicedRequestCount: serviceStatus === 'serviced' ? pending : 0,
        failures: serviceStatus === 'failed' ? [{ requestId: 'kiln-frame:1', failure: { error: { message: 'kiln encode threw' } } }] : [] };
    },
    finish() { return { receipts: [] }; },
  };
}
const invocation = { invocationId: 'sf3d:dino:cooperative', schedulerRevision: null };

// 1. Pending demand is serviced at the boundary and recorded on the descriptor.
{
  const interlock = makeInterlock({ pending: 2 });
  const runtime = createSf3dCooperativeRuntime(fakeDevice, { foregroundOpportunities: interlock });
  const out = await runtime.prepareCommandDutyAtBoundary({ phase: 'dinov2-tokenizer', dutyId: 'dino:block:3' }, invocation);
  assert.equal(interlock.calls.length, 1, 'one service turn per boundary');
  const b = interlock.calls[0];
  assert.equal(b.position, 'before-encode');
  assert.equal(b.invocationId, 'sf3d:dino:cooperative');
  assert.equal(b.dutyId, 'dino:block:3');
  assert.equal(b.phase, 'dinov2-tokenizer');
  assert.match(b.boundaryId, /^sf3d:dino:cooperative:foreground-boundary:\d+$/);
  assert.equal(out.metadata.foregroundOpportunityService.status, 'serviced');
  assert.equal(out.metadata.foregroundOpportunityService.servicedRequestCount, 2);
  assert.equal(runtime.foregroundOpportunities, interlock, 'interlock exposed on the runtime');
  console.log('ok  pending demand serviced before encode and recorded');
}

// 2. No demand → no synthetic service turn.
{
  const interlock = makeInterlock({ pending: 0 });
  const runtime = createSf3dCooperativeRuntime(fakeDevice, { foregroundOpportunities: interlock });
  const out = await runtime.prepareCommandDutyAtBoundary({ phase: 'post-processor', dutyId: 'pp:1' }, invocation);
  assert.equal(interlock.calls.length, 0);
  assert.equal(out.metadata?.foregroundOpportunityService, undefined);
  console.log('ok  no demand → no service turn');
}

// 3. Failed service refuses the encode.
{
  const interlock = makeInterlock({ pending: 1, serviceStatus: 'failed' });
  const runtime = createSf3dCooperativeRuntime(fakeDevice, { foregroundOpportunities: interlock });
  await assert.rejects(
    () => runtime.prepareCommandDutyAtBoundary({ phase: 'two-stream-backbone', dutyId: 'ts:9' }, invocation),
    /kiln encode threw/);
  console.log('ok  failed foreground service refuses the encode');
}

// 4. Demand without invocation identity refuses.
{
  const interlock = makeInterlock({ pending: 1 });
  const runtime = createSf3dCooperativeRuntime(fakeDevice, { foregroundOpportunities: interlock });
  await assert.rejects(
    () => runtime.prepareCommandDutyAtBoundary({ phase: 'texture-bake', dutyId: 'tb:1' }, { invocationId: '' }),
    /requires an active invocation identity/);
  console.log('ok  demand without invocation identity refuses');
}

// 5. No interlock → pass-through (A/B behavior unchanged).
{
  const runtime = createSf3dCooperativeRuntime(fakeDevice, {});
  const d = { phase: 'dinov2-tokenizer', dutyId: 'x' };
  assert.equal(await runtime.prepareCommandDutyAtBoundary(d, invocation), d);
  assert.equal(runtime.foregroundOpportunities, null);
  console.log('ok  no interlock → pass-through');
}

console.log('\nCOOPERATIVE RUNTIME FOREGROUND CONTRACT PASSED');
