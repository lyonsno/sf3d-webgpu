#!/usr/bin/env node
/**
 * Producer lifecycle contract (src/lib/producer_lifecycle.js).
 *
 * Advisory review A2 (2026-09-16): dispose() during an active run left the
 * bridge run open (drain timer firing, second beginRun refused forever) and
 * pending host requests without a settlement path. Contract:
 *   1. dispose() with no active run releases immediately (once);
 *   2. dispose() during an active run defers the release until that run ends,
 *      reports the deferral, and the run's own finish still happens;
 *   3. after dispose is requested, new runs and new foreground requests are
 *      refused, but the active run's end still releases exactly once;
 *   4. a second dispose is a no-op.
 */
import assert from 'node:assert/strict';
import { createProducerLifecycle } from '../src/lib/producer_lifecycle.js';

function make() {
  const log = [];
  const lc = createProducerLifecycle({ release: () => { log.push('release'); } });
  return { lc, log };
}

// 1. Idle dispose releases immediately.
{
  const { lc, log } = make();
  assert.deepEqual(lc.dispose(), { status: 'released' });
  assert.deepEqual(log, ['release']);
  assert.equal(lc.disposed, true);
  assert.throws(() => lc.beginRun('r1'), /disposed/);
  assert.throws(() => lc.assertAcceptingRequests(), /disposed/);
  console.log('ok  idle dispose releases immediately');
}

// 2 + 3. Dispose during a run defers; run end releases once; new work refused meanwhile.
{
  const { lc, log } = make();
  lc.beginRun('r2');
  assert.equal(lc.activeRunId, 'r2');
  assert.deepEqual(lc.dispose(), { status: 'deferred-until-run-ends', runId: 'r2' });
  assert.deepEqual(log, [], 'nothing released while the run is active');
  assert.equal(lc.disposed, true);
  assert.throws(() => lc.beginRun('r3'), /disposed/);
  assert.throws(() => lc.assertAcceptingRequests(), /disposed/);
  assert.equal(lc.endRun('r2'), 'released');
  assert.deepEqual(log, ['release']);
  assert.equal(lc.activeRunId, null);
  console.log('ok  dispose during a run defers; run end releases once');
}

// 4. Second dispose no-op; endRun of an unknown run refused; one run at a time.
{
  const { lc, log } = make();
  lc.beginRun('r4');
  assert.throws(() => lc.beginRun('r5'), /already has an active run/);
  assert.throws(() => lc.endRun('r9'), /not the active run/);
  assert.equal(lc.endRun('r4'), 'idle');
  assert.deepEqual(lc.dispose(), { status: 'released' });
  assert.deepEqual(lc.dispose(), { status: 'already-disposed' });
  assert.deepEqual(log, ['release']);
  console.log('ok  second dispose no-op; run exclusivity; unknown endRun refused');
}

console.log('\nPRODUCER LIFECYCLE CONTRACT PASSED');
