#!/usr/bin/env node
/**
 * Producer lifecycle contract (src/lib/producer_lifecycle.js).
 *
 * Advisory review A2 (2026-09-16): dispose() during an active run left the
 * bridge run open (drain timer firing, second beginRun refused forever) and
 * pending host requests without a settlement path. Contract:
 *   1. dispose() with no active run starts one asynchronous release (once);
 *   2. dispose() during an active run defers the release until that run ends,
 *      reports the deferral, and the run's own finish still happens;
 *   3. after dispose is requested, new runs and new foreground requests are
 *      refused, but the active run's end still releases exactly once;
 *   4. a second dispose returns the same completion authority.
 */
import assert from 'node:assert/strict';
import { createProducerLifecycle } from '../src/lib/producer_lifecycle.js';

function make() {
  const log = [];
  const lc = createProducerLifecycle({ release: () => { log.push('release'); } });
  return { lc, log };
}

// 1. Idle dispose starts release immediately and exposes its completion.
{
  const { lc, log } = make();
  const disposal = lc.dispose();
  assert.equal(disposal.status, 'releasing');
  assert.deepEqual(log, []);
  assert.deepEqual(await disposal.completion, { status: 'released' });
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
  const disposal = lc.dispose();
  assert.equal(disposal.status, 'deferred-until-run-ends');
  assert.equal(disposal.runId, 'r2');
  assert.deepEqual(log, [], 'nothing released while the run is active');
  assert.equal(lc.disposed, true);
  assert.throws(() => lc.beginRun('r3'), /disposed/);
  assert.throws(() => lc.assertAcceptingRequests(), /disposed/);
  assert.equal(lc.endRun('r2'), 'release-started');
  assert.deepEqual(await disposal.completion, { status: 'released' });
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
  const first = lc.dispose();
  const second = lc.dispose();
  assert.equal(first.status, 'releasing');
  assert.equal(second.status, 'already-disposed');
  assert.equal(first.completion, second.completion);
  assert.deepEqual(await first.completion, { status: 'released' });
  assert.deepEqual(log, ['release']);
  console.log('ok  second dispose no-op; run exclusivity; unknown endRun refused');
}

// 5. Run setup (r2 HIGH, 2026-09-16): every fallible input — run id, route
//    overrides, injected worker roles — is validated BEFORE lifecycle or bridge
//    state is acquired, and a failure after the first acquisition releases both
//    exactly once. A rejected run must never leave an active run behind.
{
  const { prepareProducerRun } = await import('../src/lib/producer_lifecycle.js');
  const acquired = [];
  const fakeForeground = {
    async beginRun(runId) {
      acquired.push(`foreground:${runId}`);
      return {
        runId,
        foregroundOpportunities: { runId },
        withForeground: async (_phase, work) => await work(),
        finish: async () => { acquired.push(`foreground-finish:${runId}`); return { status: 'succeeded' }; },
      };
    },
  };
  const { lc, log } = make();
  const buildOptions = (overrides) => { if (overrides?.boom) throw new Error('bad route overrides'); return Object.freeze({ ...overrides }); };
  // empty run id → rejected before any acquisition
  await assert.rejects(() => prepareProducerRun({ lifecycle: lc, foreground: fakeForeground, runId: '', buildOptions }), /runId must be a non-empty string/);
  assert.equal(lc.activeRunId, null); assert.deepEqual(acquired, []);
  // bad route options → rejected before any acquisition
  await assert.rejects(() => prepareProducerRun({ lifecycle: lc, foreground: fakeForeground, runId: 'r-bad', buildOptions: () => buildOptions({ boom: true }) }), /bad route overrides/);
  assert.equal(lc.activeRunId, null); assert.deepEqual(acquired, []);
  // a valid run acquires both and can be released through the returned boundary
  const prepared = await prepareProducerRun({ lifecycle: lc, foreground: fakeForeground, runId: 'r-ok', buildOptions });
  assert.equal(lc.activeRunId, 'r-ok'); assert.deepEqual(acquired, ['foreground:r-ok']);
  assert.equal(prepared.options.foregroundOpportunities.runId, 'r-ok');
  const report = await prepared.release();
  assert.equal(report.status, 'succeeded'); assert.equal(lc.activeRunId, null);
  assert.deepEqual(acquired, ['foreground:r-ok', 'foreground-finish:r-ok']);
  await prepared.release();                       // idempotent
  assert.deepEqual(acquired, ['foreground:r-ok', 'foreground-finish:r-ok']);
  // dispose after a rejected run releases immediately (nothing was left active)
  const disposal = lc.dispose();
  assert.equal(disposal.status, 'releasing');
  assert.deepEqual(await disposal.completion, { status: 'released' });
  assert.deepEqual(log, ['release']);
  console.log('ok  run setup validates before acquisition; release boundary is exactly-once');
}

console.log('\nPRODUCER LIFECYCLE CONTRACT PASSED');
