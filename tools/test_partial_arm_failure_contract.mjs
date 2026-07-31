/**
 * Deterministic fail-first contract for the partial-arm failure assembly — the
 * F2 remedy from cranial-depth-enema's exact-commit review of 7393ab06.
 *
 * A fifth-arm failure must NOT discard the four completed arm receipts, and the
 * durable failure report must name the failing arm, the last completed prior
 * arm, and the failing arm's post-drain settlement. This proves the pure
 * assembly used by the smoke's catch path; the browser-exercised witness lives
 * in the smoke's --inject-fail-arm mode.
 *
 * Run: node tools/test_partial_arm_failure_contract.mjs
 */
import assert from 'node:assert/strict';
import { assemblePartialArmFailure } from './partial_arm_failure.mjs';

// Simulate the browser-owned ledger + failure record after the bounded fifth
// arm throws with four prior arms completed and its own duties drained.
const ledger = [
  { name: 'monolithic-control', cooperativeStatus: null, completionPolicy: null },
  { name: 'three-plane-cooperative', cooperativeStatus: 'succeeded', completionPolicy: 'strict-prefix' },
  { name: 'eighteen-stage-cooperative', cooperativeStatus: 'succeeded', completionPolicy: 'strict-prefix' },
  { name: 'channel-range-cooperative', cooperativeStatus: 'succeeded', completionPolicy: 'strict-prefix' },
];
const failure = {
  message: 'injected failure in arm channel-range-bounded-prefix-2',
  failingArm: 'channel-range-bounded-prefix-2',
  completedArms: ledger.map((e) => e.name),
  lastCompletedArm: 'channel-range-cooperative',
  drainAfterFailure: {
    status: 'succeeded',
    completionPolicy: 'bounded-prefix',
    issuedGpuDutyCount: 702,
    retiredGpuDutyCount: 702,
    inFlightGpuDutyCount: 0,
    maxObservedInFlightGpuDuties: 2,
  },
};

// --- 1. Completed-arm evidence is preserved, in order -------------------------
const out = assemblePartialArmFailure({ ledger, failure });
assert.equal(out.phase, 'browser-arms-partial');
assert.deepEqual(out.completedArmNames, [
  'monolithic-control',
  'three-plane-cooperative',
  'eighteen-stage-cooperative',
  'channel-range-cooperative',
]);
assert.equal(out.completedArms.length, 4, 'all four completed arm receipts preserved');
console.log('ok  four completed-arm receipts preserved in order');

// --- 2. Names failing arm + last completed prior arm --------------------------
assert.equal(out.failingArm, 'channel-range-bounded-prefix-2');
assert.equal(out.lastCompletedArm, 'channel-range-cooperative');
console.log('ok  names the failing arm and the last completed prior arm');

// --- 3. Preserves the failing arm's post-drain settlement ---------------------
assert.equal(out.drainAfterFailure.completionPolicy, 'bounded-prefix');
assert.equal(out.drainAfterFailure.issuedGpuDutyCount, 702);
assert.equal(out.drainAfterFailure.retiredGpuDutyCount, 702);
assert.equal(out.drainAfterFailure.inFlightGpuDutyCount, 0);
assert.equal(out.evidencePreserved, true);
console.log('ok  preserves bounded arm post-drain settlement (702/702, 0 in-flight)');

// --- 4. FAIL-FIRST: an empty/lost ledger is NOT silently treated as evidence --
{
  const lost = assemblePartialArmFailure({ ledger: [], failure: null });
  assert.equal(lost.completedArmNames.length, 0);
  assert.equal(lost.lastCompletedArm, null);
  assert.equal(lost.evidencePreserved, false,
    'a lost/empty ledger must not report evidence preserved');
}
console.log('ok  lost/empty ledger reports evidencePreserved=false (no false closure)');

// --- 5. A failure with only drain state (no completed arms) still preserves ---
{
  const drainOnly = assemblePartialArmFailure({
    ledger: [],
    failure: { message: 'x', failingArm: 'channel-range-bounded-prefix-2', drainAfterFailure: { inFlightGpuDutyCount: 0 } },
  });
  assert.equal(drainOnly.evidencePreserved, true);
  assert.equal(drainOnly.failingArm, 'channel-range-bounded-prefix-2');
}
console.log('ok  drain-only failure still preserves evidence + names failing arm');

console.log('\nALL PARTIAL-ARM FAILURE CONTRACT CHECKS PASSED');
