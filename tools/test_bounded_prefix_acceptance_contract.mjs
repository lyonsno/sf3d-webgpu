/**
 * Deterministic fail-first contract for the bounded-prefix arm ACCEPTANCE gate
 * (kit >=0.1.42), the F1 remedy from cranial-depth-enema's exact-commit review
 * of 7393ab06.
 *
 * The prior hand-rolled `boundedPrefixHonored` predicate could certify
 * incomplete evidence: it accepted any positive number of adapter queue fences,
 * dropped the per-duty gpuDuties ledger, and never required denominator-bearing
 * bounded-arm progress. The review named three live falsifiers that left the
 * gate green:
 *   1. replace the bounded arm's queue-fence array with one fence;
 *   2. remove every bounded-arm progress message;
 *   3. omit or corrupt every bounded duty's raw queue duration.
 *
 * `acceptBoundedPrefixArm` now delegates to the public kit validator
 * `validateWebGpuCooperativeExecutionReport` over the COMPLETE report plus the
 * SF3D denominator-bearing progress string. This test proves the lawful report
 * is accepted and every falsifier (the three named, plus unfenced submission and
 * residual in-flight) is rejected with a specific error — before any browser or
 * GPU is involved.
 *
 * Run: node tools/test_bounded_prefix_acceptance_contract.mjs
 */
import assert from 'node:assert/strict';
import {
  makeValidBoundedReport,
  withSingleFence,
  withMissingProgress,
  withNonfiniteRawQueueDuration,
  withUnfencedSubmission,
  withResidualInFlight,
} from './bounded_prefix_report_fixture.mjs';
import {
  acceptBoundedPrefixArm,
  expectedChannelDutyCount,
  boundedPrefixExpectations,
  SF3D_POST_PROCESSOR_ROUTE_ID,
  SF3D_POST_PROCESSOR_CHANNEL_MANIFEST_ID,
  SF3D_POST_PROCESSOR_COOPERATIVE_INVOCATION_ID,
} from './bounded_prefix_acceptance.mjs';

const EXPECTED = 702;
const DEPTH = 2;
const PROGRESS_OK = [
  'Post-processor duties 351/702 (50%)',
  `Post-processor duties ${EXPECTED}/${EXPECTED} (100%)`,
];

function accept(report, progressMessages = PROGRESS_OK) {
  return acceptBoundedPrefixArm({
    report,
    progressMessages,
    expectedGpuDutyCount: EXPECTED,
    maxInFlightGpuDuties: DEPTH,
  });
}

// --- 0. Duty-count formula + identity constants are the exact contract --------
assert.equal(expectedChannelDutyCount(16), 702,
  'channelsPerDuty=16 must yield the exact 702-duty channel-range boundary');
assert.throws(() => expectedChannelDutyCount(0), /positive safe integer/);
const exp = boundedPrefixExpectations({ expectedGpuDutyCount: EXPECTED, maxInFlightGpuDuties: DEPTH });
assert.equal(exp.expectedRouteId, SF3D_POST_PROCESSOR_ROUTE_ID);
assert.equal(exp.expectedManifestId, SF3D_POST_PROCESSOR_CHANNEL_MANIFEST_ID);
assert.equal(exp.expectedInvocationId, SF3D_POST_PROCESSOR_COOPERATIVE_INVOCATION_ID);
assert.equal(exp.expectedSchedulingMode, 'cooperative');
assert.equal(exp.expectedCompletionPolicy, 'bounded-prefix');
assert.equal(exp.expectedGpuDutyCount, EXPECTED);
assert.equal(exp.expectedMaxInFlightGpuDuties, DEPTH);
assert.equal(exp.requireConfiguredDepthObserved, true);
console.log('ok  702-duty formula + exact route/manifest/invocation/depth expectations');

// --- 1. The lawful complete report is accepted --------------------------------
const valid = accept(makeValidBoundedReport());
assert.equal(valid.ok, true, `valid bounded report must be accepted; errors: ${JSON.stringify(valid.errors)}`);
assert.equal(valid.errors.length, 0);
assert.equal(valid.validation.ok, true);
assert.equal(valid.progressHonest, true);
console.log('ok  lawful 702-duty depth-2 bounded report is accepted');

// --- 2. FALSIFIER: one-of-N fence telemetry (review falsifier #1) -------------
{
  const r = accept(withSingleFence(makeValidBoundedReport()));
  assert.equal(r.ok, false, 'one-fence report must be rejected');
  assert.ok(r.errors.some((e) => /observedPrefixFenceCount/.test(e)),
    `expected a prefix-fence settlement error, got: ${JSON.stringify(r.errors)}`);
}
console.log('ok  one-of-N fence telemetry rejected (was: one fence passed the gate)');

// --- 3. FALSIFIER: missing bounded progress in the report (review falsifier #2)
{
  const r = accept(withMissingProgress(makeValidBoundedReport()));
  assert.equal(r.ok, false, 'report with no terminal progress must be rejected');
  assert.ok(r.errors.some((e) => /progress/i.test(e)));
}
// SF3D-specific: report is fine but the human-facing denominator progress string
// is absent — the smoke must still reject it.
{
  const r = accept(makeValidBoundedReport(), ['Post-processor duties 351/702 (50%)']);
  assert.equal(r.ok, false, 'missing 100% progress string must be rejected');
  assert.equal(r.progressHonest, false);
  assert.ok(r.errors.some((e) => /denominator-bearing/.test(e)));
}
console.log('ok  missing bounded progress rejected (report terminal + SF3D string)');

// --- 4. FALSIFIER: missing/nonfinite raw queue duration (review falsifier #3) -
{
  const r = accept(withNonfiniteRawQueueDuration(makeValidBoundedReport()));
  assert.equal(r.ok, false, 'nonfinite raw queue duration must be rejected');
  assert.ok(r.errors.some((e) => /rawQueueDurationMs/.test(e)));
}
console.log('ok  missing/nonfinite raw queue duration rejected');

// --- 5. FALSIFIER: unfenced submitted duty ------------------------------------
{
  const r = accept(withUnfencedSubmission(makeValidBoundedReport()));
  assert.equal(r.ok, false, 'nonzero unfenced submissions must be rejected');
  assert.ok(r.errors.some((e) => /unfenced/i.test(e)));
}
console.log('ok  nonzero unfenced submitted duties rejected');

// --- 6. FALSIFIER: residual terminal in-flight work ---------------------------
{
  const r = accept(withResidualInFlight(makeValidBoundedReport()));
  assert.equal(r.ok, false, 'nonzero terminal in-flight work must be rejected');
  assert.ok(r.errors.some((e) => /inFlight/i.test(e)));
}
console.log('ok  nonzero terminal in-flight work rejected');

// --- 7. Wrong identity is not this boundary's evidence ------------------------
{
  const r = accept(makeValidBoundedReport({ manifestId: 'sf3d.post-processor-layer-cooperative-boundaries.v0' }));
  assert.equal(r.ok, false, 'a report from a different manifest must be rejected');
  assert.ok(r.errors.some((e) => /manifestId/.test(e)));
}
console.log('ok  mismatched manifest identity rejected');

console.log('\nALL BOUNDED-PREFIX ACCEPTANCE CONTRACT CHECKS PASSED');
