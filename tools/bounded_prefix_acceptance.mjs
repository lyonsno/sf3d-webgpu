/**
 * Validator-backed bounded-prefix acceptance for the SF3D postprocessor
 * channel-range boundary (kit >=0.1.42).
 *
 * This replaces the hand-rolled `boundedPrefixHonored` predicate that
 * cranial-depth-enema's exact-commit review of 7393ab06 found could certify
 * incomplete evidence (F1): it accepted any positive number of adapter queue
 * fences, dropped the per-duty `gpuDuties` ledger (raw queue durations, timing
 * authority, retirement status), and never required denominator-bearing
 * bounded-arm progress. The three live falsifiers were: one-of-N fence
 * telemetry, missing bounded progress, and missing/nonfinite raw queue
 * duration.
 *
 * The public kit validator `validateWebGpuCooperativeExecutionReport` is the
 * authoritative runtime completion gate. It validates route/manifest/invocation
 * identity, scheduling/completion-policy identity, denominator-bearing terminal
 * progress, exact range coverage, bounded completion authority, caller-selected
 * bounded depth, submitted/observed/unfenced queue settlement, zero residual
 * work, uncapped retained duty rows, finite raw queue timing, and the
 * one-to-one GPU-range↔duty-ledger bijection. We feed it the COMPLETE kit
 * execution report (not a lossy scalar projection) with the exact SF3D
 * expectations, and additionally assert the SF3D-specific denominator-bearing
 * progress STRING the smoke emits, which lives outside the kit report.
 */
import {
  validateWebGpuCooperativeExecutionReport,
} from '@kaminos/webgpu-inference-kit';

// Exact SF3D postprocessor channel-range identity. These are the routeId /
// manifestId / invocationId the cooperative execution actually carries; a
// report that does not match them is not this boundary's evidence.
export const SF3D_POST_PROCESSOR_ROUTE_ID = 'sf3d.image-to-mesh.webgpu-local.v0';
export const SF3D_POST_PROCESSOR_CHANNEL_MANIFEST_ID =
  'sf3d.post-processor-channel-cooperative-boundaries.v0';
export const SF3D_POST_PROCESSOR_COOPERATIVE_INVOCATION_ID =
  'sf3d:post-processor:cooperative';

// 3 planes × (gather + 3×ceil(1024/16) conv-ranges + ceil(640/16) conv-range +
// pixel-shuffle-copy) at channelsPerDuty=16. Kept as a function so the caller's
// channelsPerDuty is the single source of truth, not a copied constant.
export function expectedChannelDutyCount(channelsPerDuty) {
  if (!Number.isSafeInteger(channelsPerDuty) || channelsPerDuty <= 0) {
    throw new TypeError('channelsPerDuty must be a positive safe integer');
  }
  return 3 * (
    2
    + 3 * Math.ceil(1024 / channelsPerDuty)
    + Math.ceil(640 / channelsPerDuty)
  );
}

/**
 * Build the exact bounded-prefix expectations for the SF3D channel-range arm.
 * requireConfiguredDepthObserved:true forces the report to prove the configured
 * depth was actually reached (maxObservedInFlight === maxInFlight), not merely
 * not-breached.
 */
export function boundedPrefixExpectations({
  expectedGpuDutyCount,
  maxInFlightGpuDuties,
}) {
  if (!Number.isSafeInteger(expectedGpuDutyCount) || expectedGpuDutyCount < 0) {
    throw new TypeError('expectedGpuDutyCount must be a nonnegative safe integer');
  }
  if (!Number.isSafeInteger(maxInFlightGpuDuties) || maxInFlightGpuDuties <= 0) {
    throw new TypeError('maxInFlightGpuDuties must be a positive safe integer');
  }
  return {
    expectedStatus: 'succeeded',
    expectedRouteId: SF3D_POST_PROCESSOR_ROUTE_ID,
    expectedManifestId: SF3D_POST_PROCESSOR_CHANNEL_MANIFEST_ID,
    expectedInvocationId: SF3D_POST_PROCESSOR_COOPERATIVE_INVOCATION_ID,
    expectedSchedulingMode: 'cooperative',
    expectedCompletionPolicy: 'bounded-prefix',
    expectedGpuDutyCount,
    expectedMaxInFlightGpuDuties: maxInFlightGpuDuties,
    requireConfiguredDepthObserved: true,
  };
}

/**
 * The SF3D-specific denominator-bearing progress evidence: the bounded arm must
 * have emitted at least one `Post-processor duties N/N (100%)` message with the
 * exact expected N. The kit validator checks the report's terminal `progress`
 * object; this additionally binds the human-facing/string progress channel the
 * smoke surfaces, so a bounded arm that silently drops progress messages fails.
 */
export function boundedProgressHonest(progressMessages, expectedGpuDutyCount) {
  if (!Array.isArray(progressMessages)) return false;
  const re = new RegExp(
    `Post-processor duties ${expectedGpuDutyCount}/${expectedGpuDutyCount} \\(100%\\)`,
  );
  return progressMessages.some((message) => re.test(message));
}

/**
 * Accept (or reject) a bounded-prefix arm from its COMPLETE kit execution report
 * plus its SF3D progress messages. Returns a structured verdict:
 *   { ok, validation, progressHonest, errors }
 * `validation` is the raw validator result (ok + errors + effective identity).
 * `errors` is the union of validator errors and any SF3D progress failure, so a
 * caller can surface exactly why acceptance failed.
 *
 * This function is pure and deterministic — it is the unit under the fail-first
 * falsifier tests.
 */
export function acceptBoundedPrefixArm({
  report,
  progressMessages,
  expectedGpuDutyCount,
  maxInFlightGpuDuties,
}) {
  const validation = validateWebGpuCooperativeExecutionReport(
    report,
    boundedPrefixExpectations({ expectedGpuDutyCount, maxInFlightGpuDuties }),
  );
  const progressHonest = boundedProgressHonest(progressMessages, expectedGpuDutyCount);
  const errors = [...validation.errors];
  if (!progressHonest) {
    errors.push(
      `bounded arm must emit denominator-bearing `
      + `"Post-processor duties ${expectedGpuDutyCount}/${expectedGpuDutyCount} (100%)" progress`,
    );
  }
  return {
    ok: validation.ok && progressHonest,
    validation,
    progressHonest,
    errors,
  };
}
