/**
 * Validator-backed acceptance for the strict-prefix SF3D cooperative
 * mechanisms (DINO, two-stream, texture bake), the counterpart of
 * bounded_prefix_acceptance.mjs for the post-processor.
 *
 * Review 2026-09-16 (HIGH): the witness harness built a kit-backed validation
 * only for the bounded-prefix post-processor; DINO, two-stream and texture
 * bake had no mandatory validation record and the acceptor treated absence as
 * acceptable. Every requested mechanism's COMPLETE kit execution report now
 * goes through the public `validateWebGpuCooperativeExecutionReport` with the
 * exact route / manifest / invocation / scheduling mode / completion policy
 * (and the measured duty count when the caller pins one), and the verdict is
 * preserved in the witness where the acceptor requires it.
 */
import { validateWebGpuCooperativeExecutionReport } from '@kaminos/webgpu-inference-kit';
import { expectedCooperativeIdentity } from './product_route_witness_report.mjs';

/** Kit expectations for one mechanism, derived from the requested route options. */
export function cooperativeMechanismExpectations(key, requestedOptions, { expectedGpuDutyCount = null } = {}) {
  const identity = expectedCooperativeIdentity(requestedOptions)[key];
  if (!identity) throw new Error(`${key} is not a requested cooperative mechanism`);
  const expectations = {
    expectedStatus: 'succeeded',
    expectedRouteId: identity.routeId,
    expectedManifestId: identity.manifestId,
    expectedInvocationId: identity.invocationId,
    expectedSchedulingMode: identity.schedulingMode,
    expectedCompletionPolicy: identity.completionPolicy,
  };
  if (expectedGpuDutyCount != null) expectations.expectedGpuDutyCount = expectedGpuDutyCount;
  return Object.freeze(expectations);
}

/** { ok, errors, expectations } — the kit validator's verdict over the complete report. */
export function acceptCooperativeMechanismReport(key, report, requestedOptions, options = {}) {
  const expectations = cooperativeMechanismExpectations(key, requestedOptions, options);
  const verdict = validateWebGpuCooperativeExecutionReport(report, expectations);
  return Object.freeze({
    ok: verdict.ok === true,
    errors: Object.freeze([...(verdict.errors || [])]),
    expectations,
  });
}
