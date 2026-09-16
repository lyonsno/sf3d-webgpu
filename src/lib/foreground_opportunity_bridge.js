/**
 * Foreground-opportunity bridge — the SF3D producer's explicit integration of
 * the kit's foreground-opportunity interlock for a host that shares SF3D's
 * GPUDevice (the Kaminos kiln composition).
 *
 * Contract:
 *   - request({ requestId, run, metadata }) returns a kit-shaped handle
 *     { requestId, completion, cancel }. run(context) receives
 *     { device, queue, signal, submit(commandBuffers, { submissionId, metadata }) }
 *     exactly as the kit interlock hands it out.
 *   - While an SF3D run is active the request is queued on that run's kit
 *     interlock and serviced before the next SF3D GPU duty encodes (the
 *     cooperative runtime calls serviceAtBoundary from
 *     prepareCommandDutyAtBoundary). Nothing in that path is SF3D-owned
 *     scheduling; it is the kit's interlock verbatim.
 *   - Two holes a bare interlock leaves open are closed here:
 *       1. no run active → there is never a boundary; the request executes
 *          immediately on the device (receipt marked servicedOutsideRun);
 *       2. run active but SF3D is in a CPU-only stretch (UV unwrap, UV
 *          rasterize, GLB export, worker waits) → no scheduler boundary
 *          arrives; an idle drain services pending demand through a
 *          producer-owned before-encode boundary once no scheduler boundary
 *          has serviced demand for drainAfterMs (default one frame). When
 *          scheduler boundaries are frequent (every cooperative duty) the
 *          drain never fires; it is a liveness floor, not a cap.
 *   - finish() drains any still-pending demand through a final boundary
 *     (never cancels host frames) and returns the kit's finish report plus
 *     producer counters (scheduler / idle-drain / finish-drain services).
 *
 * Retention: the kit interlock retains every receipt of a run (uncapped, kit
 * policy). Outside-run receipts are delivered to their requester through
 * completion; the bridge keeps counters and the last receipt, not the history
 * (a host frame loop between runs would otherwise grow without bound).
 */
import {
  WEBGPU_FOREGROUND_OPPORTUNITY_SCHEMA,
  createWebGpuForegroundOpportunityInterlock,
} from '@kaminos/webgpu-inference-kit';

export const SF3D_FOREGROUND_BRIDGE_SCHEMA = 'sf3d.foreground-opportunity-bridge.v0';
export const SF3D_FOREGROUND_OUTSIDE_RUN_RECEIPT_SCHEMA = 'sf3d.foreground-opportunity-outside-run-receipt.v0';
export const SF3D_FOREGROUND_IDLE_DRAIN_PHASE = 'sf3d-producer-idle-drain';
export const SF3D_FOREGROUND_RUN_FINISH_PHASE = 'sf3d-producer-run-finish';

const isPlainObject = (v) => v != null && typeof v === 'object' && !Array.isArray(v);
const isNonEmptyString = (v) => typeof v === 'string' && v.trim().length > 0;
const normalizeError = (e) => ({ name: e?.name ?? 'Error', message: e?.message ?? String(e) });
const clone = (v) => (v == null ? null : JSON.parse(JSON.stringify(v)));
const defaultNow = () => globalThis.performance?.now?.() ?? Date.now();

export function createForegroundOpportunityBridge({
  routeId,
  device,
  queue = device?.queue,
  now = defaultNow,
  drainIntervalMs = 8,
  drainAfterMs = 16.7,
} = {}) {
  if (!isNonEmptyString(routeId)) throw new Error('routeId must be a non-empty string');
  if (!device || typeof device !== 'object') throw new Error('device must be an object');
  if (!queue || typeof queue.submit !== 'function') throw new Error('queue.submit must be a function');
  if (!(Number.isFinite(drainIntervalMs) && drainIntervalMs > 0)) throw new Error('drainIntervalMs must be > 0');
  if (!(Number.isFinite(drainAfterMs) && drainAfterMs >= 0)) throw new Error('drainAfterMs must be >= 0');

  const state = {
    activeRun: null,
    runCount: 0,
    outsideRun: { inFlight: new Map(), receiptCount: 0, completedCount: 0, failedCount: 0, lastReceipt: null },
  };

  function validateRequest(input) {
    if (!isPlainObject(input)) throw new Error('foreground opportunity request must be an object');
    if (!isNonEmptyString(input.requestId)) throw new Error('requestId must be a non-empty string');
    if (typeof input.run !== 'function') throw new Error('foreground opportunity run must be a function');
    if (input.metadata != null && !isPlainObject(input.metadata)) {
      throw new Error('foreground opportunity metadata must be an object when provided');
    }
  }

  // --- No active run: execute immediately on the device (kit-shaped context). ---
  function executeOutsideRun(input) {
    validateRequest(input);
    const { requestId } = input;
    if (state.outsideRun.inFlight.has(requestId)) {
      throw new Error(`duplicate foreground opportunity request ${requestId}`);
    }
    const metadata = clone(input.metadata || {});
    const abortController = new AbortController();
    const submissions = [];
    const requestedAtMs = now();
    const completion = (async () => {
      let result = null;
      let failure = null;
      const startedAtMs = now();
      try {
        result = await input.run(Object.freeze({
          schema: WEBGPU_FOREGROUND_OPPORTUNITY_SCHEMA,
          routeId,
          runId: null,
          requestId,
          boundary: null,
          device,
          queue,
          signal: abortController.signal,
          submit(commandBuffers, submissionInput = {}) {
            if (abortController.signal.aborted) throw new Error('foreground opportunity was canceled before submission');
            if (!Array.isArray(commandBuffers) || commandBuffers.length === 0) {
              throw new Error('foreground opportunity submit requires a non-empty command buffer array');
            }
            if (!isPlainObject(submissionInput)) throw new Error('foreground submission input must be an object');
            const submissionId = submissionInput.submissionId || `${requestId}:submission:${submissions.length + 1}`;
            if (!isNonEmptyString(submissionId)) throw new Error('submissionId must be a non-empty string');
            if (submissions.some(row => row.submissionId === submissionId)) {
              throw new Error(`duplicate foreground submission ${submissionId}`);
            }
            let submissionMetadata;
            try { submissionMetadata = clone(submissionInput.metadata || {}); } catch (error) {
              throw new Error(`foreground submission metadata must be JSON-serializable: ${error.message}`);
            }
            const submittedAtMs = now();
            try {
              queue.submit(commandBuffers);
              const row = Object.freeze({
                submissionId, submissionSequence: submissions.length + 1, commandBufferCount: commandBuffers.length,
                submittedAtMs, returnedAtMs: now(), submissionStatus: 'queue-submit-returned', metadata: submissionMetadata,
                authority: 'queue-submit-call-returned-no-gpu-completion-or-presentation-claim',
              });
              submissions.push(row);
              return row;
            } catch (error) {
              submissions.push(Object.freeze({
                submissionId, submissionSequence: submissions.length + 1, commandBufferCount: commandBuffers.length,
                submittedAtMs, returnedAtMs: now(), submissionStatus: 'queue-submit-threw', metadata: submissionMetadata,
                failure: normalizeError(error), authority: 'queue-submit-call-failed-no-gpu-submission-claim',
              }));
              throw error;
            }
          },
        }));
      } catch (error) {
        failure = { phase: 'foreground-callback', error: normalizeError(error) };
      }
      let resultClone = null;
      if (!failure) {
        try { resultClone = clone(result ?? null); } catch (error) {
          failure = { phase: 'foreground-result-serialization', error: normalizeError(error) };
        }
      }
      const successfulSubmissionCount = submissions.filter(row => row.submissionStatus === 'queue-submit-returned').length;
      const canceled = abortController.signal.aborted;
      const receipt = Object.freeze({
        schema: SF3D_FOREGROUND_OUTSIDE_RUN_RECEIPT_SCHEMA,
        routeId,
        runId: null,
        requestId,
        status: canceled
          ? 'canceled-during-service'
          : (failure ? (successfulSubmissionCount > 0 ? 'failed-after-submission' : 'failed-before-submission') : 'completed'),
        servicedOutsideRun: true,
        requestedAtMs,
        startedAtMs,
        settledAtMs: now(),
        boundary: null,
        result: resultClone,
        submissions: Object.freeze(submissions.slice()),
        successfulSubmissionCount,
        failure,
        cancellation: canceled ? { reason: String(abortController.signal.reason || 'foreground-opportunity-canceled') } : null,
        metadata,
        authority: 'immediate-queue-submit-outside-sf3d-run-no-gpu-completion-or-presentation-claim',
      });
      state.outsideRun.inFlight.delete(requestId);
      state.outsideRun.receiptCount += 1;
      if (receipt.status === 'completed') state.outsideRun.completedCount += 1; else state.outsideRun.failedCount += 1;
      state.outsideRun.lastReceipt = receipt;
      return receipt;
    })();
    state.outsideRun.inFlight.set(requestId, completion);
    return Object.freeze({
      requestId,
      completion,
      cancel(reason = 'foreground-opportunity-canceled') {
        abortController.abort(String(reason));
        return Object.freeze({ status: 'cancellation-requested', requestId, reason: String(reason) });
      },
    });
  }

  function request(input) {
    if (state.activeRun) {
      validateRequest(input);
      const handle = state.activeRun.interlock.request(input);
      state.activeRun.requestCount += 1;
      return handle;
    }
    return executeOutsideRun(input);
  }

  // --- Active run: kit interlock + idle drain. ---
  function beginRun(runId) {
    if (state.activeRun) {
      throw new Error(`sf3d foreground bridge already has an active run (${state.activeRun.runId})`);
    }
    if (!isNonEmptyString(runId)) throw new Error('runId must be a non-empty string');
    const interlock = createWebGpuForegroundOpportunityInterlock({ routeId, runId, device, queue, now });
    const run = {
      runId,
      interlock,
      active: true,
      requestCount: 0,
      lastServiceAtMs: now(),
      schedulerBoundaryServiceCount: 0,
      idleDrainBoundaryCount: 0,
      idleDrainServicedCount: 0,
      finishDrainServicedCount: 0,
      drainFailures: [],
      drainTimer: null,
      drainInFlight: null,
    };
    state.runCount += 1;
    state.activeRun = run;

    const foregroundOpportunities = Object.freeze({
      schema: WEBGPU_FOREGROUND_OPPORTUNITY_SCHEMA,
      routeId,
      runId,
      async serviceAtBoundary(boundary) {
        const service = await interlock.serviceAtBoundary(boundary);
        run.lastServiceAtMs = now();
        if (service.status !== 'no-demand') run.schedulerBoundaryServiceCount += 1;
        return service;
      },
      pressureSnapshot: () => interlock.pressureSnapshot(),
      snapshot: () => interlock.snapshot(),
    });

    const producerBoundary = (phase, sequence, reason) => ({
      invocationId: `${runId}:${phase}`,
      boundaryId: `${runId}:${phase}:${sequence}`,
      dutyId: `${phase}:${sequence}`,
      phase,
      position: 'before-encode',
      metadata: { runtimeLabel: 'sf3d-producer-foreground-bridge', reason, drainAfterMs, drainIntervalMs },
    });

    const drainTick = async () => {
      run.drainTimer = null;
      if (!run.active) return;
      const pressure = interlock.pressureSnapshot();
      if (pressure.pendingRequestCount > 0 && pressure.activeServiceCount === 0 && pressure.queuedServiceCount === 0
          && now() - run.lastServiceAtMs >= drainAfterMs) {
        run.idleDrainBoundaryCount += 1;
        run.drainInFlight = (async () => {
          try {
            const service = await interlock.serviceAtBoundary(
              producerBoundary(SF3D_FOREGROUND_IDLE_DRAIN_PHASE, run.idleDrainBoundaryCount, 'no-scheduler-boundary-within-budget'));
            run.idleDrainServicedCount += service.servicedRequestCount ?? 0;
          } catch (error) {
            run.drainFailures.push(normalizeError(error));
          } finally {
            run.lastServiceAtMs = now();
            run.drainInFlight = null;
          }
        })();
        await run.drainInFlight;
      }
      if (run.active) run.drainTimer = setTimeout(drainTick, drainIntervalMs);
    };
    run.drainTimer = setTimeout(drainTick, drainIntervalMs);

    async function finish() {
      if (!run.active) throw new Error(`sf3d foreground run ${runId} already finished`);
      run.active = false;
      if (run.drainTimer != null) { clearTimeout(run.drainTimer); run.drainTimer = null; }
      if (run.drainInFlight) await run.drainInFlight;
      let finishService = null;
      if (interlock.pressureSnapshot().pendingRequestCount > 0) {
        finishService = await interlock.serviceAtBoundary(
          producerBoundary(SF3D_FOREGROUND_RUN_FINISH_PHASE, 1, 'run-finished-with-pending-foreground-demand'));
        run.finishDrainServicedCount = finishService.servicedRequestCount ?? 0;
      }
      const report = interlock.finish();
      state.activeRun = null;
      return Object.freeze({
        ...report,
        producer: Object.freeze({
          schema: SF3D_FOREGROUND_BRIDGE_SCHEMA,
          runId,
          requestCount: run.requestCount,
          schedulerBoundaryServiceCount: run.schedulerBoundaryServiceCount,
          idleDrainBoundaryCount: run.idleDrainBoundaryCount,
          idleDrainServicedCount: run.idleDrainServicedCount,
          finishDrainServicedCount: run.finishDrainServicedCount,
          finishDrainStatus: finishService?.status ?? null,
          drainFailures: Object.freeze(run.drainFailures.map(f => ({ ...f }))),
          drainAfterMs,
          drainIntervalMs,
          authority: 'producer-boundary-service-counters-no-presentation-claim',
        }),
      });
    }

    return Object.freeze({ runId, foregroundOpportunities, finish });
  }

  function snapshot() {
    const run = state.activeRun;
    return Object.freeze({
      schema: SF3D_FOREGROUND_BRIDGE_SCHEMA,
      routeId,
      runCount: state.runCount,
      activeRun: run ? Object.freeze({
        runId: run.runId,
        requestCount: run.requestCount,
        pressure: run.interlock.pressureSnapshot(),
        schedulerBoundaryServiceCount: run.schedulerBoundaryServiceCount,
        idleDrainBoundaryCount: run.idleDrainBoundaryCount,
      }) : null,
      outsideRunInFlightCount: state.outsideRun.inFlight.size,
      outsideRunReceiptCount: state.outsideRun.receiptCount,
      outsideRunCompletedCount: state.outsideRun.completedCount,
      outsideRunFailedCount: state.outsideRun.failedCount,
      lastOutsideRunReceipt: state.outsideRun.lastReceipt,
    });
  }

  return Object.freeze({ schema: SF3D_FOREGROUND_BRIDGE_SCHEMA, routeId, request, beginRun, snapshot });
}
