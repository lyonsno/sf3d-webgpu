/**
 * Producer lifecycle: one run at a time, and a dispose() that is safe to call
 * at any time. A host tearing down while a run is outstanding must not leak
 * the run (bridge run open, drain timer firing, second run refused forever) or
 * release buffers the run is still using; the release is deferred until that
 * run ends, and everything new is refused from the moment dispose is requested.
 * Pure (no GPU), so the contract is testable in Node.
 */
export function createProducerLifecycle({ release }) {
  if (typeof release !== 'function') throw new Error('release must be a function');
  const state = { activeRunId: null, disposed: false, released: false };
  const doRelease = () => {
    if (state.released) return;
    state.released = true;
    release();
  };
  return Object.freeze({
    get activeRunId() { return state.activeRunId; },
    get disposed() { return state.disposed; },
    beginRun(runId) {
      if (state.disposed) throw new Error('sf3d producer is disposed');
      if (state.activeRunId != null) throw new Error(`sf3d producer already has an active run (${state.activeRunId})`);
      state.activeRunId = runId;
    },
    /** 'released' when a deferred dispose ran at this run's end, else 'idle'. */
    endRun(runId) {
      if (state.activeRunId !== runId) throw new Error(`${runId} is not the active run (${state.activeRunId ?? 'none'})`);
      state.activeRunId = null;
      if (state.disposed) { doRelease(); return 'released'; }
      return 'idle';
    },
    assertAcceptingRequests() {
      if (state.disposed) throw new Error('sf3d producer is disposed');
    },
    dispose() {
      if (state.disposed) return Object.freeze({ status: 'already-disposed' });
      state.disposed = true;
      if (state.activeRunId != null) return Object.freeze({ status: 'deferred-until-run-ends', runId: state.activeRunId });
      doRelease();
      return Object.freeze({ status: 'released' });
    },
  });
}
