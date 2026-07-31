// bounded_prefix_report_fixture.mjs
//
// Test-helper fixtures for the @kaminos/webgpu-inference-kit cooperative
// execution-report validator (bounded-prefix completion policy).
//
// makeValidBoundedReport(overrides) returns a fresh plain-object report that
// PASSES validateWebGpuCooperativeExecutionReport under the sf3d post-processor
// bounded-prefix expectations (702 GPU duties, maxInFlight 2). The named
// mutation helpers each return a MUTATED DEEP COPY that must make the validator
// FAIL, so tests can assert specific failure strings.
//
// Shapes are derived directly from the report BUILDER in
//   node_modules/@kaminos/webgpu-inference-kit/src/cooperative-execution.js
// (createReport / registerBoundedGpuDuty / retireOldestGpuDuty) and the checks
// in
//   node_modules/@kaminos/webgpu-inference-kit/src/cooperative-report-validation.js
// (validateCounts / validateBoundedDuties / validateTerminalProgress /
//  validateBoundaries).
//
// Key contract facts encoded below (all read out of the two source files):
//   * schema string:                'kaminos.webgpu-cooperative-execution-report.v0'
//   * progress schema string:       'kaminos.webgpu-cooperative-progress.v0'
//   * retired-duty timingAuthority: 'queue-work-done'   (validateBoundedDuties,
//       allowedTimingAuthorities for non-'failed' duties)
//   * queueCompletionAuthority:     'bounded-per-gpu-duty-prefix-fence'
//   * bounded-prefix + succeeded expected count fields (validateCounts):
//       issuedGpuDutyCount, retiredGpuDutyCount, submittedGpuDutyCount
//       (all must == expectedGpuDutyCount)
//   * observedPrefixFenceCount must equal submittedGpuDutyCount
//   * inFlightGpuDutyCount==0, inFlightGpuDutyIds==[], unfencedSubmittedGpuDutyCount==0

const GPU_DUTY_COUNT = 702;
const MAX_IN_FLIGHT = 2;

const REPORT_SCHEMA = 'kaminos.webgpu-cooperative-execution-report.v0';
const PROGRESS_SCHEMA = 'kaminos.webgpu-cooperative-progress.v0';

const ROUTE_ID = 'sf3d.image-to-mesh.webgpu-local.v0';
const MANIFEST_ID = 'sf3d.post-processor-channel-cooperative-boundaries.v0';
const INVOCATION_ID = 'sf3d:post-processor:cooperative';
const BOUNDARY_ID = 'sf3d.post-processor.gpu-command.v0';
const PHASE_ID = 'sf3d.post-processor.phase.v0';

// Fixed timestamps only — no Date.now() (harness forbids wall-clock use).
const STARTED_AT_MS = 1000;
const ENDED_AT_MS = 2000;
const DURATION_MS = ENDED_AT_MS - STARTED_AT_MS;

// Structured-clone deep copy (matches the kit's own JSON clone semantics).
function deepCopy(value) {
  return JSON.parse(JSON.stringify(value));
}

// One GPU-command range per duty. Item coverage must be contiguous starting at
// 0 with positive itemCount and itemEnd === itemStart + itemCount
// (validateBoundaryRanges requireComplete branch). One item per range keeps the
// arithmetic trivial and totalItems === GPU_DUTY_COUNT.
function buildRanges() {
  const ranges = [];
  for (let i = 0; i < GPU_DUTY_COUNT; i += 1) {
    ranges.push({
      rangeId: `${BOUNDARY_ID}:${i}`,
      rangeIndex: i,
      itemStart: i,
      itemCount: 1,
      itemEnd: i + 1,
    });
  }
  return ranges;
}

// One retired gpuDuty per completed GPU-command range. submittedAtMs/retiredAtMs
// derive from the index via fixed arithmetic (no wall clock); rawQueueDurationMs
// must equal retiredAtMs - submittedAtMs exactly (sameDuration check).
function buildGpuDuties() {
  const duties = [];
  for (let i = 0; i < GPU_DUTY_COUNT; i += 1) {
    const submittedAtMs = STARTED_AT_MS + i;
    const retiredAtMs = submittedAtMs + 1;
    duties.push({
      dutyId: `${BOUNDARY_ID}:${i}`,
      dutyIndex: i,
      status: 'retired',
      rangeId: `${BOUNDARY_ID}:${i}`,
      rangeIndex: i,
      boundaryId: BOUNDARY_ID,
      submittedAtMs,
      retiredAtMs,
      rawQueueDurationMs: retiredAtMs - submittedAtMs, // = 1, finite nonnegative
      timingAuthority: 'queue-work-done',
      failure: null,
    });
  }
  return duties;
}

// Terminal progress must prove 702/702 completion at 100% with matching
// routeId/invocationId/status (validateTerminalProgress).
function buildProgress() {
  return {
    schema: PROGRESS_SCHEMA,
    routeId: ROUTE_ID,
    invocationId: INVOCATION_ID,
    status: 'succeeded',
    completedItems: GPU_DUTY_COUNT,
    totalItems: GPU_DUTY_COUNT,
    progress: 1,
    percent: 100,
    phases: [
      {
        phaseId: PHASE_ID,
        status: 'complete',
        completedItems: GPU_DUTY_COUNT,
        totalItems: GPU_DUTY_COUNT,
        progress: 1,
        percent: 100,
      },
    ],
  };
}

// Single completed gpu-command boundary carrying all 702 ranges. On success the
// boundary must be complete with completedItems===totalItems, progress===1,
// actualRangeCount===ranges.length, rangeCountAuthority==='actual'.
function buildBoundaries(ranges) {
  return [
    {
      phaseId: PHASE_ID,
      boundaryId: BOUNDARY_ID,
      kind: 'gpu-command',
      unit: 'command-duty',
      status: 'complete',
      completedItems: GPU_DUTY_COUNT,
      totalItems: GPU_DUTY_COUNT,
      progress: 1,
      progressWeight: 1,
      rangeCount: ranges.length,
      actualRangeCount: ranges.length,
      rangeCountAuthority: 'actual',
      ranges: deepCopy(ranges),
      planner: null,
      failure: null,
    },
  ];
}

export function makeValidBoundedReport(overrides = {}) {
  const ranges = buildRanges();
  const report = {
    schema: REPORT_SCHEMA,
    status: 'succeeded',
    routeId: ROUTE_ID,
    manifestId: MANIFEST_ID,
    invocationId: INVOCATION_ID,
    schedulingMode: 'cooperative',
    completionPolicy: 'bounded-prefix',

    // bounded-prefix depth accounting
    maxInFlightGpuDuties: MAX_IN_FLIGHT,
    maxObservedInFlightGpuDuties: MAX_IN_FLIGHT, // must equal configured depth
    queueCompletionAuthority: 'bounded-per-gpu-duty-prefix-fence',

    // settlement counts — all internally consistent for a succeeded run
    issuedGpuDutyCount: GPU_DUTY_COUNT,
    retiredGpuDutyCount: GPU_DUTY_COUNT,
    inFlightGpuDutyCount: 0,
    inFlightGpuDutyIds: [],
    submittedGpuDutyCount: GPU_DUTY_COUNT,
    observedPrefixFenceCount: GPU_DUTY_COUNT, // == submittedGpuDutyCount
    unfencedSubmittedGpuDutyCount: 0,

    gpuDuties: buildGpuDuties(),

    retention: 'uncapped',
    startedAtMs: STARTED_AT_MS,
    endedAtMs: ENDED_AT_MS,
    durationMs: DURATION_MS,

    failure: null,

    progress: buildProgress(),
    boundaries: buildBoundaries(ranges),

    lastTrustworthyBoundary: null, // only checked on non-success paths
  };

  return { ...report, ...overrides };
}

// ---------------------------------------------------------------------------
// Mutation helpers. Each returns a deep copy with the original untouched, and
// each violates exactly one bounded-prefix invariant so the validator fails.
// ---------------------------------------------------------------------------

// observedPrefixFenceCount must equal submittedGpuDutyCount. Collapsing the
// fence count to 1 breaks the bounded prefix-fence settlement contract.
export function withSingleFence(report) {
  const copy = deepCopy(report);
  copy.observedPrefixFenceCount = 1;
  return copy;
}

// Terminal progress must be an object proving completion. Null progress trips
// validateTerminalProgress immediately.
export function withMissingProgress(report) {
  const copy = deepCopy(report);
  copy.progress = null;
  return copy;
}

// Each retired duty's rawQueueDurationMs must be finite nonnegative. null trips
// isFiniteNonnegative. (JSON cannot carry NaN/Infinity, so null is the durable
// non-finite sentinel.)
export function withNonfiniteRawQueueDuration(report) {
  const copy = deepCopy(report);
  copy.gpuDuties[0].rawQueueDurationMs = null;
  return copy;
}

// unfencedSubmittedGpuDutyCount must be zero (validateCounts).
export function withUnfencedSubmission(report) {
  const copy = deepCopy(report);
  copy.unfencedSubmittedGpuDutyCount = 3;
  return copy;
}

// Terminal in-flight work must be zero: inFlightGpuDutyCount and
// inFlightGpuDutyIds must both be empty (validateCounts).
export function withResidualInFlight(report) {
  const copy = deepCopy(report);
  copy.inFlightGpuDutyCount = 1;
  copy.inFlightGpuDutyIds = [`${BOUNDARY_ID}:0`];
  return copy;
}
