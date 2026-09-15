/**
 * product_route_witness_report.mjs — pure assembly + acceptance for the
 * product-route foreground-liveness witness (tools/smoke_product_route.mjs).
 *
 * No browser, no GPU: this module turns raw probe data (rAF frame intervals,
 * absolute stage spans, cooperative execution reports, offload records, output
 * hash) into one durable report and decides whether it is acceptable. Keeping
 * it pure means the falsifiers below can be exercised deterministically —
 * the witness must be able to lie and be caught:
 *   - the GLB hash drifting from the canonical output;
 *   - a requested CPU offload silently running on the main thread;
 *   - a requested cooperative mechanism with no settled kit report;
 *   - a probe that recorded no frames, or an inference window that never closed;
 *   - a hidden page (rAF throttled → fake smoothness);
 *   - a contender that was requested but never completed a submission;
 *   - a non-finite gap; a max-gap budget breach.
 */

export const PRODUCT_ROUTE_WITNESS_SCHEMA = 'sf3d.product-route-witness.v0';
export const CANONICAL_DEMO_CHAIR_GLB_SHA256 =
  'e1f70de3407df24d571bf68f70fac2b59373bdd948075a2387f1834e4faff8b7';

/** Worker option role → key in the full-pipeline `offloads` record. */
export const WORKER_ROLE_TO_OFFLOAD = Object.freeze({
  preprocessWorker: 'preprocess',
  clipPrepWorker: 'clipPrep',
  marchingTetWorker: 'marchingTet',
  uvUnwrapWorker: 'uvUnwrap',
  materializeWorker: 'materialize',
});

/** Mechanism flag in the options description → cooperative report key. */
export const COOPERATIVE_MECHANISM_TO_REPORT = Object.freeze({
  cooperativeDino: 'dinov2-tokenizer',
  cooperativeTwoStream: 'two-stream-backbone',
  cooperativePostProcessor: 'post-processor',
  cooperativeBake: 'texture-bake',
});

const finite = (v) => typeof v === 'number' && Number.isFinite(v);

function requireFiniteInterval(interval, i) {
  if (!interval || !finite(interval.start) || !finite(interval.end) || interval.end < interval.start) {
    throw new RangeError(`frame interval ${i} is not a finite ordered [start, end]`);
  }
}

/** Clip frame intervals to the inference window; drop zero-length remainders. */
export function scopeFrameIntervals(intervals, window) {
  if (!window || !finite(window.startMs) || !finite(window.endMs) || window.endMs <= window.startMs) {
    throw new RangeError('inference window must be a finite [startMs, endMs] with endMs > startMs');
  }
  const scoped = [];
  intervals.forEach((f, i) => {
    requireFiniteInterval(f, i);
    const start = Math.max(f.start, window.startMs);
    const end = Math.min(f.end, window.endMs);
    if (end > start) scoped.push({ start, end, gap: end - start });
  });
  return scoped;
}

/** Nearest-rank percentile over an ascending array. */
export function percentile(sortedAsc, p) {
  if (!sortedAsc.length) return null;
  const idx = Math.min(sortedAsc.length - 1, Math.max(0, Math.ceil((p / 100) * sortedAsc.length) - 1));
  return sortedAsc[idx];
}

export function summarizeFrameGaps(intervals, window) {
  const scoped = scopeFrameIntervals(intervals, window);
  const gaps = scoped.map(f => f.gap).sort((a, b) => a - b);
  const worst = scoped.slice().sort((a, b) => b.gap - a.gap).slice(0, 5)
    .map(f => ({ startMs: +f.start.toFixed(1), endMs: +f.end.toFixed(1), gapMs: +f.gap.toFixed(1) }));
  return Object.freeze({
    frameIntervalCount: gaps.length,
    p50Ms: gaps.length ? +percentile(gaps, 50).toFixed(1) : null,
    p95Ms: gaps.length ? +percentile(gaps, 95).toFixed(1) : null,
    p99Ms: gaps.length ? +percentile(gaps, 99).toFixed(1) : null,
    maxMs: gaps.length ? +gaps[gaps.length - 1].toFixed(1) : null,
    over16_7: gaps.filter(g => g > 16.7).length,
    over33_3: gaps.filter(g => g > 33.3).length,
    over100: gaps.filter(g => g > 100).length,
    worst,
  });
}

/**
 * Attribute each scoped frame gap to the stage span containing its midpoint
 * (same rule as tools/profile_foreground_tail.mjs). Spans may overlap when a
 * stage awaits GPU work queued by an earlier stage; the first match wins and
 * the ambiguity is preserved by also reporting the span durations.
 */
export function attributeGapsToStages(intervals, spans, window) {
  const scoped = scopeFrameIntervals(intervals, window);
  const byStage = {};
  const bucket = (name) => byStage[name] || (byStage[name] = { gapCount: 0, summedGapMs: 0, maxGapMs: 0, over33_3: 0 });
  for (const f of scoped) {
    const mid = (f.start + f.end) / 2;
    const hit = spans.find(s => mid >= s.start && mid < s.end);
    const name = hit ? hit.name : (spans.length && mid < spans[0].start ? 'before-pipeline' : 'unattributed');
    const b = bucket(name);
    b.gapCount += 1;
    b.summedGapMs += f.gap;
    b.maxGapMs = Math.max(b.maxGapMs, f.gap);
    if (f.gap > 33.3) b.over33_3 += 1;
  }
  for (const s of spans) {
    const b = bucket(s.name);
    b.durationMs = +(s.end - s.start).toFixed(1);
  }
  for (const b of Object.values(byStage)) {
    b.summedGapMs = +b.summedGapMs.toFixed(1);
    b.maxGapMs = +b.maxGapMs.toFixed(1);
  }
  const ranked = Object.entries(byStage)
    .filter(([, b]) => b.gapCount > 0)
    .sort((a, b) => b[1].maxGapMs - a[1].maxGapMs)
    .map(([name]) => name);
  return Object.freeze({ byStage: Object.freeze(byStage), rankedByMaxGap: Object.freeze(ranked) });
}

/** Project a kit cooperative execution report to the settlement facts the witness needs. */
export function projectCooperativeReport(report) {
  if (!report || typeof report !== 'object') return null;
  return Object.freeze({
    status: report.status ?? null,
    schedulingMode: report.schedulingMode ?? report.effectiveSchedulingMode ?? null,
    completionPolicy: report.completionPolicy ?? null,
    routeId: report.routeId ?? null,
    manifestId: report.manifestId ?? null,
    invocationId: report.invocationId ?? null,
    issuedGpuDutyCount: report.issuedGpuDutyCount ?? null,
    retiredGpuDutyCount: report.retiredGpuDutyCount ?? null,
    inFlightGpuDutyCount: report.inFlightGpuDutyCount ?? null,
    maxObservedInFlightGpuDuties: report.maxObservedInFlightGpuDuties ?? null,
    gpuDutyCount: Array.isArray(report.gpuDuties) ? report.gpuDuties.length : null,
    progress: report.progress
      ? { completedItems: report.progress.completedItems ?? null, totalItems: report.progress.totalItems ?? null, percent: report.progress.percent ?? null }
      : null,
  });
}

/**
 * Assemble the durable witness report from raw probe data. Throws on
 * structurally impossible input; acceptance (below) is where policy lives.
 */
export function assembleProductRouteWitness(input) {
  const {
    arm, source, requestedOptions, offloads, cooperativeReports = {}, cooperativeValidations = {},
    materializationOffloaded = null, frames, stageSpans, inferenceWindow, visibility,
    output, contender, stageTimings = {}, totalMs, generatedAt = new Date().toISOString(),
  } = input;
  if (typeof arm !== 'string' || !arm) throw new TypeError('arm must be a non-empty string');
  if (!Array.isArray(frames)) throw new TypeError('frames must be an array');
  if (!Array.isArray(stageSpans)) throw new TypeError('stageSpans must be an array');
  const wholeRoute = summarizeFrameGaps(frames, inferenceWindow);
  const attribution = attributeGapsToStages(frames, stageSpans, inferenceWindow);
  const cooperative = {};
  for (const [key, report] of Object.entries(cooperativeReports)) cooperative[key] = projectCooperativeReport(report);
  return Object.freeze({
    schema: PRODUCT_ROUTE_WITNESS_SCHEMA,
    generatedAt,
    arm,
    source: Object.freeze({ ...source }),
    requested: Object.freeze({ ...requestedOptions }),
    effective: Object.freeze({
      offloads: Object.freeze({ ...offloads }),
      cooperative: Object.freeze(cooperative),
      cooperativeValidations: Object.freeze({ ...cooperativeValidations }),
      materializationOffloaded,
      visibility: visibility ?? null,
    }),
    contender: Object.freeze({
      enabled: Boolean(contender?.enabled),
      submitted: contender?.submitted ?? 0,
      completed: contender?.completed ?? 0,
      errors: Object.freeze([...(contender?.errors || [])]),
    }),
    inferenceWindow: Object.freeze({ startMs: inferenceWindow?.startMs ?? null, endMs: inferenceWindow?.endMs ?? null }),
    totalMs: finite(totalMs) ? +totalMs.toFixed(1) : null,
    output: Object.freeze({ ...output }),
    cadence: Object.freeze({ wholeRoute, byStage: attribution.byStage, rankedByMaxGap: attribution.rankedByMaxGap }),
    stageTimingsMs: Object.freeze({ ...stageTimings }),
  });
}

/**
 * Decide whether a witness is acceptable. Every check names its failure so a
 * red report explains itself. `expectations`:
 *   expectedGlbSha (string|null), minFrames (default 100), maxGapBudgetMs (null = no gate),
 *   requireContender (bool), requireVisible (default true).
 */
export function acceptProductRouteWitness(report, expectations = {}) {
  const {
    expectedGlbSha = CANONICAL_DEMO_CHAIR_GLB_SHA256,
    minFrames = 100,
    maxGapBudgetMs = null,
    requireContender = false,
    requireVisible = true,
  } = expectations;
  const errors = [];
  if (report?.schema !== PRODUCT_ROUTE_WITNESS_SCHEMA) errors.push(`schema must be ${PRODUCT_ROUTE_WITNESS_SCHEMA}`);

  // Output identity.
  if (expectedGlbSha) {
    if (report.output?.glbSha256 !== expectedGlbSha) {
      errors.push(`glb sha ${report.output?.glbSha256 ?? 'missing'} != expected ${expectedGlbSha}`);
    }
  }
  if (!finite(report.output?.glbBytes) || report.output.glbBytes <= 0) errors.push('glb byte length missing or zero');

  // Probe validity.
  const w = report.inferenceWindow;
  if (!finite(w?.startMs) || !finite(w?.endMs) || w.endMs <= w.startMs) errors.push('inference window did not open and close');
  const wr = report.cadence?.wholeRoute;
  if (!wr || wr.frameIntervalCount < minFrames) errors.push(`only ${wr?.frameIntervalCount ?? 0} frame intervals inside the inference window (need >= ${minFrames})`);
  for (const k of ['p50Ms', 'p95Ms', 'p99Ms', 'maxMs']) {
    if (wr && wr.frameIntervalCount > 0 && !finite(wr[k])) errors.push(`cadence ${k} is not finite`);
  }
  if (requireVisible && report.effective?.visibility !== 'visible') errors.push(`page visibility was ${report.effective?.visibility ?? 'unknown'}, not visible (rAF cadence would be throttled)`);

  // Requested offloads must have been effective.
  const workers = report.requested?.workers || {};
  for (const [role, requested] of Object.entries(workers)) {
    const key = WORKER_ROLE_TO_OFFLOAD[role];
    if (!key) continue;
    const effective = report.effective?.offloads?.[key];
    if (requested && effective !== 'worker') errors.push(`${role} requested but ${key} ran on ${effective ?? 'unknown'}`);
  }
  if (workers.materializeWorker && report.requested?.cooperativeBake && report.effective?.materializationOffloaded !== true) {
    errors.push('materialize worker requested but bake telemetry does not report materializationOffloaded');
  }

  // Requested cooperative mechanisms must have a settled report.
  for (const [flag, key] of Object.entries(COOPERATIVE_MECHANISM_TO_REPORT)) {
    if (!report.requested?.[flag]) continue;
    const c = report.effective?.cooperative?.[key];
    if (!c) { errors.push(`${flag} requested but no cooperative report for ${key}`); continue; }
    if (c.status !== 'succeeded') errors.push(`${key} cooperative report status ${c.status ?? 'missing'} != succeeded`);
    if (c.schedulingMode !== 'cooperative') errors.push(`${key} scheduling mode ${c.schedulingMode ?? 'missing'} != cooperative`);
    if (finite(c.inFlightGpuDutyCount) && c.inFlightGpuDutyCount !== 0) errors.push(`${key} left ${c.inFlightGpuDutyCount} GPU duties in flight`);
    if (finite(c.issuedGpuDutyCount) && finite(c.retiredGpuDutyCount) && c.issuedGpuDutyCount !== c.retiredGpuDutyCount) {
      errors.push(`${key} issued ${c.issuedGpuDutyCount} != retired ${c.retiredGpuDutyCount}`);
    }
    const v = report.effective?.cooperativeValidations?.[key];
    if (v && v.ok !== true) errors.push(`${key} kit validation failed: ${(v.errors || []).join('; ')}`);
  }

  // Contender.
  if (requireContender && !report.contender?.enabled) errors.push('contender required but not enabled');
  if (report.contender?.enabled) {
    if (!(report.contender.completed > 0)) errors.push('contender enabled but completed zero submissions during the run');
    if (report.contender.errors?.length) errors.push(`contender errors: ${report.contender.errors.join('; ')}`);
  }

  // Budget.
  if (finite(maxGapBudgetMs) && wr && finite(wr.maxMs) && wr.maxMs > maxGapBudgetMs) {
    errors.push(`whole-route max frame gap ${wr.maxMs}ms exceeds budget ${maxGapBudgetMs}ms`);
  }

  return Object.freeze({ ok: errors.length === 0, errors: Object.freeze(errors) });
}
