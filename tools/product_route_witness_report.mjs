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

export const PRODUCT_ROUTE_WITNESS_SCHEMA = 'sf3d.product-route-witness.v1';
export const CANONICAL_DEMO_CHAIR_GLB_SHA256 =
  'e1f70de3407df24d571bf68f70fac2b59373bdd948075a2387f1834e4faff8b7';
/** Measured cooperative duty counts for the product route on demo_chair.png (kit submittedGpuDutyCount). */
export const CANONICAL_DEMO_CHAIR_DUTY_COUNTS = Object.freeze({
  'dinov2-tokenizer': 24,
  'two-stream-backbone': 2922,
  'post-processor': 702,
  'texture-bake': 61,
});

/**
 * Exact cooperative identity every requested mechanism must carry. Values are
 * the source modules' constants (asserted equal by the contract test): a report
 * with any other route/manifest/invocation is not this route's evidence.
 */
export const SF3D_ROUTE_ID = 'sf3d.image-to-mesh.webgpu-local.v0';
export const COOPERATIVE_MECHANISM_IDENTITY = Object.freeze({
  'dinov2-tokenizer': Object.freeze({
    flag: 'cooperativeDino', invocationId: 'sf3d:dino:cooperative', completionPolicy: 'strict-prefix',
    manifestIds: Object.freeze({ default: 'sf3d.dino-encoder-cooperative-boundaries.v0' }),
  }),
  'two-stream-backbone': Object.freeze({
    flag: 'cooperativeTwoStream', invocationId: 'sf3d:two-stream:cooperative', completionPolicy: 'strict-prefix',
    granularityOption: 'twoStreamDutyGranularity', defaultGranularity: 'stage',
    manifestIds: Object.freeze({ stage: 'sf3d.two-stream-cooperative-boundaries.v0', 'attention-tile': 'sf3d.two-stream-attention-cooperative-boundaries.v0' }),
  }),
  'post-processor': Object.freeze({
    flag: 'cooperativePostProcessor', invocationId: 'sf3d:post-processor:cooperative',
    completionPolicyOption: 'postProcessorCompletionPolicy', defaultCompletionPolicy: 'strict-prefix',
    granularityOption: 'postProcessorDutyGranularity', defaultGranularity: 'plane',
    manifestIds: Object.freeze({
      plane: 'sf3d.post-processor-cooperative-boundaries.v0',
      layer: 'sf3d.post-processor-layer-cooperative-boundaries.v0',
      'channel-range': 'sf3d.post-processor-channel-cooperative-boundaries.v0',
    }),
  }),
  'texture-bake': Object.freeze({
    flag: 'cooperativeBake', invocationId: 'sf3d:texture-bake:cooperative', completionPolicy: 'strict-prefix',
    manifestIds: Object.freeze({ default: 'sf3d.texture-bake-cooperative-boundaries.v0' }),
  }),
});

/** Expected {routeId, manifestId, invocationId, schedulingMode, completionPolicy} per requested mechanism. */
export function expectedCooperativeIdentity(requested = {}) {
  const out = {};
  for (const [key, id] of Object.entries(COOPERATIVE_MECHANISM_IDENTITY)) {
    if (!requested?.[id.flag]) continue;
    const granularity = id.granularityOption ? (requested[id.granularityOption] ?? id.defaultGranularity) : 'default';
    const manifestId = id.manifestIds[granularity];
    if (!manifestId) throw new Error(`${key}: unknown duty granularity ${granularity}`);
    const completionPolicy = id.completionPolicyOption ? (requested[id.completionPolicyOption] ?? id.defaultCompletionPolicy) : id.completionPolicy;
    out[key] = Object.freeze({ routeId: SF3D_ROUTE_ID, manifestId, invocationId: id.invocationId, schedulingMode: 'cooperative', completionPolicy });
  }
  return Object.freeze(out);
}

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
    // The kit's measured submission count; the duty array length is a second
    // witness of the same fact. Both survive re-projection (the browser projects
    // once to shrink the payload, assembly projects again).
    submittedGpuDutyCount: report.submittedGpuDutyCount ?? null,
    gpuDutyCount: Array.isArray(report.gpuDuties) ? report.gpuDuties.length : (report.gpuDutyCount ?? null),
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
    foregroundOpportunities = null, producerRoute = null,
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
      producerRoute: producerRoute ? Object.freeze({
        invocation: producerRoute.invocation ?? null,
        deviceRelation: producerRoute.deviceRelation ?? null,
        producerResourcesMatchApp: producerRoute.producerResourcesMatchApp ?? null,
        rendererDeviceRelationship: producerRoute.rendererDeviceRelationship ?? 'not-observed-by-this-witness',
        producer: Object.freeze({
          routeId: producerRoute.producer?.routeId ?? null,
          commit: producerRoute.producer?.commit ?? null,
          kitVersion: producerRoute.producer?.kitVersion ?? null,
          deviceInjected: producerRoute.producer?.deviceInjected ?? null,
          deviceTopology: producerRoute.producer?.deviceTopology ?? null,
          backend: producerRoute.producer?.backend ? Object.freeze({ ...producerRoute.producer.backend }) : null,
        }),
        runIdentity: producerRoute.runIdentity ? Object.freeze({ ...producerRoute.runIdentity }) : null,
        browserKit: producerRoute.browserKit ? Object.freeze({ ...producerRoute.browserKit }) : null,
      }) : null,
    }),
    contender: Object.freeze({
      enabled: Boolean(contender?.enabled),
      // 'second-device' (default): a second GPUDevice on the same GPU, submits
      // through its own queue. 'same-device-foreground-opportunity': host frames
      // on SF3D's own device through the producer's foreground-opportunity bridge.
      mode: contender?.mode ?? (contender?.enabled ? 'second-device' : null),
      submitted: contender?.submitted ?? 0,
      completed: contender?.completed ?? 0,
      errors: Object.freeze([...(contender?.errors || [])]),
      receipts: contender?.receipts ? Object.freeze({ ...contender.receipts }) : null,
    }),
    // The common foreground-service finish report, present only for the
    // same-device arm.
    foregroundOpportunities: foregroundOpportunities ? Object.freeze({
      status: foregroundOpportunities.status ?? null,
      requestCount: foregroundOpportunities.requestCount ?? null,
      receiptCount: foregroundOpportunities.receiptCount ?? null,
      pendingRequestCount: foregroundOpportunities.pendingRequestCount ?? null,
      activeRequestCount: foregroundOpportunities.activeRequestCount ?? null,
      // Always 0 for SF3D by construction: the cooperative runtime consults the
      // interlock's pressure before calling serviceAtBoundary, so boundaries
      // without demand never reach the kit (see createSf3dCooperativeRuntime).
      // Carried for the kit's own accounting, not asserted on.
      noDemandBoundaryCount: foregroundOpportunities.noDemandBoundaryCount ?? null,
    }) : null,
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
    expectedDutyCounts = null,
  } = expectations;
  const errors = [];
  if (report?.schema !== PRODUCT_ROUTE_WITNESS_SCHEMA) errors.push(`schema must be ${PRODUCT_ROUTE_WITNESS_SCHEMA}`);

  // Bind the exercised call path and actual producer device to this report.
  // A fresh requestAdapter() probe is not evidence about the producer's device.
  const route = report.effective?.producerRoute;
  if (!route) errors.push('producer route identity is missing');
  else {
    if (!['direct-full-pipeline', 'producer.run'].includes(route.invocation)) errors.push(`producer invocation ${route.invocation ?? 'missing'} is unsupported`);
    if (route.deviceRelation !== 'producer-device===window._sf3d_device') errors.push(`producer/window device relation ${route.deviceRelation ?? 'missing'} != producer-device===window._sf3d_device`);
    if (route.producerResourcesMatchApp !== true) errors.push('producer weights/pipelines do not match the app resources');
    if (route.rendererDeviceRelationship !== 'not-observed-by-this-witness') errors.push(`renderer device relationship ${route.rendererDeviceRelationship ?? 'missing'} is outside this witness contract`);
    const producer = route.producer || {};
    if (producer.routeId !== SF3D_ROUTE_ID) errors.push(`producer routeId ${producer.routeId ?? 'missing'} != expected ${SF3D_ROUTE_ID}`);
    if (!producer.commit || producer.commit !== report.source?.commit) errors.push(`producer commit ${producer.commit ?? 'missing'} != source commit ${report.source?.commit ?? 'missing'}`);
    if (!producer.kitVersion) errors.push('producer kit version is missing');
    else if (producer.kitVersion !== report.source?.kitVersion) errors.push(`producer kit version ${producer.kitVersion} != installed kit version ${report.source?.kitVersion ?? 'missing'}`);
    const expectedTopology = producer.deviceInjected === true ? 'host-injected-device' : producer.deviceInjected === false ? 'producer-owned-device' : null;
    if (!expectedTopology || producer.deviceTopology !== expectedTopology) errors.push(`producer device topology ${producer.deviceTopology ?? 'missing'} disagrees with deviceInjected ${producer.deviceInjected ?? 'missing'}`);
    if (producer.backend?.kind !== 'webgpu-local' || producer.backend?.runtime !== 'browser' || !producer.backend?.adapterName) errors.push('producer backend identity is missing or invalid');
    const browserKit = route.browserKit;
    if (!browserKit) errors.push('browser-executed kit identity is missing');
    else {
      if (browserKit.packageName !== '@kaminos/webgpu-inference-kit') errors.push(`browser kit package ${browserKit.packageName ?? 'missing'} != @kaminos/webgpu-inference-kit`);
      if (browserKit.exportedVersion !== report.source?.kitVersion) errors.push(`browser kit version ${browserKit.exportedVersion ?? 'missing'} != installed kit version ${report.source?.kitVersion ?? 'missing'}`);
      if (!/^[0-9a-f]{64}$/i.test(browserKit.exportFingerprint ?? '')) errors.push('browser kit export fingerprint is missing or invalid');
      if (!Array.isArray(browserKit.servedModules) || browserKit.servedModules.length === 0
        || !Number.isInteger(browserKit.servedModuleCount) || browserKit.servedModuleCount !== browserKit.servedModules.length) {
        errors.push('browser-served kit module bytes are missing');
      } else {
        for (const module of browserKit.servedModules) {
          if (typeof module.url !== 'string' || !module.url.includes('/node_modules/')
            || !Number.isInteger(module.bytes) || module.bytes <= 0
            || !/^[0-9a-f]{64}$/i.test(module.sha256 ?? '')) {
            errors.push('browser-served kit module bytes are missing or invalid');
            break;
          }
        }
      }
      if (!/^[0-9a-f]{64}$/i.test(browserKit.servedModuleSetSha256 ?? '')) errors.push('browser-served kit module set digest is missing or invalid');
      if (typeof browserKit.witnessModuleUrl !== 'string' || !browserKit.witnessModuleUrl) errors.push('browser kit witness module URL is missing');
    }
    if (route.invocation === 'producer.run') {
      const identity = route.runIdentity;
      if (!identity) errors.push('producer.run identity is missing');
      else {
        if (identity.schema !== 'sf3d.producer-run-identity.v0') errors.push(`producer run identity schema ${identity.schema ?? 'missing'} is invalid`);
        if (identity.routeId !== producer.routeId) errors.push(`producer run routeId ${identity.routeId ?? 'missing'} != producer routeId ${producer.routeId ?? 'missing'}`);
        if (identity.deviceTopology !== producer.deviceTopology) errors.push(`producer run deviceTopology ${identity.deviceTopology ?? 'missing'} != producer deviceTopology ${producer.deviceTopology ?? 'missing'}`);
        if (identity.producerCommit !== producer.commit) errors.push(`producer run commit ${identity.producerCommit ?? 'missing'} != producer commit ${producer.commit ?? 'missing'}`);
        if (identity.kitVersion !== producer.kitVersion) errors.push(`producer run kit version ${identity.kitVersion ?? 'missing'} != producer kit version ${producer.kitVersion ?? 'missing'}`);
      }
    } else if (route.runIdentity != null) errors.push('direct-full-pipeline must not claim a producer.run identity');
  }

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
  let expectedIdentity = {};
  try { expectedIdentity = expectedCooperativeIdentity(report.requested || {}); } catch (e) { errors.push(e.message); }
  for (const [flag, key] of Object.entries(COOPERATIVE_MECHANISM_TO_REPORT)) {
    if (!report.requested?.[flag]) continue;
    const c = report.effective?.cooperative?.[key];
    if (!c) { errors.push(`${flag} requested but no cooperative report for ${key}`); continue; }
    if (c.status !== 'succeeded') errors.push(`${key} cooperative report status ${c.status ?? 'missing'} != succeeded`);
    if (c.schedulingMode !== 'cooperative') errors.push(`${key} scheduling mode ${c.schedulingMode ?? 'missing'} != cooperative`);
    // Identity binding: the report must be THIS mechanism's evidence (exact
    // route / manifest for the requested granularity / invocation / policy).
    const exp = expectedIdentity[key];
    if (exp) {
      for (const field of ['routeId', 'manifestId', 'invocationId', 'completionPolicy']) {
        if (c[field] !== exp[field]) errors.push(`${key} ${field} ${c[field] ?? 'missing'} != expected ${exp[field]}`);
      }
    }
    if (finite(c.inFlightGpuDutyCount) && c.inFlightGpuDutyCount !== 0) errors.push(`${key} left ${c.inFlightGpuDutyCount} GPU duties in flight`);
    if (finite(c.issuedGpuDutyCount) && finite(c.retiredGpuDutyCount) && c.issuedGpuDutyCount !== c.retiredGpuDutyCount) {
      errors.push(`${key} issued ${c.issuedGpuDutyCount} != retired ${c.retiredGpuDutyCount}`);
    }
    // Strict-prefix boundaries do not carry issued/retired counters; their
    // settlement evidence is complete denominator-bearing progress.
    if (c.progress && finite(c.progress.totalItems)) {
      if (c.progress.completedItems !== c.progress.totalItems) {
        errors.push(`${key} progress ${c.progress.completedItems}/${c.progress.totalItems} incomplete`);
      }
    } else {
      errors.push(`${key} cooperative report carries no denominator-bearing progress`);
    }
    // Kit validation is mandatory for every requested mechanism: absence is a
    // rejection, not a silently lower authority.
    const v = report.effective?.cooperativeValidations?.[key];
    if (!v) errors.push(`${key} requested but no kit validation record (validateWebGpuCooperativeExecutionReport) was preserved`);
    else if (v.ok !== true) errors.push(`${key} kit validation failed: ${(v.errors || []).join('; ')}`);
  }

  // Contender.
  if (requireContender && !report.contender?.enabled) errors.push('contender required but not enabled');
  if (report.contender?.enabled) {
    if (!(report.contender.completed > 0)) errors.push('contender enabled but completed zero submissions during the run');
    if (report.contender.errors?.length) errors.push(`contender errors: ${report.contender.errors.join('; ')}`);
    const receipts = report.contender.receipts;
    if (receipts) {
      const c = report.contender;
      if (receipts.failed > 0) errors.push(`contender receipts: ${receipts.failed} failed`);
      if (receipts.canceled > 0) errors.push(`contender receipts: ${receipts.canceled} canceled`);
      // Accounting: every submission settled, every completion located, and the
      // in-run receipts equal the producer's own foreground report.
      if (c.completed !== c.submitted) errors.push(`contender completed ${c.completed} != submitted ${c.submitted}`);
      const settled = receipts.completed + receipts.failed + receipts.canceled;
      if (settled !== c.submitted) errors.push(`contender receipts ${receipts.completed}+${receipts.failed}+${receipts.canceled} != submitted ${c.submitted}`);
      if (receipts.completed !== c.completed) errors.push(`contender receipts.completed ${receipts.completed} != completed ${c.completed}`);
      const located = receipts.outsideRun + receipts.schedulerBoundary + receipts.foregroundWindow + receipts.runFinish;
      if (located !== receipts.completed) errors.push(`contender service locations ${located} != completed ${receipts.completed}`);
      const inRun = receipts.schedulerBoundary + receipts.foregroundWindow + receipts.runFinish;
      if (!(inRun > 0)) errors.push('same-device contender serviced zero host frames inside the run');
      const fg = report.foregroundOpportunities;
      if (fg) {
        if (fg.requestCount !== inRun) errors.push(`foreground requestCount ${fg.requestCount} != in-run contender receipts ${inRun}`);
        if (fg.receiptCount !== fg.requestCount) errors.push(`foreground receiptCount ${fg.receiptCount} != requestCount ${fg.requestCount}`);
      }
    }
    if (report.contender.mode === 'same-device-foreground-opportunity' && !report.foregroundOpportunities) {
      errors.push('same-device contender requires a foreground opportunity report from the producer');
    }
  }

  // Measured duty counts (product route on the canonical image): the kit's
  // submittedGpuDutyCount must be preserved and must equal the pinned counts.
  if (expectedDutyCounts) {
    for (const [key, expected] of Object.entries(expectedDutyCounts)) {
      const c = report.effective?.cooperative?.[key];
      if (!c) { errors.push(`${key} has no cooperative report to pin ${expected} GPU duties against`); continue; }
      const submitted = c.submittedGpuDutyCount ?? c.gpuDutyCount;
      if (!finite(submitted)) { errors.push(`${key} preserves no submitted GPU duty count (expected ${expected})`); continue; }
      if (submitted !== expected) errors.push(`${key} submitted ${submitted} GPU duties, expected ${expected}`);
    }
  }

  // Foreground-opportunity report (same-device host frames): every host
  // request must have been serviced (none pending/active at finish) and the
  // common service must have settled every receipt.
  const fg = report.foregroundOpportunities;
  if (fg) {
    if (fg.status !== 'succeeded') errors.push(`foreground opportunity report status ${fg.status} (demand left unsettled at run finish)`);
  }

  // Budget.
  if (finite(maxGapBudgetMs) && wr && finite(wr.maxMs) && wr.maxMs > maxGapBudgetMs) {
    errors.push(`whole-route max frame gap ${wr.maxMs}ms exceeds budget ${maxGapBudgetMs}ms`);
  }

  return Object.freeze({ ok: errors.length === 0, errors: Object.freeze(errors) });
}
