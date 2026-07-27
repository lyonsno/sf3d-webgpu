const round = (value, digits = 1) => {
  const scale = 10 ** digits;
  return Math.round((value + Number.EPSILON) * scale) / scale;
};

const requireFinite = (value, label) => {
  if (!Number.isFinite(value)) throw new TypeError(`${label} must be finite`);
  return value;
};

export function parsePositiveSafeIntegerOption(rawValue, {
  label,
  defaultValue,
} = {}) {
  if (typeof label !== 'string' || !label.trim()) {
    throw new TypeError('option label must be non-empty');
  }
  const defaulted = rawValue === undefined;
  const value = defaulted ? defaultValue : rawValue;
  if (
    (typeof value !== 'string' && typeof value !== 'number')
    || (typeof value === 'string' && !/^[1-9]\d*$/.test(value))
  ) {
    throw new TypeError(`${label} must be a positive safe integer`);
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new TypeError(`${label} must be a positive safe integer`);
  }
  return Object.freeze({
    requested: defaulted ? null : String(rawValue),
    effective: parsed,
    defaulted,
  });
}

export function validateGlbPayload(payload) {
  if (!(payload instanceof Uint8Array)) {
    throw new TypeError('GLB payload must be a Uint8Array');
  }
  if (payload.byteLength < 12) {
    throw new TypeError('GLB payload must contain a complete 12-byte header');
  }
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const magic = view.getUint32(0, true);
  const version = view.getUint32(4, true);
  const declaredBytes = view.getUint32(8, true);
  if (magic !== 0x46546c67) throw new TypeError('GLB payload has invalid magic');
  if (version !== 2) throw new TypeError(`GLB payload has unsupported version ${version}`);
  if (declaredBytes !== payload.byteLength) {
    throw new TypeError(`GLB payload declares ${declaredBytes} bytes but contains ${payload.byteLength}`);
  }
  return Object.freeze({
    byteLength: payload.byteLength,
    version,
  });
}

const intervalOverlap = (startA, endA, startB, endB) => (
  Math.max(0, Math.min(endA, endB) - Math.max(startA, startB))
);

const percentile = (sorted, fraction) => {
  if (sorted.length === 0) return null;
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)];
};

const median = (values, label) => {
  if (!Array.isArray(values) || values.length === 0) throw new TypeError(`${label} must be non-empty`);
  const sorted = values.map((value, index) => requireFinite(value, `${label}[${index}]`)).sort((a, b) => a - b);
  const midpoint = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[midpoint - 1] + sorted[midpoint]) / 2
    : sorted[midpoint];
};

const summarizeGaps = (gaps) => {
  const sorted = [...gaps].sort((a, b) => a - b);
  return Object.freeze({
    frameIntervalCount: sorted.length,
    p50Ms: sorted.length ? round(percentile(sorted, 0.50)) : null,
    p95Ms: sorted.length ? round(percentile(sorted, 0.95)) : null,
    p99Ms: sorted.length ? round(percentile(sorted, 0.99)) : null,
    maxMs: sorted.length ? round(sorted.at(-1)) : null,
    over16_7Count: sorted.filter(value => value > 16.7).length,
    over33_3Count: sorted.filter(value => value > 33.3).length,
    over100Count: sorted.filter(value => value > 100).length,
  });
};

const normalizeFrames = (frames) => frames.map((frame, index) => {
  const start = requireFinite(frame?.start, `frames[${index}].start`);
  const end = requireFinite(frame?.end, `frames[${index}].end`);
  const gap = requireFinite(frame?.gap, `frames[${index}].gap`);
  if (end < start || gap < 0) throw new RangeError(`frames[${index}] has an invalid interval`);
  return { start, end, gap };
});

const normalizeSpans = (spans) => {
  if (!Array.isArray(spans) || spans.length === 0) {
    throw new TypeError('output.stageSpans must contain at least one stage');
  }
  const seen = new Set();
  return spans.map((span, index) => {
    const name = typeof span?.name === 'string' ? span.name.trim() : '';
    if (!name || seen.has(name)) throw new TypeError(`output.stageSpans[${index}] has an invalid name`);
    seen.add(name);
    const start = requireFinite(span.start, `output.stageSpans[${index}].start`);
    const end = requireFinite(span.end, `output.stageSpans[${index}].end`);
    if (end < start) throw new RangeError(`output.stageSpans[${index}] has an invalid interval`);
    return { name, start, end };
  });
};

export function normalizeArmExecution(options = {}) {
  const cooperativeBake = options.cooperativeBake === true;
  const requestedBatch = cooperativeBake ? options.bakeBatchTexels : null;
  const bakeBatchTexels = Number.isSafeInteger(requestedBatch) && requestedBatch > 0
    ? requestedBatch
    : null;
  return Object.freeze({
    cooperativeDino: options.cooperativeDino === true,
    cooperativeBake,
    bakeSchedulingMode: cooperativeBake
      ? (options.bakeSchedulingMode === 'disabled' ? 'disabled' : 'cooperative')
      : 'monolithic',
    bakeBatchTexels,
    decoderArena: options.decoderArena === true,
    materializeWorker: options.materializeWorker != null,
  });
}

export function buildFullRouteArmReceipt({
  name,
  ordinal,
  options,
  output,
  frames,
  pipelineStartMs,
  pipelineEndMs,
}) {
  if (typeof name !== 'string' || !name.trim()) throw new TypeError('name must be non-empty');
  if (!Number.isSafeInteger(ordinal) || ordinal < 1) throw new TypeError('ordinal must be a positive integer');
  const totalMs = requireFinite(output?.totalMs, 'output.totalMs');
  if (totalMs <= 0) throw new RangeError('output.totalMs must be positive');
  const glbBytes = output?.glb?.byteLength;
  if (!Number.isSafeInteger(glbBytes) || glbBytes <= 0) {
    throw new TypeError('output.glb.byteLength must be a positive safe integer');
  }
  const routeStart = requireFinite(pipelineStartMs, 'pipelineStartMs');
  const routeEnd = requireFinite(pipelineEndMs, 'pipelineEndMs');
  if (routeEnd <= routeStart) throw new RangeError('pipeline interval must be positive');

  const normalizedFrames = normalizeFrames(Array.isArray(frames) ? frames : []);
  const spans = normalizeSpans(output.stageSpans);
  const stageDurationsMs = {};
  const byStage = {};

  for (const span of spans) {
    stageDurationsMs[span.name] = round(span.end - span.start);
    const overlapping = normalizedFrames
      .map(frame => ({ frame, overlapMs: intervalOverlap(frame.start, frame.end, span.start, span.end) }))
      .filter(item => item.overlapMs > 0);
    const gapSummary = summarizeGaps(overlapping.map(item => item.frame.gap));
    byStage[span.name] = Object.freeze({
      durationMs: stageDurationsMs[span.name],
      frameIntervalCount: gapSummary.frameIntervalCount,
      p95GapMs: gapSummary.p95Ms,
      p99GapMs: gapSummary.p99Ms,
      maxGapMs: gapSummary.maxMs,
      maxAttributedOverlapMs: overlapping.length
        ? round(Math.max(...overlapping.map(item => item.overlapMs)))
        : null,
      overlapMs: round(overlapping.reduce((sum, item) => sum + item.overlapMs, 0)),
      over16_7Count: gapSummary.over16_7Count,
      over33_3Count: gapSummary.over33_3Count,
      over100Count: gapSummary.over100Count,
    });
  }

  const routeFrames = normalizedFrames.filter(
    frame => intervalOverlap(frame.start, frame.end, routeStart, routeEnd) > 0,
  );

  return Object.freeze({
    name: name.trim(),
    ordinal,
    execution: normalizeArmExecution(options),
    totalMs: round(totalMs),
    pipelineIntervalMs: round(routeEnd - routeStart),
    glbBytes,
    stageDurationsMs: Object.freeze(stageDurationsMs),
    cadence: Object.freeze({
      wholeRoute: summarizeGaps(routeFrames.map(frame => frame.gap)),
      byStage: Object.freeze(byStage),
    }),
  });
}

export function summarizeCounterbalancedPair(arms, {
  controlArm,
  candidateArm,
  mechanismStage,
}) {
  if (!Array.isArray(arms) || arms.length !== 4) {
    throw new TypeError('counterbalanced pair requires exactly four episodes');
  }
  const expectedOrder = [controlArm, candidateArm, candidateArm, controlArm];
  const episodeOrder = arms.map(arm => arm?.name);
  if (episodeOrder.some((name, index) => name !== expectedOrder[index])) {
    throw new TypeError(`counterbalanced pair order must be ${expectedOrder.join(' -> ')}`);
  }
  const controls = [arms[0], arms[3]];
  const candidates = [arms[1], arms[2]];
  const collect = (episodes, field, reader) => episodes.map((episode, index) => (
    requireFinite(reader(episode), `${field}[${index}]`)
  ));
  const controlTotals = collect(controls, 'control totalMs', arm => arm.totalMs);
  const candidateTotals = collect(candidates, 'candidate totalMs', arm => arm.totalMs);
  const controlStage = collect(
    controls,
    `control ${mechanismStage}`,
    arm => arm.stageDurationsMs?.[mechanismStage],
  );
  const candidateStage = collect(
    candidates,
    `candidate ${mechanismStage}`,
    arm => arm.stageDurationsMs?.[mechanismStage],
  );
  const controlGap = collect(
    controls,
    `control ${mechanismStage} attributed gap`,
    arm => arm.cadence?.byStage?.[mechanismStage]?.maxAttributedOverlapMs,
  );
  const candidateGap = collect(
    candidates,
    `candidate ${mechanismStage} attributed gap`,
    arm => arm.cadence?.byStage?.[mechanismStage]?.maxAttributedOverlapMs,
  );

  const controlTotalMedian = median(controlTotals, 'control totalMs');
  const candidateTotalMedian = median(candidateTotals, 'candidate totalMs');
  const controlStageMedian = median(controlStage, `control ${mechanismStage}`);
  const candidateStageMedian = median(candidateStage, `candidate ${mechanismStage}`);
  const controlGapMedian = median(controlGap, `control ${mechanismStage} attributed gap`);
  const candidateGapMedian = median(candidateGap, `candidate ${mechanismStage} attributed gap`);
  if (
    controlTotalMedian <= 0
    || candidateTotalMedian <= 0
    || controlStageMedian <= 0
    || candidateStageMedian <= 0
    || controlGapMedian <= 0
    || candidateGapMedian <= 0
  ) {
    throw new RangeError('counterbalanced control and candidate medians must be positive');
  }
  const fullDelta = candidateTotalMedian - controlTotalMedian;
  const stageDelta = candidateStageMedian - controlStageMedian;
  const cadenceRatio = candidateGapMedian / controlGapMedian;

  return Object.freeze({
    design: 'A-B-B-A',
    episodeOrder: Object.freeze(episodeOrder),
    controlArm,
    candidateArm,
    mechanismStage,
    controlEpisodes: Object.freeze({
      totalMs: Object.freeze(controlTotals),
      mechanismStageMs: Object.freeze(controlStage),
      mechanismMaxAttributedGapMs: Object.freeze(controlGap),
    }),
    candidateEpisodes: Object.freeze({
      totalMs: Object.freeze(candidateTotals),
      mechanismStageMs: Object.freeze(candidateStage),
      mechanismMaxAttributedGapMs: Object.freeze(candidateGap),
    }),
    medians: Object.freeze({
      controlTotalMs: round(controlTotalMedian),
      candidateTotalMs: round(candidateTotalMedian),
      observedFullRouteDeltaMs: round(fullDelta),
      observedFullRouteRatio: round(candidateTotalMedian / controlTotalMedian, 6),
      controlMechanismStageMs: round(controlStageMedian),
      candidateMechanismStageMs: round(candidateStageMedian),
      mechanismStageDeltaMs: round(stageDelta),
      mechanismStageRatio: round(candidateStageMedian / controlStageMedian, 6),
      outsideMechanismObservedDeltaMs: round(fullDelta - stageDelta),
      controlMechanismMaxAttributedGapMs: round(controlGapMedian),
      candidateMechanismMaxAttributedGapMs: round(candidateGapMedian),
      mechanismCadenceRatio: round(cadenceRatio, 6),
    }),
    orderDrift: Object.freeze({
      controlLastMinusFirstMs: round(controlTotals[1] - controlTotals[0]),
      candidateSecondMinusFirstMs: round(candidateTotals[1] - candidateTotals[0]),
    }),
    cadenceHypothesisSatisfied: cadenceRatio < 0.5,
    causalAuthority: 'counterbalanced-mechanism-stage',
  });
}

export function buildBenchmarkFailureReport(error, failurePhase, lastTrustworthyEvidence) {
  return Object.freeze({
    ok: false,
    failurePhase,
    error: String(error),
    lastTrustworthyEvidence,
  });
}

export function compareFullRouteArms(control, candidate, { mechanismStage }) {
  if (typeof mechanismStage !== 'string' || !mechanismStage) {
    throw new TypeError('mechanismStage must be non-empty');
  }
  const controlStageMs = control?.stageDurationsMs?.[mechanismStage];
  const candidateStageMs = candidate?.stageDurationsMs?.[mechanismStage];
  requireFinite(control?.totalMs, 'control.totalMs');
  requireFinite(candidate?.totalMs, 'candidate.totalMs');
  requireFinite(controlStageMs, `control ${mechanismStage} duration`);
  requireFinite(candidateStageMs, `candidate ${mechanismStage} duration`);
  if (control.totalMs <= 0 || controlStageMs <= 0) {
    throw new RangeError('control durations must be positive');
  }

  const fullDelta = candidate.totalMs - control.totalMs;
  const mechanismDelta = candidateStageMs - controlStageMs;
  return Object.freeze({
    controlArm: control.name,
    candidateArm: candidate.name,
    mechanismStage,
    observedFullRouteDeltaMs: round(fullDelta),
    observedFullRouteRatio: round(candidate.totalMs / control.totalMs, 6),
    mechanismStageDeltaMs: round(mechanismDelta),
    mechanismStageRatio: round(candidateStageMs / controlStageMs, 6),
    outsideMechanismObservedDeltaMs: round(fullDelta - mechanismDelta),
    causalAuthority: 'mechanism-stage-only',
  });
}
