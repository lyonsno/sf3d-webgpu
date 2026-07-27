const round = (value, digits = 1) => {
  const scale = 10 ** digits;
  return Math.round((value + Number.EPSILON) * scale) / scale;
};

const requireFinite = (value, label) => {
  if (!Number.isFinite(value)) throw new TypeError(`${label} must be finite`);
  return value;
};

const intervalOverlap = (startA, endA, startB, endB) => (
  Math.max(0, Math.min(endA, endB) - Math.max(startA, startB))
);

const percentile = (sorted, fraction) => {
  if (sorted.length === 0) return null;
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)];
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
  if (totalMs < 0) throw new RangeError('output.totalMs must be non-negative');
  const glbBytes = output?.glb?.byteLength;
  if (!Number.isSafeInteger(glbBytes) || glbBytes < 0) {
    throw new TypeError('output.glb.byteLength must be a non-negative safe integer');
  }
  const routeStart = requireFinite(pipelineStartMs, 'pipelineStartMs');
  const routeEnd = requireFinite(pipelineEndMs, 'pipelineEndMs');
  if (routeEnd < routeStart) throw new RangeError('pipeline interval is invalid');

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
    glbBytes,
    stageDurationsMs: Object.freeze(stageDurationsMs),
    cadence: Object.freeze({
      wholeRoute: summarizeGaps(routeFrames.map(frame => frame.gap)),
      byStage: Object.freeze(byStage),
    }),
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
