import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const MAGIC = 0x33445346;
const ENTRY_SIZE = 160;
const HEADER_PREFIX_BYTES = 16;
const MIB = 1024 ** 2;

function execute(command, args = []) {
  return execFileSync(command, args, { encoding: 'utf8' });
}

function align4(value) {
  return Math.ceil(value / 4) * 4;
}

function requireSafeNonnegativeInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new TypeError(`${name} must be a non-negative safe integer`);
  }
  return value;
}

export function parseMemoryPressure(text) {
  const match = String(text).match(/System-wide memory free percentage:\s*(\d+)%/i);
  return match ? Number(match[1]) : null;
}

export function parseSwapUsage(text) {
  const match = String(text).match(/total\s*=\s*([\d.]+)M\s+used\s*=\s*([\d.]+)M\s+free\s*=\s*([\d.]+)M/i);
  if (!match) return null;
  return {
    totalBytes: Number(match[1]) * MIB,
    usedBytes: Number(match[2]) * MIB,
    freeBytes: Number(match[3]) * MIB,
  };
}

export function observeMacMemory({ exec = execute, statfs = fs.statfsSync, volumePath = '/' } = {}) {
  const observerErrors = [];
  let pressurePercent = null;
  let swap = null;
  let dataVolumeFreeBytes = null;
  try {
    pressurePercent = parseMemoryPressure(exec('memory_pressure', ['-Q']));
    if (pressurePercent == null) throw new Error('free percentage was absent');
  } catch (error) {
    observerErrors.push({ source: 'memory_pressure -Q', error: error.message });
  }
  try {
    swap = parseSwapUsage(exec('sysctl', ['vm.swapusage']));
    if (swap == null) throw new Error('swap values were absent');
  } catch (error) {
    observerErrors.push({ source: 'sysctl vm.swapusage', error: error.message });
  }
  try {
    const stats = statfs(volumePath);
    dataVolumeFreeBytes = Number(stats.bavail) * Number(stats.bsize);
    requireSafeNonnegativeInteger(dataVolumeFreeBytes, 'dataVolumeFreeBytes');
  } catch (error) {
    observerErrors.push({ source: `statfs ${volumePath}`, error: error.message });
  }
  return {
    schema: 'sf3d.mac-memory-observation.v0',
    source: 'live-macos',
    platform: process.platform,
    hostname: os.hostname(),
    hostTotalBytes: os.totalmem(),
    hostFreeBytes: os.freemem(),
    hostMemoryPressureFreePercent: pressurePercent,
    hostSwapTotalBytes: swap?.totalBytes ?? null,
    hostSwapUsedBytes: swap?.usedBytes ?? null,
    hostSwapFreeBytes: swap?.freeBytes ?? null,
    dataVolumeFreeBytes,
    volumePath: path.resolve(volumePath),
    observedAt: new Date().toISOString(),
    observerErrors,
  };
}

export function readMemoryObservation(observationPath) {
  const resolved = path.resolve(observationPath);
  const observation = JSON.parse(fs.readFileSync(resolved, 'utf8'));
  return { ...observation, replayPath: resolved };
}

export function inspectWeightFile(weightPath) {
  const resolved = fs.realpathSync(weightPath);
  const stat = fs.statSync(resolved);
  requireSafeNonnegativeInteger(stat.size, 'weight artifact size');
  if (stat.size < HEADER_PREFIX_BYTES) throw new Error('weight artifact is smaller than its 16-byte header');

  const fd = fs.openSync(resolved, 'r');
  try {
    const prefix = Buffer.alloc(HEADER_PREFIX_BYTES);
    if (fs.readSync(fd, prefix, 0, prefix.length, 0) !== prefix.length) {
      throw new Error('could not read complete weight header prefix');
    }
    const magic = prefix.readUInt32LE(0);
    const version = prefix.readUInt32LE(4);
    const tensorCount = prefix.readUInt32LE(8);
    const headerSize = prefix.readUInt32LE(12);
    if (magic !== MAGIC) throw new Error(`invalid weight magic 0x${magic.toString(16)}`);
    if (version !== 1) throw new Error(`unsupported weight version ${version}`);
    const expectedHeaderSize = HEADER_PREFIX_BYTES + tensorCount * ENTRY_SIZE;
    if (headerSize !== expectedHeaderSize) {
      throw new Error(`weight header size ${headerSize} does not match ${tensorCount} tensor entries (${expectedHeaderSize})`);
    }
    if (headerSize > stat.size) throw new Error('weight header extends beyond the artifact');

    const table = Buffer.alloc(headerSize - HEADER_PREFIX_BYTES);
    if (fs.readSync(fd, table, 0, table.length, HEADER_PREFIX_BYTES) !== table.length) {
      throw new Error('could not read complete weight tensor table');
    }

    let tensorPayloadBytes = 0;
    let fp16SourceBytes = 0;
    let fp32SourceBytes = 0;
    let expandedGpuUpperBoundBytes = 0;
    let largestPerTensorTransientBytes = 0;
    for (let index = 0; index < tensorCount; index++) {
      const offset = index * ENTRY_SIZE;
      const dtype = table.readUInt32LE(offset + 128);
      const dataOffset = table.readUInt32LE(offset + 152);
      const size = table.readUInt32LE(offset + 156);
      if (dtype !== 0 && dtype !== 1) throw new Error(`tensor ${index} has unsupported dtype ${dtype}`);
      if (dtype === 0 && size % 4 !== 0) throw new Error(`fp32 tensor ${index} has non-f32 byte size ${size}`);
      if (dtype === 1 && size % 2 !== 0) throw new Error(`fp16 tensor ${index} has odd byte size ${size}`);
      const absoluteEnd = headerSize + dataOffset + size;
      if (!Number.isSafeInteger(absoluteEnd) || absoluteEnd > stat.size) {
        throw new Error(`tensor ${index} extends beyond the weight artifact`);
      }
      const gpuBytes = align4(dtype === 0 ? size : size * 2);
      const conversionBytes = dtype === 1 ? size * 2 : 0;
      const transientBytes = size + conversionBytes;
      tensorPayloadBytes += size;
      expandedGpuUpperBoundBytes += gpuBytes;
      largestPerTensorTransientBytes = Math.max(largestPerTensorTransientBytes, transientBytes);
      if (dtype === 0) fp32SourceBytes += size;
      else fp16SourceBytes += size;
    }

    for (const [value, name] of [
      [tensorPayloadBytes, 'tensorPayloadBytes'],
      [expandedGpuUpperBoundBytes, 'expandedGpuUpperBoundBytes'],
      [largestPerTensorTransientBytes, 'largestPerTensorTransientBytes'],
    ]) requireSafeNonnegativeInteger(value, name);

    return {
      schema: 'sf3d.weight-memory-projection.v0',
      path: resolved,
      representation: 'flat-v1-mixed-fp32-fp16-expanded-to-fp32-gpu',
      weightArtifactBytes: stat.size,
      headerBytes: headerSize,
      tensorCount,
      tensorPayloadBytes,
      fp16SourceBytes,
      fp32SourceBytes,
      expandedGpuUpperBoundBytes,
      largestPerTensorTransientBytes,
      peakAdditionalBytes: stat.size + expandedGpuUpperBoundBytes + largestPerTensorTransientBytes,
      accounting: {
        sourceArtifact: 'entire streamed weight artifact remains resident until eager uploads complete',
        gpuStorage: 'all tensors conservatively counted at the loader effective FP32 GPU width',
        transient: 'largest tensor raw cross-chunk copy plus FP16-to-FP32 conversion may overlap source and GPU storage',
        excludes: ['browser and Vite baseline', 'pipeline buffers after weight load', 'later inference activations'],
      },
    };
  } finally {
    fs.closeSync(fd);
  }
}

export function loadMemoryAdmissionPlan(planPath) {
  const resolved = path.resolve(planPath);
  const plan = JSON.parse(fs.readFileSync(resolved, 'utf8'));
  if (plan?.schema !== 'sf3d.memory-admission-plan.v0') throw new Error('unsupported memory admission plan schema');
  if (typeof plan.id !== 'string' || !plan.id.trim()) throw new Error('memory admission plan id is required');
  if (typeof plan.targetPhase !== 'string' || !plan.targetPhase.trim()) throw new Error('memory admission target phase is required');
  if (typeof plan?.projectionCoverage?.through !== 'string' || !plan.projectionCoverage.through.trim()) {
    throw new Error('memory admission projectionCoverage.through is required');
  }
  const percent = plan?.predecessor?.hostMemoryPressureFreePercentLowWater;
  if (!Number.isFinite(percent) || percent <= 0 || percent >= 100) {
    throw new Error('predecessor hostMemoryPressureFreePercentLowWater must be between 0 and 100');
  }
  if (typeof plan.predecessor.provenance !== 'string' || !plan.predecessor.provenance.trim()) {
    throw new Error('predecessor provenance is required');
  }
  return { plan, path: resolved };
}

export function evaluateMemoryAdmission({ plan, planPath, observation, projection, requiredThrough = plan.targetPhase }) {
  if (!observation || typeof observation !== 'object') throw new Error('memory observation is required');
  if (observation.platform !== 'darwin') throw new Error(`memory admission requires a macOS observation, got ${observation.platform ?? 'unknown'}`);
  const total = requireSafeNonnegativeInteger(observation.hostTotalBytes, 'hostTotalBytes');
  const pressurePercent = observation.hostMemoryPressureFreePercent;
  if (!Number.isFinite(pressurePercent) || pressurePercent < 0 || pressurePercent > 100) {
    throw new Error('hostMemoryPressureFreePercent is missing or invalid; refusing without an authority-bearing pressure observation');
  }
  const minTotal = plan.appliesTo?.minHostTotalBytes;
  const maxTotal = plan.appliesTo?.maxHostTotalBytes;
  if (minTotal != null && total < minTotal) throw new Error(`plan ${plan.id} does not apply below ${minTotal} host bytes`);
  if (maxTotal != null && total > maxTotal) throw new Error(`plan ${plan.id} does not apply above ${maxTotal} host bytes`);

  const pressureAvailableBytes = Math.floor(total * pressurePercent / 100);
  const reservePercent = plan.predecessor.hostMemoryPressureFreePercentLowWater;
  const survivalReserveBytes = Math.ceil(total * reservePercent / 100);
  const projectedPressureAvailableBytes = pressureAvailableBytes - projection.peakAdditionalBytes;
  const reasons = [];
  if (plan.projectionCoverage.through !== requiredThrough) {
    reasons.push(`projection covers through ${plan.projectionCoverage.through}, not required ${requiredThrough}`);
  }
  if (pressureAvailableBytes < survivalReserveBytes) {
    reasons.push(`current pressure-available bytes ${pressureAvailableBytes} are already below the measured predecessor reserve ${survivalReserveBytes}`);
  }
  if (projectedPressureAvailableBytes < survivalReserveBytes) {
    reasons.push(`projected pressure-available bytes ${projectedPressureAvailableBytes} would fall below the measured predecessor reserve ${survivalReserveBytes}`);
  }

  return {
    schema: 'sf3d.memory-admission-result.v0',
    verdict: reasons.length ? 'refused' : 'admitted',
    targetPhase: plan.targetPhase,
    requiredThrough,
    effective: {
      planId: plan.id,
      planPath: planPath ? path.resolve(planPath) : null,
      observationSource: observation.source ?? 'unknown',
      observationReplayPath: observation.replayPath ?? null,
      reserveSource: 'last-successful-predecessor-pressure-low-water',
    },
    observation,
    projection,
    decision: {
      pressureAvailableBytes,
      survivalReservePercent: reservePercent,
      survivalReserveBytes,
      projectedPressureAvailableBytes,
      marginAboveReserveBytes: projectedPressureAvailableBytes - survivalReserveBytes,
      reasons,
    },
    refusalContinuation: plan.refusalContinuation ?? null,
  };
}

export function defaultPlanPathForObservation(observation, repoRoot) {
  const total = observation?.hostTotalBytes;
  if (observation?.platform !== 'darwin' || !Number.isSafeInteger(total)) return null;
  if (total < 15 * 1024 ** 3 || total > 18 * 1024 ** 3) return null;
  return path.join(repoRoot, 'tools/memory-admission-plans/m2-pro-16gib-observed-v0.json');
}

export function runMemoryAdmission({
  repoRoot,
  weightPath,
  planPath = null,
  observationPath = null,
  requiredThrough = 'weight-and-model-load',
  observe = observeMacMemory,
}) {
  const observation = observationPath
    ? readMemoryObservation(observationPath)
    : observe({ volumePath: path.dirname(path.resolve(weightPath)) });
  const effectivePlanPath = planPath || defaultPlanPathForObservation(observation, repoRoot);
  if (!effectivePlanPath) {
    return {
      schema: 'sf3d.memory-admission-result.v0',
      verdict: 'not-required',
      targetPhase: 'weight-and-model-load',
      requiredThrough,
      effective: {
        planId: null,
        planPath: null,
        observationSource: observation.source ?? 'unknown',
        observationReplayPath: observation.replayPath ?? null,
      },
      observation,
      projection: null,
      decision: { reasons: ['no matching automatic host-survival plan and none was requested'] },
      refusalContinuation: null,
    };
  }
  const loaded = loadMemoryAdmissionPlan(effectivePlanPath);
  const projection = inspectWeightFile(weightPath);
  return evaluateMemoryAdmission({ plan: loaded.plan, planPath: loaded.path, observation, projection, requiredThrough });
}
