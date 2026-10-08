#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import {
  defaultPlanPathForObservation,
  evaluateMemoryAdmission,
  inspectWeightFile,
  loadMemoryAdmissionPlan,
  parseMemoryPressure,
  parseSwapUsage,
} from './memory_admission.mjs';

const GIB = 1024 ** 3;
const repo = path.resolve(new URL('..', import.meta.url).pathname);
const stateRoot = path.join(os.homedir(), '.local/state/sf3d');
fs.mkdirSync(stateRoot, { recursive: true });
const root = fs.mkdtempSync(path.join(stateRoot, 'test-memory-admission-'));

function writeWeightFixture(filePath) {
  const headerSize = 16 + 160;
  const fd = fs.openSync(filePath, 'wx');
  try {
    const header = Buffer.alloc(headerSize);
    header.writeUInt32LE(0x33445346, 0);
    header.writeUInt32LE(1, 4);
    header.writeUInt32LE(1, 8);
    header.writeUInt32LE(headerSize, 12);
    header.write('fixture.fp16', 16, 'ascii');
    header.writeUInt32LE(1, 16 + 128);
    header.writeUInt32LE(1, 16 + 132);
    header.writeUInt32LE(512 * 1024 * 1024, 16 + 136);
    header.writeUInt32LE(0, 16 + 152);
    header.writeUInt32LE(1 * GIB, 16 + 156);
    fs.writeSync(fd, header);
    fs.ftruncateSync(fd, headerSize + 1 * GIB);
  } finally {
    fs.closeSync(fd);
  }
}

try {
  const weightsPath = path.join(root, 'weights.bin');
  const planPath = path.join(root, 'plan.json');
  const observationPath = path.join(root, 'observation.json');
  const reportPath = path.join(root, 'report.json');
  writeWeightFixture(weightsPath);
  fs.writeFileSync(planPath, JSON.stringify({
    schema: 'sf3d.memory-admission-plan.v0',
    id: 'test-m2-unsafe',
    targetPhase: 'weight-and-model-load',
    projectionCoverage: { through: 'weight-and-model-load' },
    appliesTo: { minHostTotalBytes: 15 * GIB, maxHostTotalBytes: 17 * GIB },
    predecessor: {
      phase: 'model-ready',
      hostMemoryPressureFreePercentLowWater: 24,
      provenance: 'deterministic contract fixture',
    },
    refusalContinuation: {
      kind: 'smaller-rung-or-packed-storage',
      summary: 'Run setup-only or select an observed packed-storage revision.',
    },
  }, null, 2));
  fs.writeFileSync(observationPath, JSON.stringify({
    schema: 'sf3d.mac-memory-observation.v0',
    source: 'replay-fixture',
    platform: 'darwin',
    hostTotalBytes: 16 * GIB,
    hostFreeBytes: 2 * GIB,
    hostMemoryPressureFreePercent: 48,
    hostSwapTotalBytes: 5 * GIB,
    hostSwapUsedBytes: 3.5 * GIB,
    hostSwapFreeBytes: 1.5 * GIB,
    dataVolumeFreeBytes: 30 * GIB,
    observedAt: '2026-10-08T00:00:00.000Z',
  }, null, 2));

  const run = spawnSync(process.execPath, [
    'tools/smoke_product_route.mjs',
    '--allow-dirty',
    '--image', 'public/demo_chair.png',
    '--report', reportPath,
    '--weights', weightsPath,
    '--memory-admission-plan', planPath,
    '--memory-observation', observationPath,
  ], {
    cwd: repo,
    encoding: 'utf8',
    env: { ...process.env, SF3D_WITNESS_INJECT_FAILURE: 'vite-start' },
  });

  assert.notEqual(run.status, 0, 'unsafe projection must refuse the witness');
  const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
  assert.equal(report.failurePhase, 'memory-admission');
  assert.equal(report.partial.memoryAdmission.verdict, 'refused');
  assert.equal(report.partial.memoryAdmission.effective.planId, 'test-m2-unsafe');
  assert.equal(report.partial.memoryAdmission.effective.observationSource, 'replay-fixture');
  assert.equal(report.partial.memoryAdmission.projection.weightArtifactBytes, 1 * GIB + 176);
  assert.equal(report.partial.memoryAdmission.projection.expandedGpuUpperBoundBytes, 2 * GIB);
  assert.equal(report.partial.memoryAdmission.projection.largestPerTensorTransientBytes, 3 * GIB);
  assert.equal(report.partial.memoryAdmission.projection.peakAdditionalBytes, 6 * GIB + 176);
  assert.match(report.error.message, /refused.*weight-and-model-load/i);
  assert.match(report.partial.memoryAdmission.refusalContinuation.summary, /setup-only|packed-storage/i);
  assert.doesNotMatch(run.stderr, /injected failure at vite-start/, 'refusal must occur before Vite launch');

  const projection = inspectWeightFile(weightsPath);
  const { plan } = loadMemoryAdmissionPlan(planPath);
  const safe = evaluateMemoryAdmission({
    plan: { ...plan, futureAdditiveField: { mustRemainCompatible: true } },
    planPath,
    observation: {
      ...JSON.parse(fs.readFileSync(observationPath, 'utf8')),
      hostMemoryPressureFreePercent: 90,
    },
    projection,
    requiredThrough: 'weight-and-model-load',
  });
  assert.equal(safe.verdict, 'admitted');
  assert.ok(safe.decision.marginAboveReserveBytes > 0);
  assert.throws(() => evaluateMemoryAdmission({
    plan,
    planPath,
    observation: {
      ...JSON.parse(fs.readFileSync(observationPath, 'utf8')),
      hostMemoryPressureFreePercent: null,
    },
    projection,
    requiredThrough: 'weight-and-model-load',
  }), /refusing without.*pressure observation/i);
  assert.throws(() => evaluateMemoryAdmission({
    plan,
    planPath,
    observation: {
      ...JSON.parse(fs.readFileSync(observationPath, 'utf8')),
      hostTotalBytes: 32 * GIB,
    },
    projection,
    requiredThrough: 'weight-and-model-load',
  }), /does not apply above/i);
  const incomplete = evaluateMemoryAdmission({
    plan,
    planPath,
    observation: {
      ...JSON.parse(fs.readFileSync(observationPath, 'utf8')),
      hostMemoryPressureFreePercent: 90,
    },
    projection,
    requiredThrough: 'product-route-terminal',
  });
  assert.equal(incomplete.verdict, 'refused');
  assert.match(incomplete.decision.reasons.join('; '), /covers through weight-and-model-load, not required product-route-terminal/);
  assert.match(
    defaultPlanPathForObservation(JSON.parse(fs.readFileSync(observationPath, 'utf8')), repo),
    /m2-pro-16gib-observed-v0\.json$/,
  );
  assert.equal(parseMemoryPressure('System-wide memory free percentage: 37%'), 37);
  assert.deepEqual(parseSwapUsage('vm.swapusage: total = 5120.00M used = 3704.75M free = 1415.25M'), {
    totalBytes: 5120 * 1024 ** 2,
    usedBytes: 3704.75 * 1024 ** 2,
    freeBytes: 1415.25 * 1024 ** 2,
  });

  const missingPressurePath = path.join(root, 'missing-pressure.json');
  fs.writeFileSync(missingPressurePath, JSON.stringify({
    ...JSON.parse(fs.readFileSync(observationPath, 'utf8')),
    hostMemoryPressureFreePercent: null,
  }));
  const unobservableReportPath = path.join(root, 'unobservable-report.json');
  const unobservable = spawnSync(process.execPath, [
    'tools/smoke_product_route.mjs',
    '--allow-dirty',
    '--image', 'public/demo_chair.png',
    '--report', unobservableReportPath,
    '--weights', weightsPath,
    '--memory-admission-plan', planPath,
    '--memory-observation', missingPressurePath,
  ], {
    cwd: repo,
    encoding: 'utf8',
    env: { ...process.env, SF3D_WITNESS_INJECT_FAILURE: 'vite-start' },
  });
  assert.notEqual(unobservable.status, 0);
  const unobservableReport = JSON.parse(fs.readFileSync(unobservableReportPath, 'utf8'));
  assert.equal(unobservableReport.failurePhase, 'memory-admission');
  assert.equal(unobservableReport.partial.memoryAdmission.verdict, 'unobservable');
  assert.match(unobservableReport.error.message, /pressure observation/i);
  assert.doesNotMatch(unobservable.stderr, /injected failure at vite-start/);

  console.log('memory admission contract passed');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
