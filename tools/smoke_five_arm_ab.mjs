#!/usr/bin/env node
/**
 * SF3D exact-source SIX-ARM texture-bake A/B (Cranial arena assay contract #7 /
 * Wake required consumer evidence).
 *
 * One loaded model, six arms over the exact same route:
 *   1 monolithic         — legacy single decode dispatch + readback
 *   2 current-cooperative — cooperative batches, per-range transient scratch
 *   3 arena-only          — cooperative + decoder scratch arena (no worker mat)
 *   4 worker-only         — cooperative + worker materialization (no arena)
 *   5 arena-plus-worker   — cooperative + arena + worker materialization
 *
 * Records per arm: full-route wall, every stage duration, whole-route and
 * overlap-attributed stage cadence, effective execution identity, output GLB
 * SHA-256 (must be identical across all six), queue intervals, scratch
 * allocation count/bytes, and worker transfer time. Also carries source and
 * timing-scope identity (commit, input, weights, host, browser/adapter probe,
 * one resident model, and setup excluded from each arm).
 *
 * Usage: node tools/smoke_five_arm_ab.mjs --expected-revision SHA
 *   [--profile six-arm|paired] [--image P] [--batch N]
 */
import puppeteer from 'puppeteer-core';
import path from 'node:path';
import fs from 'node:fs';
import net from 'node:net';
import crypto from 'node:crypto';
import os from 'node:os';
import { spawn, execFileSync } from 'node:child_process';
import {
  buildBenchmarkFailureReport,
  compareFullRouteArms,
  parsePositiveSafeIntegerOption,
  summarizeCounterbalancedPair,
  validateGlbPayload,
} from './full_route_benchmark_contract.mjs';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const DEFAULT_IMAGE = path.join(
  process.env.HOME,
  '.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png',
);
const readOption = (flag) => {
  const indexes = process.argv
    .map((value, index) => (value === flag ? index : -1))
    .filter(index => index >= 0);
  const index = indexes[0] ?? -1;
  const value = index >= 0 ? process.argv[index + 1] : undefined;
  return Object.freeze({
    flag,
    provided: index >= 0,
    duplicate: indexes.length > 1,
    value: value && !value.startsWith('--') ? value : undefined,
  });
};
const requestedOptions = Object.freeze({
  image: readOption('--image'),
  batch: readOption('--batch'),
  report: readOption('--report'),
  expectedRevision: readOption('--expected-revision'),
  expectedOutputSha: readOption('--expected-output-sha'),
  profile: readOption('--profile'),
});
const REPORT_PATH = path.resolve(
  requestedOptions.report.value || '/tmp/sf3d-five-arm-ab.json',
);
let IMAGE;
let BATCH;
let EXPECTED_REVISION;
let EXPECTED_OUTPUT_SHA;
let PROFILE;
const ALLOW_DIRTY = process.argv.includes('--allow-dirty-source');
const ARM_ORDERS = Object.freeze({
  'six-arm': Object.freeze([
    'monolithic',
    'current-cooperative',
    'arena-only',
    'worker-only',
    'arena-plus-worker',
    'arena-disabled',
  ]),
  paired: Object.freeze([
    'monolithic',
    'arena-plus-worker',
    'arena-plus-worker',
    'monolithic',
  ]),
});

const allocatePort = () => new Promise((res, rej) => { const s = net.createServer(); s.unref(); s.on('error', rej); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); }); });
const sha256File = (p) => execFileSync('shasum', ['-a', '256', p], { encoding: 'utf8' }).trim().split(/\s+/)[0];
const sha256Text = (value) => crypto.createHash('sha256').update(value).digest('hex');
const sha256Directory = (root) => {
  const files = [];
  const visit = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(absolute);
      else if (entry.isFile()) files.push(absolute);
    }
  };
  visit(root);
  const hash = crypto.createHash('sha256');
  for (const file of files.sort()) {
    hash.update(path.relative(root, file));
    hash.update('\0');
    hash.update(fs.readFileSync(file));
    hash.update('\0');
  }
  return hash.digest('hex');
};
const hashGlbArm = (arm) => {
  const bytes = Buffer.from(arm.glbB64, 'base64');
  validateGlbPayload(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  return {
    ...arm,
    glbSha: crypto.createHash('sha256').update(bytes).digest('hex'),
    glbB64: undefined,
  };
};
const procs = [];
const cleanup = () => { for (const p of procs) { try { p.kill(); } catch {} } };
let lastEvidence = {};
const fail = (m, phase = 'unknown') => {
  const failure = buildBenchmarkFailureReport(m, phase, lastEvidence);
  fs.writeFileSync(REPORT_PATH, JSON.stringify(failure, null, 2));
  console.error(`\n✗ SMOKE FAILED [${phase}]: ${m}`);
  cleanup();
  process.exit(1);
};

(async () => {
  for (const option of Object.values(requestedOptions)) {
    if (option.duplicate) fail(`${option.flag} may be provided only once`, 'requested-config');
    if (option.provided && option.value === undefined) {
      fail(`${option.flag} requires a value`, 'requested-config');
    }
  }
  let batchIdentity;
  try {
    batchIdentity = parsePositiveSafeIntegerOption(
      requestedOptions.batch.provided ? requestedOptions.batch.value : undefined,
      { label: '--batch', defaultValue: 4096 },
    );
  } catch (error) {
    fail(error.message, 'requested-config');
  }
  BATCH = batchIdentity.effective;
  IMAGE = path.resolve(requestedOptions.image.value || DEFAULT_IMAGE);
  EXPECTED_REVISION = requestedOptions.expectedRevision.value
    || process.env.SF3D_EXPECTED_REVISION
    || '';
  EXPECTED_OUTPUT_SHA = requestedOptions.expectedOutputSha.value || '';
  PROFILE = requestedOptions.profile.value || 'six-arm';
  if (!ARM_ORDERS[PROFILE]) fail(`unknown profile: ${PROFILE}`, 'profile');
  if (EXPECTED_OUTPUT_SHA && !/^[a-f0-9]{64}$/i.test(EXPECTED_OUTPUT_SHA)) {
    fail('--expected-output-sha must be a 64-character hexadecimal SHA-256', 'requested-config');
  }
  if (PROFILE === 'paired' && !EXPECTED_OUTPUT_SHA) {
    fail('--expected-output-sha is required for the paired profile', 'requested-config');
  }
  // Source identity gate.
  const effectiveRevision = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: REPO, encoding: 'utf8' }).trim();
  const dirty = execFileSync('git', ['status', '--short'], { cwd: REPO, encoding: 'utf8' }).trim();
  if (!EXPECTED_REVISION) fail('--expected-revision required', 'source-identity');
  if (effectiveRevision !== EXPECTED_REVISION) fail(`effective ${effectiveRevision} != requested ${EXPECTED_REVISION}`, 'source-identity');
  if (dirty && (!ALLOW_DIRTY || PROFILE === 'paired')) {
    fail(`worktree dirty:\n${dirty}`, 'source-identity');
  }
  if (!fs.existsSync(IMAGE)) fail(`input missing: ${IMAGE}`, 'input');
  const effectiveInputPath = fs.realpathSync(IMAGE);
  const requestedWeightsPath = path.join(REPO, 'public', 'weights.bin');
  if (!fs.existsSync(requestedWeightsPath)) fail('weights missing', 'weights');
  const effectiveWeightsPath = fs.realpathSync(requestedWeightsPath);
  const kitPackagePath = path.join(
    REPO,
    'node_modules',
    '@kaminos',
    'webgpu-inference-kit',
    'package.json',
  );
  if (!fs.existsSync(kitPackagePath)) fail('installed inference-kit package missing', 'package-identity');
  const kitPackage = JSON.parse(fs.readFileSync(kitPackagePath, 'utf8'));
  const kitModule = await import('@kaminos/webgpu-inference-kit');
  const packageLock = JSON.parse(fs.readFileSync(path.join(REPO, 'package-lock.json'), 'utf8'));
  const lockedKit = packageLock.packages?.['node_modules/@kaminos/webgpu-inference-kit'];
  const effectiveKit = {
    name: kitPackage.name,
    packageVersion: kitPackage.version,
    exportedVersion: kitModule.WEBGPU_INFERENCE_KIT_VERSION,
    packageJsonPath: fs.realpathSync(kitPackagePath),
    packageJsonSha: sha256File(kitPackagePath),
    installedTreeSha: sha256Directory(path.dirname(kitPackagePath)),
    lockVersion: lockedKit?.version ?? null,
    lockResolved: lockedKit?.resolved ?? null,
    lockIntegrity: lockedKit?.integrity ?? null,
  };
  if (
    effectiveKit.name !== '@kaminos/webgpu-inference-kit'
    || effectiveKit.packageVersion !== effectiveKit.exportedVersion
    || effectiveKit.packageVersion !== effectiveKit.lockVersion
  ) {
    fail(`inference-kit identity mismatch: ${JSON.stringify(effectiveKit)}`, 'package-identity');
  }
  const dirtyDiff = dirty
    ? execFileSync('git', ['diff', '--binary', 'HEAD'], { cwd: REPO, encoding: 'utf8' })
    : '';
  const source = {
    requested: {
      revision: EXPECTED_REVISION,
      inputPath: requestedOptions.image.value || DEFAULT_IMAGE,
      weightsPath: requestedWeightsPath,
      requestedBatch: batchIdentity.requested,
      profile: requestedOptions.profile.value || 'six-arm',
      expectedOutputSha: EXPECTED_OUTPUT_SHA || null,
      allowDirtySource: ALLOW_DIRTY,
    },
    effective: {
      worktreePath: fs.realpathSync(REPO),
      revision: effectiveRevision,
      inputPath: effectiveInputPath,
      inputBytes: fs.statSync(effectiveInputPath).size,
      weightsPath: effectiveWeightsPath,
      weightsBytes: fs.statSync(effectiveWeightsPath).size,
      batch: BATCH,
      profile: PROFILE,
      browserPath: fs.realpathSync(CHROME_PATH),
      kit: effectiveKit,
      dirtyStatus: dirty || null,
      dirtyDiffSha: dirty ? sha256Text(dirtyDiff) : null,
    },
    revision: effectiveRevision,
    clean: !dirty,
    inputSha: sha256File(effectiveInputPath),
    weightsSha: sha256File(effectiveWeightsPath),
    batch: BATCH,
    profile: PROFILE,
    host: {
      hostname: os.hostname(),
      platform: os.platform(),
      release: os.release(),
      architecture: os.arch(),
      logicalCpuCount: os.cpus().length,
      cpuModel: os.cpus()[0]?.model || 'unknown',
      totalMemoryBytes: os.totalmem(),
    },
  };
  lastEvidence = { source };

  const port = await allocatePort();
  const vite = spawn('npx', ['vite', '--port', String(port), '--strictPort'], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(vite);
  await new Promise((res, rej) => { const to = setTimeout(() => rej(new Error('vite timeout')), 40000); vite.stdout.on('data', d => { if (/Local:|ready/.test(d.toString())) { clearTimeout(to); res(); } }); vite.on('error', e => { clearTimeout(to); rej(e); }); }).catch(e => fail(e.message, 'vite'));

  const browserArgs = [
    '--enable-unsafe-webgpu',
    '--use-angle=metal',
    '--disable-background-timer-throttling',
    '--disable-backgrounding-occluded-windows',
    '--disable-renderer-backgrounding',
  ];
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: false,
    protocolTimeout: 1200000,
    args: browserArgs,
  });
  const page = await browser.newPage();
  await page.bringToFront();
  const pageErrors = []; page.on('pageerror', e => pageErrors.push(e.message));
  const setupStartedAt = Date.now();
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded', timeout: 30000 });
  const start = Date.now();
  let readyStatus = '';
  while (Date.now() - start < 180000) {
    readyStatus = await page.$eval('#status', el => el.textContent).catch(() => '');
    if (readyStatus.includes('Ready')) break;
    if (/error|failed/i.test(readyStatus)) fail(`route setup failed: ${readyStatus}`, 'route-setup');
    await new Promise(r => setTimeout(r, 500));
  }
  if (!readyStatus.includes('Ready')) fail(`route setup timed out: ${readyStatus || 'no status'}`, 'route-setup');
  source.timingScope = {
    setupWallMs: Date.now() - setupStartedAt,
    setupIncludesModelLoadAndPipelineInitialization: true,
    armTotalsExcludeSetup: true,
    allArmsShareOneResidentModel: true,
    armOrder: ARM_ORDERS[PROFILE],
    browserControlArgs: browserArgs,
  };
  source.backend = await page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    const info = adapter?.info;
    return {
      authority: 'same-browser-capability-probe',
      ua: navigator.userAgent,
      vendor: info?.vendor,
      architecture: info?.architecture,
      device: info?.device,
      description: info?.description,
      features: adapter ? [...adapter.features].sort() : [],
      limits: adapter ? {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      } : null,
    };
  });
  const browserExecutedKit = await page.evaluate(async () => {
    const witness = await import('/tools/browser_kit_identity.js');
    return witness.readBrowserKitIdentity();
  });
  source.effective.kit.browserExecutedKit = browserExecutedKit;
  if (
    browserExecutedKit.packageName !== effectiveKit.name
    || browserExecutedKit.exportedVersion !== effectiveKit.exportedVersion
  ) {
    fail(
      `browser inference-kit identity mismatch: ${JSON.stringify(browserExecutedKit)}`,
      'package-identity',
    );
  }
  lastEvidence = { source };

  const imageB64 = fs.readFileSync(IMAGE).toString('base64');
  await page.evaluate(async (b64) => { await new Promise((res, rej) => { const img = new Image(); img.onload = () => { window._img = img; res(); }; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; }); }, imageB64);
  const partialArms = [];
  await page.exposeFunction('__sf3dBenchmarkCheckpoint', rawArm => {
    const arm = hashGlbArm(rawArm);
    partialArms.push(arm);
    lastEvidence = {
      schema: 'sf3d.partial-full-route-scheduling-episodes.v1',
      ok: false,
      status: 'running',
      source,
      completedEpisodeCount: partialArms.length,
      arms: [...partialArms],
    };
    fs.writeFileSync(REPORT_PATH, JSON.stringify(lastEvidence, null, 2));
  });

  const arms = await page.evaluate(async (BATCH, PROFILE) => {
    const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
    const { buildFullRouteArmReceipt } = await import('/tools/full_route_benchmark_contract.mjs');
    const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines, img = window._img;
    const toB64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
    let matWorker = null;
    const getMatWorker = () => (matWorker ||= new Worker(new URL('/src/lib/materialize_worker.js', location.origin), { type: 'module' }));

    async function arm(name, ordinal, opts) {
      const frames = []; let on = true, last = null;
      const tick = (t) => { if (last != null) frames.push({ start: last, end: t, gap: t - last }); last = t; if (on) requestAnimationFrame(tick); };
      const visibilityStart = document.visibilityState;
      if (visibilityStart !== 'visible') throw new Error(`${name} started with page visibility ${visibilityStart}`);
      await new Promise(resolve => requestAnimationFrame(t => {
        last = t;
        requestAnimationFrame(tick);
        resolve();
      }));
      const pipelineStartMs = performance.now();
      const out = await runFullPipelineToGlb(device, pipelines, weights, img, opts);
      const pipelineEndMs = performance.now();
      await new Promise(resolve => requestAnimationFrame(resolve));
      on = false;
      await new Promise(r => setTimeout(r, 20));
      const visibilityEnd = document.visibilityState;
      if (visibilityEnd !== 'visible') throw new Error(`${name} ended with page visibility ${visibilityEnd}`);
      const route = buildFullRouteArmReceipt({
        name,
        ordinal,
        options: opts,
        output: out,
        frames,
        pipelineStartMs,
        pipelineEndMs,
      });
      const bake = out.cooperativeReports?.['texture-bake'] || null;
      const tel = bake?.textureBakeTelemetry || null;
      return {
        ...route,
        pageVisibility: {
          start: visibilityStart,
          end: visibilityEnd,
        },
        glbB64: toB64(new Uint8Array(out.glb)),
        bakeStageMs: route.stageDurationsMs['texture-bake'] ?? null,
        bakeMaxGapMs: route.cadence.byStage['texture-bake']?.maxGapMs ?? null,
        bakeMaxAttributedGapMs: route.cadence.byStage['texture-bake']?.maxAttributedOverlapMs ?? null,
        scratchAllocatedBytes: tel?.scratch?.allocatedBytes ?? null,
        scratchAllocatedCount: tel?.scratch?.allocatedCount ?? null,
        queueWaitTotalMs: tel?.queueFences ? +tel.queueFences.reduce((s, q) => s + (q.queueWaitMs || 0), 0).toFixed(1) : null,
        cpuMaterializationMs: tel?.phases?.cpuMaterializationMs ?? null,
        materializationOffloaded: tel?.phases?.materializationOffloaded ?? null,
        workerTransferMs: tel?.phases?.materializationWorkerTransferMs ?? null,
        arenaSnapshot: out.arenaSnapshot || null,
        rangeCount: bake?.boundaries?.[0]?.actualRangeCount ?? null,
      };
    }

    const base = { cooperativeDino: false };
    const coop = { ...base, cooperativeBake: true, bakeSchedulingMode: 'cooperative', bakeBatchTexels: BATCH };
    const runByName = async (name, ordinal) => {
      if (name === 'monolithic') return arm(name, ordinal, { ...base });
      if (name === 'current-cooperative') return arm(name, ordinal, { ...coop });
      if (name === 'arena-only') return arm(name, ordinal, { ...coop, decoderArena: true });
      if (name === 'worker-only') return arm(name, ordinal, { ...coop, materializeWorker: getMatWorker() });
      if (name === 'arena-plus-worker') {
        return arm(name, ordinal, { ...coop, decoderArena: true, materializeWorker: getMatWorker() });
      }
      if (name === 'arena-disabled') {
        return arm(name, ordinal, { ...coop, decoderArena: true, bakeSchedulingMode: 'disabled' });
      }
      throw new Error(`unknown arm: ${name}`);
    };
    const order = PROFILE === 'paired'
      ? ['monolithic', 'arena-plus-worker', 'arena-plus-worker', 'monolithic']
      : ['monolithic', 'current-cooperative', 'arena-only', 'worker-only', 'arena-plus-worker', 'arena-disabled'];
    const results = [];
    for (let index = 0; index < order.length; index++) {
      const result = await runByName(order[index], index + 1);
      results.push(result);
      await globalThis.__sf3dBenchmarkCheckpoint(result);
    }
    if (matWorker) matWorker.terminate();
    return results;
  }, BATCH, PROFILE);

  await browser.close().catch(() => {}); cleanup();
  if (pageErrors.length) fail(`page errors: ${pageErrors.join(' | ')}`, 'browser');

  const withSha = arms.map(hashGlbArm);
  if (
    withSha.length !== partialArms.length
    || withSha.some((arm, index) => (
      arm.ordinal !== partialArms[index]?.ordinal
      || arm.name !== partialArms[index]?.name
      || arm.glbSha !== partialArms[index]?.glbSha
    ))
  ) {
    fail('final browser episodes do not match durable per-episode checkpoints', 'episode-checkpoint');
  }
  const shas = new Set(withSha.map(a => a.glbSha));
  const identical = shas.size === 1;
  const expectedOutputMatches = !EXPECTED_OUTPUT_SHA
    || withSha.every(arm => arm.glbSha === EXPECTED_OUTPUT_SHA);
  lastEvidence = {
    schema: 'sf3d.raw-full-route-scheduling-episodes.v1',
    source,
    outputIdentical: identical,
    expectedOutputSha: EXPECTED_OUTPUT_SHA || null,
    expectedOutputMatches,
    canonicalGlbSha: withSha[0]?.glbSha ?? null,
    distinctShas: [...shas],
    arms: withSha,
  };

  let report;
  let acceptanceError = null;
  if (PROFILE === 'paired') {
    const pairedComparison = summarizeCounterbalancedPair(withSha, {
      controlArm: 'monolithic',
      candidateArm: 'arena-plus-worker',
      mechanismStage: 'texture-bake',
    });
    const candidateScratchBytes = withSha
      .filter(arm => arm.name === 'arena-plus-worker')
      .map(arm => arm.scratchAllocatedBytes);
    const candidateScratchStable = candidateScratchBytes.every(
      bytes => Number.isSafeInteger(bytes) && bytes >= 0 && bytes === candidateScratchBytes[0],
    );
    report = {
      schema: 'sf3d.counterbalanced-full-route-scheduling-comparison.v1',
      ok: identical
        && expectedOutputMatches
        && pairedComparison.cadenceHypothesisSatisfied
        && candidateScratchStable,
      source,
      outputIdentical: identical,
      expectedOutputSha: EXPECTED_OUTPUT_SHA,
      expectedOutputMatches,
      canonicalGlbSha: withSha[0].glbSha,
      distinctShas: [...shas],
      pairedComparison,
      candidateScratchBytes,
      candidateScratchStable,
      arms: withSha,
    };
    if (!identical) acceptanceError = `GLB DIFFERS across episodes: ${[...shas].map(s => s.slice(0, 10)).join(', ')}`;
    else if (!expectedOutputMatches) acceptanceError = `GLB does not match expected SHA-256 ${EXPECTED_OUTPUT_SHA}`;
    else if (!candidateScratchStable) acceptanceError = `candidate scratch identity changed: ${candidateScratchBytes.join(', ')}`;
    else if (!pairedComparison.cadenceHypothesisSatisfied) {
      acceptanceError = `counterbalanced texture-bake cadence did not collapse: ${pairedComparison.medians.candidateMechanismMaxAttributedGapMs}ms vs ${pairedComparison.medians.controlMechanismMaxAttributedGapMs}ms`;
    }
  } else {
    // Six-arm acceptance retains the mechanism decomposition. Full-route
    // performance claims belong to the counterbalanced paired profile.
    const byName = Object.fromEntries(withSha.map(a => [a.name, a]));
    const cur = byName['current-cooperative'];
    const mono = byName.monolithic;
    const best = byName['arena-plus-worker'];
    const scratchCollapsed = best.scratchAllocatedBytes != null && cur.scratchAllocatedBytes != null
      && best.scratchAllocatedBytes < cur.scratchAllocatedBytes * 0.2;
    const gapCollapsed = best.bakeMaxAttributedGapMs != null && mono.bakeMaxAttributedGapMs != null
      && best.bakeMaxAttributedGapMs < mono.bakeMaxAttributedGapMs * 0.5;
    const fullRouteComparison = compareFullRouteArms(
      mono,
      best,
      { mechanismStage: 'texture-bake' },
    );
    report = {
      schema: 'sf3d.six-arm-mechanism-comparison.v1',
      ok: identical && expectedOutputMatches && scratchCollapsed && gapCollapsed,
      source,
      outputIdentical: identical,
      expectedOutputSha: EXPECTED_OUTPUT_SHA || null,
      expectedOutputMatches,
      canonicalGlbSha: withSha[0].glbSha,
      distinctShas: [...shas],
      winning: {
        arm: 'arena-plus-worker',
        totalMs: best.totalMs,
        wholeRouteMaxGapMs: best.cadence.wholeRoute.maxMs,
        bakeMaxGapMs: best.bakeMaxGapMs,
        bakeMaxAttributedGapMs: best.bakeMaxAttributedGapMs,
        scratchAllocatedBytes: best.scratchAllocatedBytes,
        scratchCollapsedVsCurrent: scratchCollapsed,
        gapCollapsedVsMonolithic: gapCollapsed,
      },
      comparisons: {
        monolithicVsArenaPlusWorker: fullRouteComparison,
      },
      arms: withSha,
    };
    if (!identical) acceptanceError = `GLB DIFFERS across arms: ${[...shas].map(s => s.slice(0, 10)).join(', ')}`;
    else if (!expectedOutputMatches) acceptanceError = `GLB does not match expected SHA-256 ${EXPECTED_OUTPUT_SHA}`;
    else if (!scratchCollapsed) acceptanceError = `arena+worker did not collapse scratch: ${best.scratchAllocatedBytes} vs current ${cur.scratchAllocatedBytes}`;
    else if (!gapCollapsed) acceptanceError = `arena+worker did not collapse attributed gap: ${best.bakeMaxAttributedGapMs}ms vs monolithic ${mono.bakeMaxAttributedGapMs}ms`;
  }
  lastEvidence = report;
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));

  console.log(`\n=== SF3D ${PROFILE} full-route scheduling comparison ===`);
  console.log(`source ${source.revision.slice(0, 10)} clean=${source.clean} ${source.backend.vendor}/${source.backend.architecture} batch=${BATCH}`);
  console.log(`GLB identical across all episodes: ${identical} (sha ${withSha[0].glbSha.slice(0, 12)}…)`);
  console.log('episode'.padEnd(20), 'routeWall'.padStart(10), 'routeMaxGap'.padStart(12), 'bakeWall'.padStart(9), 'bakeAttrib'.padStart(11), 'scratchMB'.padStart(10), 'cpuMatMs'.padStart(9), 'xferMs'.padStart(7));
  for (const a of withSha) {
    console.log(
      `${a.ordinal}:${a.name}`.padEnd(20),
      String(a.totalMs).padStart(10),
      String(a.cadence.wholeRoute.maxMs).padStart(12),
      String(a.bakeStageMs).padStart(9),
      String(a.bakeMaxAttributedGapMs).padStart(11),
      String(a.scratchAllocatedBytes != null ? (a.scratchAllocatedBytes / 1e6).toFixed(1) : 'n/a').padStart(10),
      String(a.cpuMaterializationMs != null ? a.cpuMaterializationMs.toFixed(0) : 'n/a').padStart(9),
      String(a.workerTransferMs != null ? a.workerTransferMs.toFixed(0) : 'n/a').padStart(7),
    );
  }
  if (PROFILE === 'paired') {
    const medians = report.pairedComparison.medians;
    console.log(`\nmedian observed full-route delta: ${medians.observedFullRouteDeltaMs}ms (${medians.observedFullRouteRatio}x)`);
    console.log(`median texture-bake mechanism delta: ${medians.mechanismStageDeltaMs}ms (${medians.mechanismStageRatio}x)`);
    console.log(`median outside-texture-bake observed delta: ${medians.outsideMechanismObservedDeltaMs}ms (not scheduler-causal)`);
    console.log(`median attributed bake cadence: ${medians.controlMechanismMaxAttributedGapMs}ms -> ${medians.candidateMechanismMaxAttributedGapMs}ms (${medians.mechanismCadenceRatio}x)`);
    console.log(`order drift: control ${report.pairedComparison.orderDrift.controlLastMinusFirstMs}ms, candidate ${report.pairedComparison.orderDrift.candidateSecondMinusFirstMs}ms`);
  } else {
    const comparison = report.comparisons.monolithicVsArenaPlusWorker;
    console.log(`\nobserved full-route delta: ${comparison.observedFullRouteDeltaMs}ms (${comparison.observedFullRouteRatio}x)`);
    console.log(`texture-bake mechanism delta: ${comparison.mechanismStageDeltaMs}ms (${comparison.mechanismStageRatio}x)`);
    console.log(`outside-texture-bake observed delta: ${comparison.outsideMechanismObservedDeltaMs}ms (not scheduler-causal)`);
    console.log(`arena+worker: scratchCollapsed=${report.winning.scratchCollapsedVsCurrent} attributedGapCollapsed=${report.winning.gapCollapsedVsMonolithic}`);
  }
  if (acceptanceError) fail(acceptanceError, identical ? 'acceptance' : 'output-identity');
  console.log(`✓ SMOKE PASSED — ${PROFILE} output and scheduling contracts hold. report: ${REPORT_PATH}`);
  process.exit(0);
})().catch(e => fail(e.message, 'exception'));
