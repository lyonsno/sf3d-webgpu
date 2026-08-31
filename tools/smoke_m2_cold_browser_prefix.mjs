#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import puppeteer from 'puppeteer-core';
import { writeJsonReportDurable } from './parent_phase_journal.mjs';

const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PREFIX_ORDER = Object.freeze([
  'weights', 'model', 'pipelines', 'arena', 'worker', 'arm-entry',
]);
const VALUE_OPTIONS = new Set([
  '--prefix', '--expected-revision', '--report', '--image', '--batch',
]);

function parseOptions(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (!VALUE_OPTIONS.has(flag)) throw new Error(`unknown option ${flag}`);
    if (values.has(flag)) throw new Error(`${flag} may be provided only once`);
    if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value`);
    values.set(flag, value);
  }
  const prefix = values.get('--prefix');
  if (!PREFIX_ORDER.includes(prefix)) throw new Error(`unknown prefix: ${prefix ?? 'missing'}`);
  const revision = values.get('--expected-revision');
  if (!revision || !/^[a-f0-9]{40}$/i.test(revision)) {
    throw new Error('--expected-revision must be an exact 40-character commit');
  }
  const report = values.get('--report');
  if (!report) throw new Error('--report is required');
  const batchText = values.get('--batch') ?? '4096';
  const batch = Number(batchText);
  if (!Number.isSafeInteger(batch) || batch <= 0 || String(batch) !== batchText) {
    throw new Error('--batch must be a positive safe integer');
  }
  return {
    prefix,
    expectedRevision: revision.toLowerCase(),
    reportPath: path.resolve(report),
    image: path.resolve(values.get('--image') ?? path.join(REPO, 'public/demo_chair.png')),
    batch,
  };
}

const options = parseOptions(process.argv.slice(2));
const parentCheckpointFd = process.env.SF3D_PARENT_CHECKPOINT_FD === '3' ? 3 : null;
const checkpoints = [];
function emitParentCheckpoint(event, details = {}) {
  const checkpoint = { event, ...details };
  checkpoints.push(checkpoint);
  if (parentCheckpointFd !== null) {
    fs.writeSync(parentCheckpointFd, `${JSON.stringify(checkpoint)}\n`);
  }
}
function durableBoundary(boundary, details = {}) {
  emitParentCheckpoint('phase-after', {
    phase: boundary.replace(/-(started|completed|published|entered)$/, ''),
    boundary,
    trustworthy: true,
    ...details,
  });
}

function sha256File(filePath) {
  return execFileSync('shasum', ['-a', '256', filePath], { encoding: 'utf8' })
    .trim().split(/\s+/)[0];
}
function allocatePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}
function waitForVite(vite) {
  return new Promise((resolve, reject) => {
    vite.stdout.on('data', data => {
      if (/Local:|ready/.test(data.toString())) resolve();
    });
    vite.on('error', reject);
    vite.on('exit', code => {
      if (code && code !== 0) reject(new Error(`vite exited ${code}`));
    });
  });
}

let browser = null;
let vite = null;
let source = null;
let prefixResult = null;
let failurePhase = 'preflight';

try {
  const revision = execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: REPO, encoding: 'utf8',
  }).trim();
  const dirty = execFileSync('git', ['status', '--short'], {
    cwd: REPO, encoding: 'utf8',
  }).trim();
  if (revision !== options.expectedRevision) {
    throw new Error(`effective revision ${revision} does not match requested ${options.expectedRevision}`);
  }
  if (dirty) throw new Error(`source worktree is dirty: ${dirty.split('\n').join(', ')}`);
  const weightsPath = fs.realpathSync(path.join(REPO, 'public/weights.bin'));
  const imagePath = fs.realpathSync(options.image);
  source = {
    routeId: 'sf3d.image-to-mesh.webgpu-local.v0',
    revision,
    clean: true,
    prefix: options.prefix,
    batch: options.batch,
    inputPath: imagePath,
    inputBytes: fs.statSync(imagePath).size,
    inputSha: sha256File(imagePath),
    weightsPath,
    weightsBytes: fs.statSync(weightsPath).size,
    weightsSha: sha256File(weightsPath),
    browserPath: fs.realpathSync(CHROME_PATH),
    host: {
      hostname: os.hostname(),
      platform: os.platform(),
      release: os.release(),
      architecture: os.arch(),
      totalMemoryBytes: os.totalmem(),
      logicalCpuCount: os.cpus().length,
      cpuModel: os.cpus()[0]?.model ?? null,
    },
  };
  emitParentCheckpoint('phase-after', {
    phase: 'source-identity',
    boundary: 'source-and-prefix-identity-verified',
    trustworthy: true,
    source,
  });

  const port = await allocatePort();
  failurePhase = 'vite-start';
  emitParentCheckpoint('phase-before', { phase: 'vite-start', port });
  vite = spawn('npx', [
    'vite', '--host', '127.0.0.1', '--port', String(port), '--strictPort',
  ], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  await waitForVite(vite);
  durableBoundary('vite-listening', { port });

  failurePhase = 'browser-launch';
  emitParentCheckpoint('phase-before', { phase: 'browser-launch' });
  browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: false,
    protocolTimeout: 0,
    args: [
      '--enable-unsafe-webgpu',
      '--use-angle=metal',
      '--disable-background-timer-throttling',
      '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding',
    ],
  });
  durableBoundary('browser-process-started', { browserPid: browser.process()?.pid ?? null });
  const page = await browser.newPage();
  await page.exposeFunction('__sf3dParentCheckpoint', checkpoint => {
    emitParentCheckpoint(checkpoint.event ?? 'phase-after', checkpoint);
  });
  await page.bringToFront();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));

  failurePhase = 'route-navigation';
  await page.goto(`http://127.0.0.1:${port}/webgpu_probe.html`, {
    waitUntil: 'domcontentloaded',
    timeout: 0,
  });
  durableBoundary('route-document-loaded', { route: 'webgpu_probe.html' });
  const backend = await page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    const info = adapter?.info;
    return {
      webgpuAvailable: Boolean(navigator.gpu),
      adapterAvailable: Boolean(adapter),
      vendor: info?.vendor ?? null,
      architecture: info?.architecture ?? null,
      device: info?.device ?? null,
      description: info?.description ?? null,
      features: adapter ? [...adapter.features].sort() : [],
    };
  });
  if (!backend.adapterAvailable) throw new Error('cold-browser adapter probe failed');
  source.backend = backend;
  durableBoundary('pre-route-adapter-verified', { backend });

  const targetIndex = PREFIX_ORDER.indexOf(options.prefix);
  if (targetIndex === 0) {
    failurePhase = 'weights-fetch';
    prefixResult = await page.evaluate(async ({ expectedBytes, weightsSha }) => {
      await globalThis.__sf3dParentCheckpoint({
        event: 'phase-after', phase: 'weights-fetch',
        boundary: 'weights-fetch-started', trustworthy: true,
        url: '/weights.bin', expectedBytes, sourceDigest: weightsSha,
      });
      const response = await fetch('/weights.bin');
      if (!response.ok) throw new Error(`weights fetch failed: ${response.status}`);
      const reader = response.body.getReader();
      let receivedBytes = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        receivedBytes += value.byteLength;
      }
      await globalThis.__sf3dParentCheckpoint({
        event: 'phase-after', phase: 'weights-fetch',
        boundary: 'weights-fetch-completed', trustworthy: true,
        url: '/weights.bin', receivedBytes, expectedBytes,
        declaredBytes: Number(response.headers.get('content-length')) || null,
        sourceDigest: weightsSha,
      });
      return { receivedBytes };
    }, { expectedBytes: source.weightsBytes, weightsSha: source.weightsSha });
    if (prefixResult.receivedBytes !== source.weightsBytes) {
      throw new Error(`weights bytes ${prefixResult.receivedBytes} != expected ${source.weightsBytes}`);
    }
  } else {
    failurePhase = 'model';
    prefixResult = await page.evaluate(async ({ weightsSha }) => {
      const { initGPU } = await import('/src/lib/gpu.js');
      const { loadWeights } = await import('/src/lib/weights.js');
      const gpu = await initGPU();
      globalThis.__sf3dPrefixDevice = gpu.device;
      globalThis.__sf3dPrefixWeights = await loadWeights(
        gpu.device,
        '/weights.bin',
        undefined,
        async (boundary, details) => {
          await globalThis.__sf3dParentCheckpoint({
            event: 'phase-after',
            phase: boundary.replace(/-(started|completed)$/, ''),
            boundary,
            trustworthy: true,
            sourceDigest: weightsSha,
            ...details,
          });
        },
      );
      return { modelLoaded: Boolean(globalThis.__sf3dPrefixWeights) };
    }, { weightsSha: source.weightsSha });
    if (!prefixResult.modelLoaded) throw new Error('model prefix returned no weights object');
  }

  if (targetIndex >= PREFIX_ORDER.indexOf('pipelines')) {
    failurePhase = 'pipeline-construction';
    prefixResult = await page.evaluate(async () => {
      await globalThis.__sf3dParentCheckpoint({
        event: 'phase-after', phase: 'pipeline-construction',
        boundary: 'pipeline-construction-started', trustworthy: true,
      });
      const { initPipelines } = await import('/src/lib/inference.js');
      globalThis.__sf3dPrefixPipelines = initPipelines(globalThis.__sf3dPrefixDevice);
      const pipelineCount = Object.keys(globalThis.__sf3dPrefixPipelines).length;
      await globalThis.__sf3dParentCheckpoint({
        event: 'phase-after', phase: 'pipeline-construction',
        boundary: 'pipeline-construction-completed', trustworthy: true,
        pipelineCount,
      });
      return { pipelineCount };
    });
  }

  if (targetIndex >= PREFIX_ORDER.indexOf('arena')) {
    failurePhase = 'arena-allocation';
    prefixResult = await page.evaluate(async (batch) => {
      const { createDecoderScratchArena, decoderArenaCapacityBytes } = await import('/src/lib/decoder_scratch_arena.js');
      const requestedBytes = decoderArenaCapacityBytes(batch);
      await globalThis.__sf3dParentCheckpoint({
        event: 'phase-after', phase: 'arena-allocation',
        boundary: 'arena-allocation-started', trustworthy: true,
        requestedBytes, batch,
      });
      globalThis.__sf3dPrefixArena = createDecoderScratchArena(
        globalThis.__sf3dPrefixDevice,
        { maxBatch: batch },
      );
      const snapshot = globalThis.__sf3dPrefixArena.snapshot();
      await globalThis.__sf3dParentCheckpoint({
        event: 'phase-after', phase: 'arena-allocation',
        boundary: 'arena-allocation-completed', trustworthy: true,
        requestedBytes, effectiveBytes: snapshot.totalBytes,
        slotCount: snapshot.slotCount, batch,
      });
      return snapshot;
    }, options.batch);
  }

  if (targetIndex >= PREFIX_ORDER.indexOf('worker')) {
    failurePhase = 'worker-creation';
    prefixResult = await page.evaluate(async () => {
      const workerUrl = new URL('/src/lib/materialize_worker.js', location.origin).href;
      await globalThis.__sf3dParentCheckpoint({
        event: 'phase-after', phase: 'worker-creation',
        boundary: 'worker-creation-started', trustworthy: true,
        workerUrl, workerName: 'sf3d-materialize',
      });
      globalThis.__sf3dPrefixWorker = new Worker(workerUrl, {
        type: 'module', name: 'sf3d-materialize',
      });
      await globalThis.__sf3dParentCheckpoint({
        event: 'phase-after', phase: 'worker-creation',
        boundary: 'worker-creation-completed', trustworthy: true,
        workerUrl, workerName: 'sf3d-materialize',
      });
      return { workerUrl, workerName: 'sf3d-materialize' };
    });
  }

  if (targetIndex >= PREFIX_ORDER.indexOf('arm-entry')) {
    failurePhase = 'arm-entry';
    const imageB64 = fs.readFileSync(source.inputPath).toString('base64');
    prefixResult = await page.evaluate(async ({ batch, imageB64 }) => {
      const image = await new Promise((resolve, reject) => {
        const candidate = new Image();
        candidate.onload = () => resolve(candidate);
        candidate.onerror = reject;
        candidate.src = `data:image/png;base64,${imageB64}`;
      });
      await globalThis.__sf3dParentCheckpoint({
        event: 'phase-after', phase: 'arm-entry',
        boundary: 'arm-start-published', trustworthy: true,
        arm: 'arena-plus-worker', batch,
      });
      const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
      await globalThis.__sf3dParentCheckpoint({
        event: 'phase-after', phase: 'arm-entry',
        boundary: 'arm-body-entered', trustworthy: true,
        arm: 'arena-plus-worker', batch,
      });
      const output = await runFullPipelineToGlb(
        globalThis.__sf3dPrefixDevice,
        globalThis.__sf3dPrefixPipelines,
        globalThis.__sf3dPrefixWeights,
        image,
        {
          cooperativeBake: true,
          bakeSchedulingMode: 'cooperative',
          bakeBatchTexels: batch,
          decoderArena: true,
          materializeWorker: globalThis.__sf3dPrefixWorker,
        },
      );
      return { glbBytes: output.glb.byteLength, totalMs: output.totalMs };
    }, { batch: options.batch, imageB64 });
  }

  if (pageErrors.length) throw new Error(`page errors: ${pageErrors.join(' | ')}`);
  durableBoundary(`prefix:${options.prefix}:complete`, { prefix: options.prefix });
  const report = {
    schema: 'sf3d.m2-cold-browser-prefix.v0',
    ok: true,
    status: 'succeeded',
    completedPrefix: options.prefix,
    source,
    prefixResult,
    checkpoints,
  };
  writeJsonReportDurable(options.reportPath, report);
  console.log(`✓ M2 COLD-BROWSER PREFIX PASSED — ${options.prefix}; report: ${options.reportPath}`);
} catch (error) {
  emitParentCheckpoint('failure', { phase: failurePhase, message: error.message });
  writeJsonReportDurable(options.reportPath, {
    schema: 'sf3d.m2-cold-browser-prefix.v0',
    ok: false,
    status: 'failed',
    requestedPrefix: options.prefix,
    failurePhase,
    error: error.message,
    source,
    prefixResult,
    checkpoints,
  });
  console.error(`M2 COLD-BROWSER PREFIX FAILED [${failurePhase}]: ${error.message}`);
  process.exitCode = 1;
} finally {
  if (browser) await browser.close().catch(() => {});
  if (vite) vite.kill();
}
