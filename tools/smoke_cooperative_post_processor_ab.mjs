#!/usr/bin/env node
/**
 * Real-browser A/B for the SF3D postprocessor plane boundary.
 *
 * Both arms execute the same complete image-to-GLB route. The control keeps the
 * postprocessor in one command buffer; the candidate submits three independent
 * plane duties through the Kaminos cooperative facade. The witness requires
 * byte-identical GLBs and reports wall/cadence only for the exact postprocessor
 * span, while preserving route/source/backend identities and writing a failure
 * report at every terminal path.
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import puppeteer from 'puppeteer-core';

const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const argValue = (flag, fallback) => {
  const index = process.argv.indexOf(flag);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
};
const EXPECTED_REVISION = argValue(
  '--expected-revision',
  process.env.SF3D_EXPECTED_REVISION || '',
);
const IMAGE = path.resolve(argValue(
  '--image',
  path.join(process.env.HOME, '.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png'),
));
const REPORT_PATH = path.resolve(argValue(
  '--report',
  '/tmp/sf3d-cooperative-post-processor-ab.json',
));
const ALLOW_DIRTY_SOURCE = process.argv.includes('--allow-dirty-source');
const WEIGHTS = path.join(REPO, 'public', 'weights.bin');
const processes = [];
let browser = null;
let lastTrustworthyEvidence = { phase: 'argument-parse' };

function git(args, options = {}) {
  return execFileSync('git', args, { cwd: REPO, encoding: 'utf8', ...options }).trim();
}

function sourceIdentity() {
  const revision = git(['rev-parse', 'HEAD']);
  const dirtyPaths = git(['status', '--short']).split('\n').filter(Boolean);
  const hash = crypto.createHash('sha256');
  hash.update(execFileSync('git', ['diff', 'HEAD', '--binary'], { cwd: REPO }));
  for (const relativePath of git(['ls-files', '--others', '--exclude-standard'])
    .split('\n').filter(Boolean).sort()) {
    hash.update(relativePath);
    const absolutePath = path.join(REPO, relativePath);
    if (fs.statSync(absolutePath).isFile()) hash.update(fs.readFileSync(absolutePath));
  }
  return {
    revision,
    requestedRevision: EXPECTED_REVISION || null,
    matchesRequestedRevision: Boolean(EXPECTED_REVISION)
      && git(['rev-parse', '--verify', `${EXPECTED_REVISION}^{commit}`]) === revision,
    clean: dirtyPaths.length === 0,
    dirtyModeExplicit: ALLOW_DIRTY_SOURCE,
    dirtyDiffSha256: dirtyPaths.length ? hash.digest('hex') : null,
    dirtyPaths,
    worktree: REPO,
  };
}

function cleanup() {
  for (const process of processes) {
    try { process.kill(); } catch {}
  }
}

function writeFailure(error, phase, details = null) {
  const report = {
    schema: 'sf3d.cooperative-post-processor-ab.v0',
    ok: false,
    failure: {
      phase,
      message: error instanceof Error ? error.message : String(error),
      details,
    },
    lastTrustworthyEvidence,
  };
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));
  return report;
}

function fail(error, phase, details = null) {
  const report = writeFailure(error, phase, details);
  console.error(`SMOKE FAILED [${phase}]: ${report.failure.message}`);
  throw error instanceof Error ? error : new Error(String(error));
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

function sha256File(file) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(file);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

try {
  const source = sourceIdentity();
  lastTrustworthyEvidence = { phase: 'source-identity', source };
  if (!EXPECTED_REVISION) fail(
    new Error('--expected-revision or SF3D_EXPECTED_REVISION is required'),
    'source-identity',
  );
  if (!source.matchesRequestedRevision) fail(
    new Error(`effective revision ${source.revision} does not match ${EXPECTED_REVISION}`),
    'source-identity',
  );
  if (!source.clean && !ALLOW_DIRTY_SOURCE) fail(
    new Error(`dirty source requires --allow-dirty-source; diff=${source.dirtyDiffSha256}`),
    'source-identity',
  );
  if (!fs.existsSync(IMAGE)) fail(new Error(`missing input image ${IMAGE}`), 'input');
  if (!fs.existsSync(WEIGHTS)) fail(new Error(`missing model weights ${WEIGHTS}`), 'weights');

  const effectiveWeights = fs.realpathSync(WEIGHTS);
  const routeIdentity = {
    source,
    input: { path: IMAGE, bytes: fs.statSync(IMAGE).size, sha256: await sha256File(IMAGE) },
    weights: {
      requestedPath: WEIGHTS,
      effectivePath: effectiveWeights,
      bytes: fs.statSync(effectiveWeights).size,
      sha256: await sha256File(effectiveWeights),
    },
    requestedArms: [
      { name: 'monolithic-control', cooperativePostProcessor: false },
      {
        name: 'three-plane-cooperative',
        cooperativePostProcessor: true,
        postProcessorSchedulingMode: 'cooperative',
      },
    ],
  };
  lastTrustworthyEvidence = { phase: 'route-identity', routeIdentity };

  const port = await allocatePort();
  const vite = spawn(
    'npx',
    ['vite', '--host', '127.0.0.1', '--port', String(port), '--strictPort'],
    {
    cwd: REPO,
    stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  processes.push(vite);
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Vite startup timed out')), 40000);
    vite.stdout.on('data', (data) => {
      if (/Local:|ready/.test(data.toString())) {
        clearTimeout(timeout);
        resolve();
      }
    });
    vite.on('error', reject);
    vite.on('exit', (code) => {
      if (code && code !== 0) reject(new Error(`Vite exited ${code}`));
    });
  }).catch((error) => fail(error, 'vite'));

  browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: false,
    protocolTimeout: 900000,
    args: ['--enable-unsafe-webgpu', '--use-angle=metal'],
  });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto(`http://127.0.0.1:${port}/`, {
    waitUntil: 'domcontentloaded',
    timeout: 30000,
  });

  let status = '';
  const setupStartedAt = Date.now();
  while (Date.now() - setupStartedAt < 180000) {
    status = await page.$eval('#status', (element) => element.textContent).catch(() => '');
    if (status.includes('Ready')) break;
    if (/error|failed/i.test(status)) break;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  if (!status.includes('Ready') || pageErrors.length) fail(
    new Error(`browser setup incomplete: status=${JSON.stringify(status)}`),
    'browser-setup',
    { pageErrors },
  );

  routeIdentity.backend = await page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    const info = adapter?.info;
    return {
      userAgent: navigator.userAgent,
      adapterInfo: info ? {
        vendor: info.vendor || null,
        architecture: info.architecture || null,
        device: info.device || null,
        description: info.description || null,
        subgroupMinSize: info.subgroupMinSize ?? null,
        subgroupMaxSize: info.subgroupMaxSize ?? null,
      } : null,
      effectiveDeviceInitialized: Boolean(window._sf3d_device),
      effectiveWeightsInitialized: Boolean(window._sf3d_weights),
      effectivePipelinesInitialized: Boolean(window._sf3d_pipelines),
    };
  });
  if (!routeIdentity.backend.effectiveDeviceInitialized
    || !routeIdentity.backend.effectiveWeightsInitialized
    || !routeIdentity.backend.effectivePipelinesInitialized) {
    fail(new Error('SF3D runtime globals are incomplete'), 'browser-setup', routeIdentity.backend);
  }
  lastTrustworthyEvidence = { phase: 'browser-setup', routeIdentity };

  const imageBase64 = fs.readFileSync(IMAGE).toString('base64');
  await page.evaluate(async (base64) => {
    await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        window.__postProcessorAssayImage = image;
        resolve();
      };
      image.onerror = reject;
      image.src = `data:image/png;base64,${base64}`;
    });
  }, imageBase64);

  const arms = await page.evaluate(async () => {
    const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
    const device = window._sf3d_device;
    const pipelines = window._sf3d_pipelines;
    const weights = window._sf3d_weights;
    const inputImage = window.__postProcessorAssayImage;

    async function digest(arrayBuffer) {
      const bytes = new Uint8Array(await crypto.subtle.digest('SHA-256', arrayBuffer));
      return [...bytes].map((value) => value.toString(16).padStart(2, '0')).join('');
    }

    async function runArm(name, options) {
      const frameGaps = [];
      const progress = [];
      let active = true;
      let prior = null;
      const tick = (timestamp) => {
        if (prior != null) frameGaps.push({
          start: prior,
          end: timestamp,
          gap: timestamp - prior,
        });
        prior = timestamp;
        if (active) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
      const output = await runFullPipelineToGlb(
        device,
        pipelines,
        weights,
        inputImage,
        options,
        (message) => progress.push(message),
      );
      active = false;
      await new Promise((resolve) => setTimeout(resolve, 80));

      const span = output.stageSpans.find((entry) => entry.name === 'post-processor');
      if (!span) throw new Error(`${name}: missing post-processor stage span`);
      const localGaps = frameGaps
        .filter((frame) => frame.start < span.end && frame.end > span.start)
        .map((frame) => frame.gap)
        .sort((left, right) => left - right);
      const cooperative = output.cooperativeReports?.['post-processor'] ?? null;
      return {
        name,
        requestedOptions: options,
        glbBytes: output.glb.byteLength,
        glbSha256: await digest(output.glb),
        mesh: { vertices: output.numVertices, faces: output.numFaces },
        fullRouteWallMs: output.totalMs,
        postProcessor: {
          wallMs: span.end - span.start,
          frameCount: localGaps.length,
          p95GapMs: localGaps.length
            ? localGaps[Math.min(localGaps.length - 1, Math.ceil(localGaps.length * 0.95) - 1)]
            : null,
          p99GapMs: localGaps.length
            ? localGaps[Math.min(localGaps.length - 1, Math.ceil(localGaps.length * 0.99) - 1)]
            : null,
          maxGapMs: localGaps.at(-1) ?? null,
          gapsOver16_7: localGaps.filter((gap) => gap > 16.7).length,
          gapsOver33_3: localGaps.filter((gap) => gap > 33.3).length,
          gapsOver100: localGaps.filter((gap) => gap > 100).length,
        },
        cooperative: cooperative ? {
          status: cooperative.status,
          schedulingMode: cooperative.schedulingMode,
          queueCompletionAuthority: cooperative.queueCompletionAuthority,
          completedItems: cooperative.progress?.completedItems,
          totalItems: cooperative.progress?.totalItems,
          rangeCount: cooperative.boundaries?.[0]?.actualRangeCount,
        } : null,
        progress,
      };
    }

    const control = await runArm('monolithic-control', {
      cooperativeDino: false,
      cooperativePostProcessor: false,
    });
    const candidate = await runArm('three-plane-cooperative', {
      cooperativeDino: false,
      cooperativePostProcessor: true,
      postProcessorSchedulingMode: 'cooperative',
    });
    return { control, candidate };
  });

  lastTrustworthyEvidence = { phase: 'browser-arms', routeIdentity, arms };
  if (pageErrors.length) fail(
    new Error(`page errors: ${pageErrors.join(' | ')}`),
    'browser-arms',
  );
  const outputIdentical = arms.control.glbSha256 === arms.candidate.glbSha256
    && arms.control.glbBytes === arms.candidate.glbBytes
    && JSON.stringify(arms.control.mesh) === JSON.stringify(arms.candidate.mesh);
  const cooperativeComplete = arms.candidate.cooperative?.status === 'succeeded'
    && arms.candidate.cooperative?.queueCompletionAuthority === 'per-gpu-duty-prefix-fence'
    && arms.candidate.cooperative?.completedItems === 3
    && arms.candidate.cooperative?.totalItems === 3
    && arms.candidate.cooperative?.rangeCount === 3;
  const progressHonest = arms.candidate.progress.some(
    (message) => /Post-processor planes 3\/3 \(100%\)/.test(message),
  );
  const report = {
    schema: 'sf3d.cooperative-post-processor-ab.v0',
    ok: outputIdentical && cooperativeComplete && progressHonest,
    routeIdentity,
    evidenceAuthority: source.clean ? 'clean-commit' : 'explicit-dirty-diff',
    outputIdentical,
    cooperativeComplete,
    progressHonest,
    deltas: {
      fullRouteWallMs: arms.candidate.fullRouteWallMs - arms.control.fullRouteWallMs,
      postProcessorWallMs:
        arms.candidate.postProcessor.wallMs - arms.control.postProcessor.wallMs,
      maxGapMs: arms.candidate.postProcessor.maxGapMs - arms.control.postProcessor.maxGapMs,
    },
    arms,
  };
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));

  console.log('\n=== SF3D cooperative postprocessor A/B ===');
  console.log(`output identical: ${outputIdentical} (${arms.control.glbSha256.slice(0, 12)}...)`);
  for (const arm of [arms.control, arms.candidate]) {
    console.log(
      `${arm.name}: full=${arm.fullRouteWallMs.toFixed(1)}ms `
      + `post=${arm.postProcessor.wallMs.toFixed(1)}ms `
      + `p95=${arm.postProcessor.p95GapMs?.toFixed(1)}ms `
      + `p99=${arm.postProcessor.p99GapMs?.toFixed(1)}ms `
      + `max=${arm.postProcessor.maxGapMs?.toFixed(1)}ms`,
    );
  }
  console.log(
    `delta: full=${report.deltas.fullRouteWallMs.toFixed(1)}ms `
    + `post=${report.deltas.postProcessorWallMs.toFixed(1)}ms `
    + `maxGap=${report.deltas.maxGapMs.toFixed(1)}ms`,
  );
  console.log(`report: ${REPORT_PATH}`);
  if (!report.ok) fail(
    new Error(
      `acceptance failed: identical=${outputIdentical} `
      + `cooperativeComplete=${cooperativeComplete} progressHonest=${progressHonest}`,
    ),
    'acceptance',
    report,
  );
} catch (error) {
  if (!fs.existsSync(REPORT_PATH)) writeFailure(error, 'exception');
  process.exitCode = 1;
} finally {
  if (browser) await browser.close().catch(() => {});
  cleanup();
}
