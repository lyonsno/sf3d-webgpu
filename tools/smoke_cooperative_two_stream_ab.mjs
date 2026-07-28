#!/usr/bin/env node
/**
 * Real-browser A/B for the dependency-safe SF3D two-stream stage graph.
 *
 * Both arms run the exact full image-to-GLB route with cooperative DINO. The
 * control retains one two-stream command buffer; the candidate submits the
 * attention query tiles, fuse-out row-range linears, and dependency
 * transitions as 4,218 duties. Every terminal path writes a report.
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import puppeteer from 'puppeteer-core';

const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const argValue = (flag, fallback) => {
  const index = process.argv.indexOf(flag);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
};
const EXPECTED_REVISION = argValue('--expected-revision', '');
const IMAGE = path.resolve(argValue(
  '--image',
  path.join(process.env.HOME, '.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png'),
));
const REPORT_PATH = path.resolve(argValue(
  '--report',
  '/tmp/sf3d-cooperative-two-stream-ab.json',
));
const ARM = argValue('--arm', 'both');
if (!['both', 'control', 'candidate'].includes(ARM)) {
  throw new RangeError(`--arm must be both, control, or candidate; got ${ARM}`);
}
const ALLOW_DIRTY = process.argv.includes('--allow-dirty-source');
const LINEAR_ROWS_PER_DUTY = Number(argValue('--linear-rows-per-duty', '128'));
if (!Number.isSafeInteger(LINEAR_ROWS_PER_DUTY) || LINEAR_ROWS_PER_DUTY <= 0) {
  throw new TypeError(
    `--linear-rows-per-duty must be a positive safe integer; `
    + `got ${LINEAR_ROWS_PER_DUTY}`,
  );
}
const LINEAR_RANGE_COUNT = Math.ceil(27648 / LINEAR_ROWS_PER_DUTY);
const EXPECTED_DUTY_COUNT = 1626 + 12 * LINEAR_RANGE_COUNT;
const WEIGHTS = path.join(REPO, 'public', 'weights.bin');
const EXPECTED_GLB_SHA256 =
  'e1f70de3407df24d571bf68f70fac2b59373bdd948075a2387f1834e4faff8b7';
const EXPECTED_GLB_BYTES = 2511516;
const children = [];
let browser = null;
let lastTrustworthyEvidence = { phase: 'argument-parse' };

const git = (args) => execFileSync('git', args, { cwd: REPO, encoding: 'utf8' }).trim();
const sha256File = async (file) => new Promise((resolve, reject) => {
  const hash = crypto.createHash('sha256');
  const stream = fs.createReadStream(file);
  stream.on('data', chunk => hash.update(chunk));
  stream.on('error', reject);
  stream.on('end', () => resolve(hash.digest('hex')));
});
const allocatePort = () => new Promise((resolve, reject) => {
  const server = net.createServer();
  server.unref();
  server.on('error', reject);
  server.listen(0, '127.0.0.1', () => {
    const { port } = server.address();
    server.close(() => resolve(port));
  });
});

function sourceIdentity() {
  const revision = git(['rev-parse', 'HEAD']);
  const dirtyPaths = git(['status', '--short']).split('\n').filter(Boolean);
  const hash = crypto.createHash('sha256');
  hash.update(execFileSync('git', ['diff', 'HEAD', '--binary'], { cwd: REPO }));
  for (const relativePath of git(['ls-files', '--others', '--exclude-standard'])
    .split('\n').filter(Boolean).sort()) {
    hash.update(relativePath);
    const file = path.join(REPO, relativePath);
    if (fs.statSync(file).isFile()) hash.update(fs.readFileSync(file));
  }
  return {
    revision,
    requestedRevision: EXPECTED_REVISION || null,
    matchesRequestedRevision: Boolean(EXPECTED_REVISION)
      && git(['rev-parse', '--verify', `${EXPECTED_REVISION}^{commit}`]) === revision,
    clean: dirtyPaths.length === 0,
    dirtyModeExplicit: ALLOW_DIRTY,
    dirtyDiffSha256: dirtyPaths.length ? hash.digest('hex') : null,
    dirtyPaths,
    worktree: REPO,
  };
}

function writeFailure(error, phase, details = null) {
  const report = {
    schema: 'sf3d.cooperative-two-stream-ab.v1',
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
  writeFailure(error, phase, details);
  throw error instanceof Error ? error : new Error(String(error));
}

try {
  const source = sourceIdentity();
  lastTrustworthyEvidence = { phase: 'source-identity', source };
  if (!EXPECTED_REVISION) fail(
    new Error('--expected-revision is required'),
    'source-identity',
  );
  if (!source.matchesRequestedRevision) fail(
    new Error(`effective revision ${source.revision} does not match ${EXPECTED_REVISION}`),
    'source-identity',
  );
  if (!source.clean && !ALLOW_DIRTY) fail(
    new Error(`dirty source requires --allow-dirty-source; diff=${source.dirtyDiffSha256}`),
    'source-identity',
  );
  if (!fs.existsSync(IMAGE)) fail(new Error(`missing input ${IMAGE}`), 'input');
  if (!fs.existsSync(WEIGHTS)) fail(new Error(`missing weights ${WEIGHTS}`), 'weights');

  const effectiveWeights = fs.realpathSync(WEIGHTS);
  const routeIdentity = {
    source,
    input: {
      path: IMAGE,
      bytes: fs.statSync(IMAGE).size,
      sha256: await sha256File(IMAGE),
    },
    weights: {
      requestedPath: WEIGHTS,
      effectivePath: effectiveWeights,
      bytes: fs.statSync(effectiveWeights).size,
      sha256: await sha256File(effectiveWeights),
    },
    requestedArms: [
      {
        name: 'monolithic-two-stream-control',
        cooperativeDino: true,
        cooperativeTwoStream: false,
      },
      {
        name: 'fine-duty-candidate',
        cooperativeDino: true,
        cooperativeTwoStream: true,
        twoStreamSchedulingMode: 'cooperative',
        twoStreamDutyGranularity: 'attention-tile',
        twoStreamLinearRowsPerDuty: LINEAR_ROWS_PER_DUTY,
      },
    ].filter(arm => ARM === 'both'
      || (ARM === 'control' && arm.name === 'monolithic-two-stream-control')
      || (ARM === 'candidate' && arm.name === 'fine-duty-candidate')),
  };
  lastTrustworthyEvidence = { phase: 'route-identity', routeIdentity };

  const port = await allocatePort();
  const vite = spawn(
    'npx',
    ['vite', '--host', '127.0.0.1', '--port', String(port), '--strictPort'],
    { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  children.push(vite);
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Vite startup timed out')), 40000);
    vite.stdout.on('data', data => {
      if (/Local:|ready/.test(data.toString())) {
        clearTimeout(timeout);
        resolve();
      }
    });
    vite.on('error', reject);
    vite.on('exit', code => {
      if (code && code !== 0) reject(new Error(`Vite exited ${code}`));
    });
  }).catch(error => fail(error, 'vite'));

  browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: false,
    protocolTimeout: 900000,
    args: ['--enable-unsafe-webgpu', '--use-angle=metal'],
  });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto(`http://127.0.0.1:${port}/`, {
    waitUntil: 'domcontentloaded',
    timeout: 30000,
  });

  let status = '';
  const setupStartedAt = Date.now();
  while (Date.now() - setupStartedAt < 180000) {
    status = await page.$eval('#status', element => element.textContent).catch(() => '');
    if (status.includes('Ready') || /error|failed/i.test(status)) break;
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  if (!status.includes('Ready') || pageErrors.length) fail(
    new Error(`browser setup incomplete: status=${JSON.stringify(status)}`),
    'browser-setup',
    { pageErrors },
  );

  routeIdentity.backend = await page.evaluate(() => ({
    userAgent: navigator.userAgent,
    webgpu: Boolean(navigator.gpu),
    effectiveDeviceInitialized: Boolean(window._sf3d_device),
    effectiveWeightsInitialized: Boolean(window._sf3d_weights),
    effectivePipelinesInitialized: Boolean(window._sf3d_pipelines),
  }));
  if (!routeIdentity.backend.webgpu
    || !routeIdentity.backend.effectiveDeviceInitialized
    || !routeIdentity.backend.effectiveWeightsInitialized
    || !routeIdentity.backend.effectivePipelinesInitialized) {
    fail(new Error('effective SF3D WebGPU route is incomplete'), 'browser-setup');
  }
  lastTrustworthyEvidence = { phase: 'browser-setup', routeIdentity };

  const imageBase64 = fs.readFileSync(IMAGE).toString('base64');
  await page.evaluate(async base64 => {
    await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        window.__twoStreamAssayImage = image;
        resolve();
      };
      image.onerror = reject;
      image.src = `data:image/png;base64,${base64}`;
    });
  }, imageBase64);

  const arms = await page.evaluate(async ({ armSelection, linearRowsPerDuty }) => {
    const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
    const device = window._sf3d_device;
    const pipelines = window._sf3d_pipelines;
    const weights = window._sf3d_weights;
    const image = window.__twoStreamAssayImage;

    async function digest(buffer) {
      const bytes = new Uint8Array(await crypto.subtle.digest('SHA-256', buffer));
      return [...bytes].map(value => value.toString(16).padStart(2, '0')).join('');
    }
    async function runArm(name, options) {
      const progress = [];
      const frameGaps = [];
      let frameProbeActive = true;
      let priorFrameAt = null;
      const probeFrame = timestamp => {
        if (priorFrameAt != null) {
          frameGaps.push({
            start: priorFrameAt,
            end: timestamp,
            gap: timestamp - priorFrameAt,
          });
        }
        priorFrameAt = timestamp;
        if (frameProbeActive) requestAnimationFrame(probeFrame);
      };
      requestAnimationFrame(probeFrame);
      const output = await runFullPipelineToGlb(
        device,
        pipelines,
        weights,
        image,
        options,
        message => progress.push(message),
      );
      frameProbeActive = false;
      await new Promise(resolve => setTimeout(resolve, 80));
      const span = output.stageSpans.find(entry => entry.name === 'two-stream-backbone');
      if (!span) throw new Error(`${name}: missing two-stream stage span`);
      const cooperative = output.cooperativeReports?.['two-stream-backbone'] ?? null;
      const localGaps = frameGaps
        .filter(frame => frame.start < span.end && frame.end > span.start)
        .map(frame => frame.gap)
        .sort((left, right) => left - right);
      const percentile = percentileValue => localGaps.length
        ? localGaps[
          Math.min(
            localGaps.length - 1,
            Math.ceil(localGaps.length * percentileValue) - 1,
          )
        ]
        : null;
      return {
        name,
        requestedOptions: options,
        glbBytes: output.glb.byteLength,
        glbSha256: await digest(output.glb),
        mesh: { vertices: output.numVertices, faces: output.numFaces },
        fullRouteWallMs: output.totalMs,
        twoStreamWallMs: span.end - span.start,
        twoStreamCadence: {
          frameCount: localGaps.length,
          p95GapMs: percentile(0.95),
          p99GapMs: percentile(0.99),
          maxGapMs: localGaps.at(-1) ?? null,
          gapsOver16_7: localGaps.filter(gap => gap > 16.7).length,
          gapsOver33_3: localGaps.filter(gap => gap > 33.3).length,
          gapsOver100: localGaps.filter(gap => gap > 100).length,
        },
        cooperative: cooperative ? {
          status: cooperative.status,
          schedulingMode: cooperative.schedulingMode,
          queueCompletionAuthority: cooperative.queueCompletionAuthority,
          completedItems: cooperative.progress?.completedItems,
          totalItems: cooperative.progress?.totalItems,
          rangeCount: cooperative.boundaries?.[0]?.actualRangeCount,
          adapterTelemetry: cooperative.adapterTelemetry,
        } : null,
        progress,
      };
    }

    const common = {
      cooperativeDino: true,
      dinoSchedulingMode: 'cooperative',
      dinoChunkBlocks: 1,
      cooperativePostProcessor: false,
    };
    const arms = { control: null, candidate: null };
    if (armSelection !== 'candidate') {
      arms.control = await runArm('monolithic-two-stream-control', {
        ...common,
        cooperativeTwoStream: false,
      });
    }
    if (armSelection !== 'control') {
      arms.candidate = await runArm('fine-duty-candidate', {
        ...common,
        cooperativeTwoStream: true,
        twoStreamSchedulingMode: 'cooperative',
        twoStreamDutyGranularity: 'attention-tile',
        twoStreamLinearRowsPerDuty: linearRowsPerDuty,
      });
    }
    return arms;
  }, { armSelection: ARM, linearRowsPerDuty: LINEAR_ROWS_PER_DUTY });

  lastTrustworthyEvidence = { phase: 'browser-arms', routeIdentity, arms };
  if (pageErrors.length) fail(
    new Error(`page errors: ${pageErrors.join(' | ')}`),
    'browser-arms',
  );
  const outputIdentical = arms.control && arms.candidate
    ? arms.control.glbSha256 === arms.candidate.glbSha256
      && arms.control.glbBytes === arms.candidate.glbBytes
      && JSON.stringify(arms.control.mesh) === JSON.stringify(arms.candidate.mesh)
    : null;
  const outputCanonical = [arms.control, arms.candidate]
    .filter(Boolean)
    .every(arm => arm.glbSha256 === EXPECTED_GLB_SHA256
      && arm.glbBytes === EXPECTED_GLB_BYTES);
  const cooperativeComplete = !arms.candidate
    || (arms.candidate.cooperative?.status === 'succeeded'
    && arms.candidate.cooperative?.queueCompletionAuthority === 'per-gpu-duty-prefix-fence'
    && arms.candidate.cooperative?.completedItems === EXPECTED_DUTY_COUNT
    && arms.candidate.cooperative?.totalItems === EXPECTED_DUTY_COUNT
    && arms.candidate.cooperative?.rangeCount === EXPECTED_DUTY_COUNT
    && arms.candidate.cooperative?.adapterTelemetry?.stageDuties?.length
      === EXPECTED_DUTY_COUNT
    && arms.candidate.cooperative?.adapterTelemetry?.queueFences?.length
      === EXPECTED_DUTY_COUNT
    && arms.candidate.cooperative?.adapterTelemetry?.browserYields?.length
      === EXPECTED_DUTY_COUNT);
  const progressHonest = !arms.candidate || arms.candidate.progress.some(
    message => message.includes(
      `Two-stream duties ${EXPECTED_DUTY_COUNT}/${EXPECTED_DUTY_COUNT} (100%)`,
    ),
  );
  const report = {
    schema: 'sf3d.cooperative-two-stream-ab.v1',
    ok: outputIdentical !== false
      && outputCanonical
      && cooperativeComplete
      && progressHonest,
    routeIdentity,
    evidenceAuthority: source.clean ? 'clean-commit' : 'explicit-dirty-diff',
    outputIdentical,
    outputCanonical,
    cooperativeComplete,
    progressHonest,
    deltas: {
      fullRouteWallMs: arms.control && arms.candidate
        ? arms.candidate.fullRouteWallMs - arms.control.fullRouteWallMs
        : null,
      twoStreamWallMs: arms.control && arms.candidate
        ? arms.candidate.twoStreamWallMs - arms.control.twoStreamWallMs
        : null,
    },
    arms,
  };
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));

  console.log('\n=== SF3D cooperative two-stream A/B ===');
  console.log(`output identical: ${outputIdentical}`);
  if (arms.control) {
    console.log(
      `control: full=${arms.control.fullRouteWallMs.toFixed(1)}ms `
      + `two-stream=${arms.control.twoStreamWallMs.toFixed(1)}ms`,
    );
  }
  if (arms.candidate) {
    console.log(
      `candidate: full=${arms.candidate.fullRouteWallMs.toFixed(1)}ms `
      + `two-stream=${arms.candidate.twoStreamWallMs.toFixed(1)}ms`,
    );
  }
  if (arms.control && arms.candidate) {
    console.log(
      `delta: full=${report.deltas.fullRouteWallMs.toFixed(1)}ms `
      + `two-stream=${report.deltas.twoStreamWallMs.toFixed(1)}ms`,
    );
  }
  console.log(`report: ${REPORT_PATH}`);
  if (!report.ok) fail(
    new Error(
      `acceptance failed: identical=${outputIdentical} `
      + `canonical=${outputCanonical} `
      + `cooperativeComplete=${cooperativeComplete} progressHonest=${progressHonest}`,
    ),
    'acceptance',
    report,
  );
} catch (error) {
  if (!fs.existsSync(REPORT_PATH)) writeFailure(error, 'exception');
  console.error(`SMOKE FAILED: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
} finally {
  if (browser) await browser.close().catch(() => {});
  for (const child of children) {
    try { child.kill(); } catch {}
  }
}
