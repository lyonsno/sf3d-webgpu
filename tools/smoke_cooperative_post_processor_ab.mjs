#!/usr/bin/env node
/**
 * Real-browser A/B/C/D for SF3D postprocessor command boundaries.
 *
 * Both arms execute the same complete image-to-GLB route. The control keeps the
 * postprocessor in one command buffer; the plane arm submits three independent
 * planes; the layer arm submits the eighteen dependency-safe gather/conv/
 * PixelShuffle duties. The channel arm splits every convolution into exact
 * caller-selected output-channel ranges. The witness requires byte-identical GLBs and reports
 * wall/cadence only for the exact postprocessor span, while preserving
 * route/source/backend identities and writing a failure report at every
 * terminal path.
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import puppeteer from 'puppeteer-core';
import { readJsonReport, writeJsonReportAtomic } from './json_report_atomic.mjs';
import { evaluatePostProcessorSmokeAcceptance } from './post_processor_smoke_acceptance.mjs';

const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const CHROME_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--use-angle=metal',
  '--disable-background-timer-throttling',
  '--disable-backgrounding-occluded-windows',
  '--disable-renderer-backgrounding',
]);
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
const CHANNELS_PER_DUTY = Number(argValue('--channels-per-duty', '16'));
const ARM_SELECTION = argValue('--arms', 'all');
const ALLOW_DIRTY_SOURCE = process.argv.includes('--allow-dirty-source');
const WEIGHTS = path.join(REPO, 'public', 'weights.bin');
const processes = [];
let browser = null;
let lastTrustworthyEvidence = { phase: 'argument-parse' };
let reportFinalized = false;

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

async function closeOwnedBrowser() {
  if (!browser) return { status: 'not-started', ownedPid: null };
  const ownedProcess = browser.process?.() ?? null;
  const ownedPid = ownedProcess?.pid ?? null;
  let timer = null;
  try {
    const outcome = await Promise.race([
      browser.close().then(() => 'graceful'),
      new Promise((resolve) => {
        timer = setTimeout(() => resolve('timeout'), 10000);
      }),
    ]);
    if (outcome === 'graceful') return { status: 'graceful', ownedPid };
    const signalSent = ownedProcess?.kill('SIGTERM') ?? false;
    browser.disconnect();
    return {
      status: 'forced-owned-process',
      ownedPid,
      signal: 'SIGTERM',
      signalSent,
    };
  } catch (error) {
    return {
      status: 'failed',
      ownedPid,
      error: {
        name: error?.name || 'Error',
        message: error?.message || String(error),
      },
    };
  } finally {
    if (timer) clearTimeout(timer);
    browser = null;
  }
}

function appendTeardown(teardown) {
  if (!fs.existsSync(REPORT_PATH)) return;
  const report = readJsonReport(REPORT_PATH);
  report.teardown = teardown;
  if (teardown.status === 'failed') {
    report.primaryAcceptanceOk = report.ok;
    report.ok = false;
    report.failure = {
      phase: 'browser-teardown',
      message: teardown.error?.message || 'browser teardown failed',
      details: teardown,
    };
    process.exitCode = 1;
  }
  writeJsonReportAtomic(REPORT_PATH, report);
}

function writeFailure(error, phase, details = null) {
  const report = {
    schema: 'sf3d.cooperative-post-processor-abcd.v1',
    ok: false,
    failure: {
      phase,
      message: error instanceof Error ? error.message : String(error),
      details,
    },
    lastTrustworthyEvidence,
  };
  writeJsonReportAtomic(REPORT_PATH, report);
  reportFinalized = true;
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
  writeJsonReportAtomic(REPORT_PATH, {
    schema: 'sf3d.cooperative-post-processor-abcd.v1',
    ok: false,
    status: 'running',
    lastTrustworthyEvidence,
  });
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
  if (!Number.isSafeInteger(CHANNELS_PER_DUTY) || CHANNELS_PER_DUTY <= 0) fail(
    new Error('--channels-per-duty must be a positive safe integer'),
    'argument-parse',
  );
  if (!['all', 'pair', 'channel'].includes(ARM_SELECTION)) fail(
    new Error('--arms must be all, pair, or channel'),
    'argument-parse',
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
        postProcessorDutyGranularity: 'plane',
      },
      {
        name: 'eighteen-stage-cooperative',
        cooperativePostProcessor: true,
        postProcessorSchedulingMode: 'cooperative',
        postProcessorDutyGranularity: 'layer',
      },
      {
        name: 'channel-range-cooperative',
        cooperativePostProcessor: true,
        postProcessorSchedulingMode: 'cooperative',
        postProcessorDutyGranularity: 'channel-range',
        postProcessorChannelsPerDuty: CHANNELS_PER_DUTY,
      },
    ].filter((arm) => ARM_SELECTION === 'all'
      || (ARM_SELECTION === 'pair'
        && ['monolithic-control', 'channel-range-cooperative'].includes(arm.name))
      || (ARM_SELECTION === 'channel' && arm.name === 'channel-range-cooperative')),
    armSelection: ARM_SELECTION,
    browser: {
      executablePath: CHROME_PATH,
      headless: false,
      args: [...CHROME_ARGS],
    },
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
    args: [...CHROME_ARGS],
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
  await page.bringToFront();

  const arms = await page.evaluate(async ({ channelsPerDuty, armSelection }) => {
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
      let prior = await new Promise((resolve) => requestAnimationFrame(resolve));
      const visibilityAtStart = {
        visibilityState: document.visibilityState,
        hasFocus: document.hasFocus(),
      };
      const tick = (timestamp) => {
        frameGaps.push({
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
      const visibilityAtEnd = {
        visibilityState: document.visibilityState,
        hasFocus: document.hasFocus(),
      };

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
        browserState: {
          visibilityAtStart,
          visibilityAtEnd,
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

    const control = armSelection === 'all' || armSelection === 'pair'
      ? await runArm('monolithic-control', {
        cooperativeDino: false,
        cooperativePostProcessor: false,
      })
      : null;
    const plane = armSelection === 'all'
      ? await runArm('three-plane-cooperative', {
        cooperativeDino: false,
        cooperativePostProcessor: true,
        postProcessorSchedulingMode: 'cooperative',
        postProcessorDutyGranularity: 'plane',
      })
      : null;
    const candidate = armSelection === 'all'
      ? await runArm('eighteen-stage-cooperative', {
        cooperativeDino: false,
        cooperativePostProcessor: true,
        postProcessorSchedulingMode: 'cooperative',
        postProcessorDutyGranularity: 'layer',
      })
      : null;
    const channel = await runArm('channel-range-cooperative', {
      cooperativeDino: false,
      cooperativePostProcessor: true,
      postProcessorSchedulingMode: 'cooperative',
      postProcessorDutyGranularity: 'channel-range',
      postProcessorChannelsPerDuty: channelsPerDuty,
    });
    return { control, plane, candidate, channel };
  }, { channelsPerDuty: CHANNELS_PER_DUTY, armSelection: ARM_SELECTION });

  lastTrustworthyEvidence = { phase: 'browser-arms', routeIdentity, arms };
  if (pageErrors.length) fail(
    new Error(`page errors: ${pageErrors.join(' | ')}`),
    'browser-arms',
  );
  const outputIdentical = arms.control
    ? arms.control.glbSha256 === arms.channel.glbSha256
      && arms.control.glbBytes === arms.channel.glbBytes
      && JSON.stringify(arms.control.mesh) === JSON.stringify(arms.channel.mesh)
      && (!arms.plane || (
        arms.control.glbSha256 === arms.plane.glbSha256
        && arms.control.glbBytes === arms.plane.glbBytes
        && JSON.stringify(arms.control.mesh) === JSON.stringify(arms.plane.mesh)
      ))
      && (!arms.candidate || (
        arms.control.glbSha256 === arms.candidate.glbSha256
        && arms.control.glbBytes === arms.candidate.glbBytes
        && JSON.stringify(arms.control.mesh) === JSON.stringify(arms.candidate.mesh)
      ))
    : null;
  const layerComplete = !arms.candidate || (arms.candidate.cooperative?.status === 'succeeded'
    && arms.candidate.cooperative?.queueCompletionAuthority === 'per-gpu-duty-prefix-fence'
    && arms.candidate.cooperative?.completedItems === 18
    && arms.candidate.cooperative?.totalItems === 18
    && arms.candidate.cooperative?.rangeCount === 18
    && arms.candidate.cooperative?.adapterTelemetry?.stageDuties?.length === 18
    && arms.candidate.cooperative?.adapterTelemetry?.queueFences?.length === 18
    && arms.candidate.cooperative?.adapterTelemetry?.browserYields?.length === 18);
  const layerProgressHonest = !arms.candidate || arms.candidate.progress.some(
    (message) => /Post-processor duties 18\/18 \(100%\)/.test(message),
  );
  const expectedChannelDuties = 3 * (
    2
    + 3 * Math.ceil(1024 / CHANNELS_PER_DUTY)
    + Math.ceil(640 / CHANNELS_PER_DUTY)
  );
  const channelComplete = arms.channel.cooperative?.status === 'succeeded'
    && arms.channel.cooperative?.queueCompletionAuthority === 'per-gpu-duty-prefix-fence'
    && arms.channel.cooperative?.completedItems === expectedChannelDuties
    && arms.channel.cooperative?.totalItems === expectedChannelDuties
    && arms.channel.cooperative?.rangeCount === expectedChannelDuties
    && arms.channel.cooperative?.adapterTelemetry?.channelsPerDuty === CHANNELS_PER_DUTY
    && arms.channel.cooperative?.adapterTelemetry?.stageDuties?.length === expectedChannelDuties
    && arms.channel.cooperative?.adapterTelemetry?.queueFences?.length === expectedChannelDuties
    && arms.channel.cooperative?.adapterTelemetry?.browserYields?.length === expectedChannelDuties;
  const channelProgressHonest = arms.channel.progress.some(
    (message) => new RegExp(
      `Post-processor duties ${expectedChannelDuties}/${expectedChannelDuties} \\(100%\\)`,
    ).test(message),
  );
  const cooperativeComplete = layerComplete && channelComplete;
  const progressHonest = layerProgressHonest && channelProgressHonest;
  const selectedArms = [
    arms.control,
    arms.plane,
    arms.candidate,
    arms.channel,
  ].filter(Boolean);
  const cadenceObserved = selectedArms.every((arm) => (
    arm.postProcessor.frameCount > 0
    && Number.isFinite(arm.postProcessor.p95GapMs)
    && Number.isFinite(arm.postProcessor.p99GapMs)
    && Number.isFinite(arm.postProcessor.maxGapMs)
    && arm.browserState.visibilityAtStart.visibilityState === 'visible'
    && arm.browserState.visibilityAtEnd.visibilityState === 'visible'
  ));
  const acceptance = evaluatePostProcessorSmokeAcceptance({
    armSelection: ARM_SELECTION,
    outputIdentical,
    cooperativeComplete,
    progressHonest,
    cadenceObserved,
  });
  const report = {
    schema: 'sf3d.cooperative-post-processor-abcd.v1',
    ok: acceptance.ok,
    paired: acceptance.paired,
    routeIdentity,
    evidenceAuthority: source.clean ? 'clean-commit' : 'explicit-dirty-diff',
    outputIdentical,
    cooperativeComplete,
    progressHonest,
    cadenceObserved,
    expectedChannelDuties,
    deltas: {
      fullRouteWallMs: arms.candidate && arms.control
        ? arms.candidate.fullRouteWallMs - arms.control.fullRouteWallMs
        : null,
      postProcessorWallMs: arms.candidate && arms.control
        ? arms.candidate.postProcessor.wallMs - arms.control.postProcessor.wallMs
        : null,
      maxGapMs: arms.candidate && arms.control
        ? arms.candidate.postProcessor.maxGapMs - arms.control.postProcessor.maxGapMs
        : null,
      layerVersusPlane: arms.candidate && arms.plane ? {
        fullRouteWallMs: arms.candidate.fullRouteWallMs - arms.plane.fullRouteWallMs,
        postProcessorWallMs:
          arms.candidate.postProcessor.wallMs - arms.plane.postProcessor.wallMs,
        maxGapMs: arms.candidate.postProcessor.maxGapMs - arms.plane.postProcessor.maxGapMs,
      } : null,
      channelVersusControl: arms.control ? {
        fullRouteWallMs: arms.channel.fullRouteWallMs - arms.control.fullRouteWallMs,
        postProcessorWallMs:
          arms.channel.postProcessor.wallMs - arms.control.postProcessor.wallMs,
        maxGapMs: arms.channel.postProcessor.maxGapMs - arms.control.postProcessor.maxGapMs,
      } : null,
      channelVersusLayer: arms.candidate ? {
        fullRouteWallMs: arms.channel.fullRouteWallMs - arms.candidate.fullRouteWallMs,
        postProcessorWallMs:
          arms.channel.postProcessor.wallMs - arms.candidate.postProcessor.wallMs,
        maxGapMs: arms.channel.postProcessor.maxGapMs - arms.candidate.postProcessor.maxGapMs,
      } : null,
    },
    arms,
  };
  writeJsonReportAtomic(REPORT_PATH, report);
  reportFinalized = true;

  console.log('\n=== SF3D cooperative postprocessor A/B/C/D ===');
  console.log(
    `output identical: ${outputIdentical} `
    + `(${(arms.control ?? arms.channel).glbSha256.slice(0, 12)}...)`,
  );
  for (const arm of [arms.control, arms.plane, arms.candidate, arms.channel].filter(Boolean)) {
    console.log(
      `${arm.name}: full=${arm.fullRouteWallMs.toFixed(1)}ms `
      + `post=${arm.postProcessor.wallMs.toFixed(1)}ms `
      + `p95=${arm.postProcessor.p95GapMs?.toFixed(1)}ms `
      + `p99=${arm.postProcessor.p99GapMs?.toFixed(1)}ms `
      + `max=${arm.postProcessor.maxGapMs?.toFixed(1)}ms`,
    );
  }
  if (report.deltas.fullRouteWallMs != null) console.log(
    `layer-control delta: full=${report.deltas.fullRouteWallMs.toFixed(1)}ms `
      + `post=${report.deltas.postProcessorWallMs.toFixed(1)}ms `
      + `maxGap=${report.deltas.maxGapMs.toFixed(1)}ms`,
  );
  if (report.deltas.channelVersusControl) console.log(
    `channel-control delta: full=${report.deltas.channelVersusControl.fullRouteWallMs.toFixed(1)}ms `
      + `post=${report.deltas.channelVersusControl.postProcessorWallMs.toFixed(1)}ms `
      + `maxGap=${report.deltas.channelVersusControl.maxGapMs.toFixed(1)}ms`,
  );
  if (report.deltas.channelVersusLayer) console.log(
    `channel-layer delta: full=${report.deltas.channelVersusLayer.fullRouteWallMs.toFixed(1)}ms `
      + `post=${report.deltas.channelVersusLayer.postProcessorWallMs.toFixed(1)}ms `
      + `maxGap=${report.deltas.channelVersusLayer.maxGapMs.toFixed(1)}ms`,
  );
  if (report.deltas.layerVersusPlane) console.log(
    `layer-plane delta: full=${report.deltas.layerVersusPlane.fullRouteWallMs.toFixed(1)}ms `
      + `post=${report.deltas.layerVersusPlane.postProcessorWallMs.toFixed(1)}ms `
      + `maxGap=${report.deltas.layerVersusPlane.maxGapMs.toFixed(1)}ms`,
  );
  console.log(`report: ${REPORT_PATH}`);
  if (acceptance.paired && !report.ok) fail(
    new Error(
      `acceptance failed: identical=${outputIdentical} `
      + `cooperativeComplete=${cooperativeComplete} progressHonest=${progressHonest} `
      + `cadenceObserved=${cadenceObserved}`,
    ),
    'acceptance',
    report,
  );
} catch (error) {
  if (!reportFinalized) writeFailure(error, 'exception');
  process.exitCode = 1;
} finally {
  try {
    appendTeardown(await closeOwnedBrowser());
  } catch (error) {
    writeFailure(error, 'report-finalization');
    process.exitCode = 1;
  } finally {
    cleanup();
  }
}
