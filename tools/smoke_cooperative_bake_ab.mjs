#!/usr/bin/env node
/**
 * Cooperative texture-bake A/B smoke.
 *
 * Proves the second SF3D cooperative boundary (texture bake) preserves exact
 * texture output while batching the per-texel decode into yieldable GPU duties,
 * and measures the foreground-gap improvement on the texture-bake stage that
 * profiling flagged as the largest GPU gap (~231ms).
 *
 * Three arms over one loaded model:
 *   monolithic  — legacy single decode dispatch (control)
 *   coop-off    — cooperative facade, disabled scheduling (declared-work A/B)
 *   coop-on      — cooperative facade, per-batch submit + browser yield
 * Asserts: albedo+normal textures byte-identical across all three; coop-on
 * reports per-gpu-duty fence; measures texture-bake-window rAF gaps per arm.
 *
 * Usage: node tools/smoke_cooperative_bake_ab.mjs --expected-revision SHA [--image PATH] [--batch N]
 */
import puppeteer from 'puppeteer-core';
import path from 'node:path';
import fs from 'node:fs';
import net from 'node:net';
import crypto from 'node:crypto';
import { execFileSync, spawn } from 'node:child_process';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const argVal = (f, d) => { const i = process.argv.indexOf(f); return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d; };
const IMAGE = path.resolve(argVal('--image', path.join(process.env.HOME, '.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png')));
const BATCH = Number(argVal('--batch', '16384')) || 16384;
const REPORT_PATH = path.resolve(argVal('--report', '/tmp/sf3d-cooperative-bake-ab.json'));
const EXPECTED_REVISION = argVal('--expected-revision', process.env.SF3D_EXPECTED_REVISION || '');
const ALLOW_DIRTY_SOURCE = process.argv.includes('--allow-dirty-source');
const WEIGHT_PATH = path.join(REPO, 'public', 'weights.bin');
const dirtyStatus = execFileSync('git', ['status', '--short'], {
  cwd: REPO,
  encoding: 'utf8',
}).trim();
const dirtyPaths = dirtyStatus.split('\n').filter(Boolean);
let dirtyDiffSha256 = null;
if (dirtyPaths.length > 0) {
  const dirtyHash = crypto.createHash('sha256');
  dirtyHash.update(execFileSync('git', ['diff', 'HEAD', '--binary'], { cwd: REPO }));
  const untracked = execFileSync(
    'git',
    ['ls-files', '--others', '--exclude-standard', '-z'],
    { cwd: REPO },
  ).toString().split('\0').filter(Boolean).sort();
  for (const relativePath of untracked) {
    dirtyHash.update(relativePath);
    const absolutePath = path.join(REPO, relativePath);
    if (fs.statSync(absolutePath).isFile()) dirtyHash.update(fs.readFileSync(absolutePath));
  }
  dirtyDiffSha256 = dirtyHash.digest('hex');
}
const effectiveRevision = execFileSync('git', ['rev-parse', 'HEAD'], {
  cwd: REPO,
  encoding: 'utf8',
}).trim();
let requestedRevisionResolution = null;
let requestedRevisionError = null;
if (EXPECTED_REVISION) {
  try {
    requestedRevisionResolution = execFileSync(
      'git',
      ['rev-parse', '--verify', `${EXPECTED_REVISION}^{commit}`],
      {
        cwd: REPO,
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'pipe'],
      },
    ).trim();
  } catch (error) {
    requestedRevisionError = error.stderr?.toString().trim() || error.message;
  }
}
const sourceIdentity = {
  revision: effectiveRevision,
  requestedRevision: EXPECTED_REVISION || null,
  requestedRevisionResolution,
  matchesRequestedRevision: Boolean(EXPECTED_REVISION)
    && requestedRevisionResolution === effectiveRevision,
  clean: dirtyPaths.length === 0,
  dirtyModeExplicit: ALLOW_DIRTY_SOURCE,
  dirtyDiffSha256,
  dirtyPaths,
  worktree: REPO,
};
const sourceAccepted = sourceIdentity.matchesRequestedRevision
  && (sourceIdentity.clean || sourceIdentity.dirtyModeExplicit);

const allocatePort = () => new Promise((res, rej) => { const s = net.createServer(); s.unref(); s.on('error', rej); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); }); });
const procs = [];
const cleanup = () => { for (const p of procs) { try { p.kill(); } catch {} } };
let activeRouteIdentity = { source: sourceIdentity, requestedBatchTexels: BATCH };
const fail = (m, failurePhase = 'unknown', failureDetails = null) => {
  fs.writeFileSync(REPORT_PATH, JSON.stringify({
    ok: false,
    failurePhase,
    lastTrustworthyEvidence: activeRouteIdentity,
    failureDetails,
    error: m,
  }, null, 2));
  console.error(`\n✗ SMOKE FAILED: ${m}`);
  cleanup();
  process.exit(1);
};

(async () => {
  if (!EXPECTED_REVISION) {
    fail('setup: --expected-revision or SF3D_EXPECTED_REVISION is required', 'source-identity');
  }
  if (requestedRevisionError) {
    fail(
      `setup: cannot resolve requested revision ${EXPECTED_REVISION}: ${requestedRevisionError}`,
      'source-identity',
    );
  }
  if (!sourceIdentity.matchesRequestedRevision) {
    fail(
      `setup: effective revision ${sourceIdentity.revision} does not match requested ${EXPECTED_REVISION}`,
      'source-identity',
    );
  }
  if (!sourceIdentity.clean && !ALLOW_DIRTY_SOURCE) {
    fail(
      `setup: worktree is dirty; diff sha256=${dirtyDiffSha256}`,
      'source-identity',
    );
  }
  if (!fs.existsSync(WEIGHT_PATH)) fail(`setup: missing model weights at ${WEIGHT_PATH}`, 'weights');
  const weightTarget = fs.realpathSync(WEIGHT_PATH);
  const weightStat = fs.statSync(weightTarget);
  const weightSha256 = execFileSync('shasum', ['-a', '256', weightTarget], { encoding: 'utf8' })
    .trim().split(/\s+/)[0];
  const routeIdentity = {
    source: sourceIdentity,
    inputImage: IMAGE,
    weights: {
      requestedPath: WEIGHT_PATH,
      effectivePath: weightTarget,
      bytes: weightStat.size,
      sha256: weightSha256,
    },
    requestedBatchTexels: BATCH,
  };
  activeRouteIdentity = routeIdentity;
  const port = await allocatePort();
  const vite = spawn('npx', ['vite', '--port', String(port), '--strictPort'], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(vite);
  await new Promise((res, rej) => { const to = setTimeout(() => rej(new Error('vite timeout')), 40000); vite.stdout.on('data', d => { if (/Local:|ready/.test(d.toString())) { clearTimeout(to); res(); } }); vite.on('error', e => { clearTimeout(to); rej(e); }); }).catch(e => fail(e.message));

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: false,
    protocolTimeout: 900000,
    args: ['--enable-unsafe-webgpu', '--use-angle=metal'],
  });
  const page = await browser.newPage();
  const pageErrors = []; page.on('pageerror', e => pageErrors.push(e.message));
  page.on('console', m => { const t = m.text(); if (/Texture bake|occupied|backbone/.test(t)) console.log(`[log] ${t.slice(0, 120)}`); });
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded', timeout: 30000 });
  const start = Date.now();
  let ready = false;
  let lastStatus = '';
  while (Date.now() - start < 180000) {
    lastStatus = await page.$eval('#status', el => el.textContent).catch(() => '');
    if (lastStatus.includes('Ready')) {
      ready = true;
      break;
    }
    if (/error|failed/i.test(lastStatus)) break;
    await new Promise(r => setTimeout(r, 500));
  }
  if (!ready || pageErrors.length) {
    fail(`setup: browser not ready; status=${JSON.stringify(lastStatus)} pageErrors=${JSON.stringify(pageErrors)}`);
  }
  routeIdentity.browser = await page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    return {
      navigatorGpu: Boolean(navigator.gpu),
      adapterInfo: adapter?.info ? { ...adapter.info } : null,
      deviceInitialized: Boolean(window._sf3d_device),
      weightsInitialized: Boolean(window._sf3d_weights),
      pipelinesInitialized: Boolean(window._sf3d_pipelines),
    };
  });

  const imageB64 = fs.readFileSync(IMAGE).toString('base64');
  await page.evaluate(async (b64) => { await new Promise((res, rej) => { const img = new Image(); img.onload = () => { window._img = img; res(); }; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; }); }, imageB64);

  const result = await page.evaluate(async (BATCH) => {
    try {
      const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
      const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines;
      const img = window._img;
      const toB64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };

      async function arm(opts) {
      // rAF gap probe scoped to the texture-bake window via stageSpans.
      const frames = []; let on = true, last = null;
      const tick = (t) => { if (last != null) frames.push({ start: last, end: t, gap: t - last }); last = t; if (on) requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
      const out = await runFullPipelineToGlb(device, pipelines, weights, img, opts);
      on = false; await new Promise(r => setTimeout(r, 60));
      const frameGapOverlaps = (frame, interval) => (
        interval
        && frame.start < (interval.endMs ?? interval.end)
        && frame.end > (interval.startMs ?? interval.start)
      );
      const maxGapOverlapping = interval => frames.reduce(
        (maxGap, frame) => frameGapOverlaps(frame, interval) ? Math.max(maxGap, frame.gap) : maxGap,
        0,
      );
      const span = (out.stageSpans || []).find(s => s.name === 'texture-bake');
      const bakeMaxGap = maxGapOverlapping(span);
      // hash the GLB (contains albedo+normal textures) for exact-output equality
      const glbHashSrc = new Uint8Array(out.glb);
      const coop = out.cooperativeReports?.['texture-bake'] || null;
      const telemetry = coop?.textureBakeTelemetry ?? null;
      const phases = telemetry?.phases ?? null;
      const materializationIntervals = phases?.materializationIntervals ?? {};
      const frameGapAttribution = {
        textureBake: bakeMaxGap,
        readback: maxGapOverlapping(phases?.readbackInterval),
        cpuMaterialization: maxGapOverlapping(phases?.cpuMaterializationInterval),
        albedo: maxGapOverlapping(materializationIntervals.albedo),
        normalDefault: maxGapOverlapping(materializationIntervals.normalDefault),
        normalOccupied: maxGapOverlapping(materializationIntervals.normalOccupied),
        albedoDilation: maxGapOverlapping(materializationIntervals.albedoDilation),
        normalDilation: maxGapOverlapping(materializationIntervals.normalDilation),
        ranges: (telemetry?.ranges ?? []).map(range => ({
          rangeIndex: range.rangeIndex,
          prepareEncode: maxGapOverlapping(range.prepareEncodeInterval),
          submit: maxGapOverlapping(range.submitInterval),
          browserYield: Math.max(
            0,
            ...(range.browserYieldIntervals ?? []).map(maxGapOverlapping),
          ),
        })),
        queueFences: (telemetry?.queueFences ?? []).map(fence => ({
          rangeIndex: fence.rangeIndex,
          queueWait: maxGapOverlapping(fence.queueInterval),
          retirement: maxGapOverlapping(fence.retirementInterval),
        })),
      };
      return {
        glbBytes: out.glb.byteLength, glbB64: toB64(glbHashSrc),
        bakeMaxGap, bakeStageMs: span ? +(span.end - span.start).toFixed(1) : null,
        frameGapAttribution,
        cooperative: coop ? {
          schedulingMode: coop.schedulingMode,
          status: coop.status,
          queueCompletionAuthority: coop.queueCompletionAuthority,
          rangeCount: coop.boundaries?.[0]?.actualRangeCount ?? null,
          completed: coop.progress?.completedItems,
          total: coop.progress?.totalItems,
          textureBakeTelemetry: coop.textureBakeTelemetry ?? null,
        } : null,
      };
      }

      const mono = await arm({ cooperativeDino: false });
      const off = await arm({ cooperativeDino: false, cooperativeBake: true, bakeSchedulingMode: 'disabled', bakeBatchTexels: BATCH });
      const on = await arm({ cooperativeDino: false, cooperativeBake: true, bakeSchedulingMode: 'cooperative', bakeBatchTexels: BATCH });
      return { mono, off, on };
    } catch (error) {
      return {
        assayFailure: {
          message: error?.message ?? String(error),
          stack: error?.stack ?? null,
          textureBakeCleanup: error?.textureBakeCleanup ?? null,
        },
      };
    }
  }, BATCH);

  if (result.assayFailure) {
    fail(result.assayFailure.message, 'browser-assay', result.assayFailure);
  }
  await browser.close().catch(() => {}); cleanup();
  if (pageErrors.length) fail(`page errors: ${pageErrors.join(' | ')}`);

  const hash = (b64) => crypto.createHash('sha256').update(Buffer.from(b64, 'base64')).digest('hex');
  const hMono = hash(result.mono.glbB64), hOff = hash(result.off.glbB64), hOn = hash(result.on.glbB64);
  const identical = hMono === hOff && hMono === hOn;

  const report = {
    ok: sourceAccepted
      && identical
      && result.on.cooperative?.queueCompletionAuthority === 'per-gpu-duty-prefix-fence'
      && result.on.cooperative?.completed === result.on.cooperative?.total,
    routeIdentity,
    evidenceAuthority: sourceIdentity.clean ? 'clean-commit' : 'explicit-dirty-diff',
    batchTexels: BATCH,
    outputIdentical: identical,
    glbSha256: { mono: hMono, off: hOff, on: hOn },
    textureBake: {
      monolithic: {
        maxGapMs: result.mono.bakeMaxGap,
        stageMs: result.mono.bakeStageMs,
        frameGapAttribution: result.mono.frameGapAttribution,
        cooperative: null,
      },
      coopOff: {
        maxGapMs: result.off.bakeMaxGap,
        stageMs: result.off.bakeStageMs,
        frameGapAttribution: result.off.frameGapAttribution,
        cooperative: result.off.cooperative,
      },
      coopOn: {
        maxGapMs: result.on.bakeMaxGap,
        stageMs: result.on.bakeStageMs,
        frameGapAttribution: result.on.frameGapAttribution,
        cooperative: result.on.cooperative,
      },
    },
  };
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));

  console.log('\n=== Cooperative texture-bake A/B ===');
  console.log(`output identical across all 3 arms: ${identical} (sha ${hMono.slice(0, 12)}…)`);
  console.log(`texture-bake max frame gap:  monolithic=${result.mono.bakeMaxGap.toFixed(1)}ms  coop-off=${result.off.bakeMaxGap.toFixed(1)}ms  coop-on=${result.on.bakeMaxGap.toFixed(1)}ms`);
  console.log(`texture-bake stage wall:     monolithic=${result.mono.bakeStageMs.toFixed(1)}ms  coop-off=${result.off.bakeStageMs.toFixed(1)}ms  coop-on=${result.on.bakeStageMs.toFixed(1)}ms`);
  console.log(`coop-on: ${result.on.cooperative?.completed}/${result.on.cooperative?.total} texels, ${result.on.cooperative?.rangeCount} batches, auth=${result.on.cooperative?.queueCompletionAuthority}`);
  if (!report.ok) fail(`assertions failed: identical=${identical} coop-on=${JSON.stringify(result.on.cooperative)}`);
  console.log(`\n✓ SMOKE PASSED — report: ${REPORT_PATH}`);
  process.exit(0);
})().catch(e => fail(e.message));
