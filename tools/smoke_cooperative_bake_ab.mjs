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
 * Usage: node tools/smoke_cooperative_bake_ab.mjs [--image PATH] [--batch N]
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
const WEIGHT_PATH = path.join(REPO, 'public', 'weights.bin');
const sourceIdentity = {
  revision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: REPO, encoding: 'utf8' }).trim(),
  dirtyPaths: execFileSync('git', ['status', '--short'], { cwd: REPO, encoding: 'utf8' })
    .trim().split('\n').filter(Boolean),
  worktree: REPO,
};

const allocatePort = () => new Promise((res, rej) => { const s = net.createServer(); s.unref(); s.on('error', rej); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); }); });
const procs = [];
const cleanup = () => { for (const p of procs) { try { p.kill(); } catch {} } };
const fail = (m) => { fs.writeFileSync(REPORT_PATH, JSON.stringify({ ok: false, error: m }, null, 2)); console.error(`\n✗ SMOKE FAILED: ${m}`); cleanup(); process.exit(1); };

(async () => {
  if (!fs.existsSync(WEIGHT_PATH)) fail(`setup: missing model weights at ${WEIGHT_PATH}`);
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
      // texture-bake gap = max gap whose midpoint is in the texture-bake span
      const span = (out.stageSpans || []).find(s => s.name === 'texture-bake');
      let bakeMaxGap = 0;
      if (span) for (const f of frames) { const mid = (f.start + f.end) / 2; if (mid >= span.start && mid < span.end) bakeMaxGap = Math.max(bakeMaxGap, f.gap); }
      // hash the GLB (contains albedo+normal textures) for exact-output equality
      const glbHashSrc = new Uint8Array(out.glb);
      const coop = out.cooperativeReports?.['texture-bake'] || null;
      return {
        glbBytes: out.glb.byteLength, glbB64: toB64(glbHashSrc),
        bakeMaxGap, bakeStageMs: span ? +(span.end - span.start).toFixed(1) : null,
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
  }, BATCH);

  await browser.close().catch(() => {}); cleanup();
  if (pageErrors.length) fail(`page errors: ${pageErrors.join(' | ')}`);

  const hash = (b64) => crypto.createHash('sha256').update(Buffer.from(b64, 'base64')).digest('hex');
  const hMono = hash(result.mono.glbB64), hOff = hash(result.off.glbB64), hOn = hash(result.on.glbB64);
  const identical = hMono === hOff && hMono === hOn;

  const report = {
    ok: identical && result.on.cooperative?.queueCompletionAuthority === 'per-gpu-duty-prefix-fence' && result.on.cooperative?.completed === result.on.cooperative?.total,
    routeIdentity,
    batchTexels: BATCH,
    outputIdentical: identical,
    glbSha256: { mono: hMono, off: hOff, on: hOn },
    textureBake: {
      monolithic: { maxGapMs: result.mono.bakeMaxGap, stageMs: result.mono.bakeStageMs, cooperative: null },
      coopOff: { maxGapMs: result.off.bakeMaxGap, stageMs: result.off.bakeStageMs, cooperative: result.off.cooperative },
      coopOn: { maxGapMs: result.on.bakeMaxGap, stageMs: result.on.bakeStageMs, cooperative: result.on.cooperative },
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
