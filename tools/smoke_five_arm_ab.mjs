#!/usr/bin/env node
/**
 * SF3D exact-source FIVE-ARM texture-bake A/B (Cranial arena assay contract #7 /
 * Wake required consumer evidence).
 *
 * One loaded model, five arms over the exact same route:
 *   1 monolithic         — legacy single decode dispatch + readback
 *   2 current-cooperative — cooperative batches, per-range transient scratch
 *   3 arena-only          — cooperative + decoder scratch arena (no worker mat)
 *   4 worker-only         — cooperative + worker materialization (no arena)
 *   5 arena-plus-worker   — cooperative + arena + worker materialization
 *
 * Records per arm: output GLB SHA-256 (must be identical across all five),
 * texture-bake stage wall + max foreground rAF gap, queue intervals, scratch
 * allocation count/bytes, worker transfer time. Also carries source identity
 * (commit, kit, input sha, weights sha, browser/adapter).
 *
 * Usage: node tools/smoke_five_arm_ab.mjs --expected-revision SHA [--image P] [--batch N]
 */
import puppeteer from 'puppeteer-core';
import path from 'node:path';
import fs from 'node:fs';
import net from 'node:net';
import crypto from 'node:crypto';
import { spawn, execFileSync } from 'node:child_process';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const argVal = (f, d) => { const i = process.argv.indexOf(f); return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d; };
const IMAGE = path.resolve(argVal('--image', path.join(process.env.HOME, '.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png')));
const BATCH = Number(argVal('--batch', '4096')) || 4096;
const REPORT_PATH = path.resolve(argVal('--report', '/tmp/sf3d-five-arm-ab.json'));
const EXPECTED_REVISION = argVal('--expected-revision', process.env.SF3D_EXPECTED_REVISION || '');
const ALLOW_DIRTY = process.argv.includes('--allow-dirty-source');

const allocatePort = () => new Promise((res, rej) => { const s = net.createServer(); s.unref(); s.on('error', rej); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); }); });
const sha256File = (p) => execFileSync('shasum', ['-a', '256', p], { encoding: 'utf8' }).trim().split(/\s+/)[0];
const procs = [];
const cleanup = () => { for (const p of procs) { try { p.kill(); } catch {} } };
let lastEvidence = {};
const fail = (m, phase = 'unknown') => { fs.writeFileSync(REPORT_PATH, JSON.stringify({ ok: false, failurePhase: phase, error: m, lastEvidence }, null, 2)); console.error(`\n✗ SMOKE FAILED [${phase}]: ${m}`); cleanup(); process.exit(1); };

(async () => {
  // Source identity gate.
  const effectiveRevision = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: REPO, encoding: 'utf8' }).trim();
  const dirty = execFileSync('git', ['status', '--short'], { cwd: REPO, encoding: 'utf8' }).trim();
  if (!EXPECTED_REVISION) fail('--expected-revision required', 'source-identity');
  if (effectiveRevision !== EXPECTED_REVISION) fail(`effective ${effectiveRevision} != requested ${EXPECTED_REVISION}`, 'source-identity');
  if (dirty && !ALLOW_DIRTY) fail(`worktree dirty:\n${dirty}`, 'source-identity');
  const weightsPath = path.join(REPO, 'public', 'weights.bin');
  if (!fs.existsSync(weightsPath)) fail('weights missing', 'weights');
  const source = { revision: effectiveRevision, clean: !dirty, inputSha: sha256File(IMAGE), weightsSha: sha256File(fs.realpathSync(weightsPath)), batch: BATCH };
  lastEvidence = { source };

  const port = await allocatePort();
  const vite = spawn('npx', ['vite', '--port', String(port), '--strictPort'], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(vite);
  await new Promise((res, rej) => { const to = setTimeout(() => rej(new Error('vite timeout')), 40000); vite.stdout.on('data', d => { if (/Local:|ready/.test(d.toString())) { clearTimeout(to); res(); } }); vite.on('error', e => { clearTimeout(to); rej(e); }); }).catch(e => fail(e.message, 'vite'));

  const browser = await puppeteer.launch({ executablePath: CHROME_PATH, headless: false, protocolTimeout: 1200000, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
  const page = await browser.newPage();
  const pageErrors = []; page.on('pageerror', e => pageErrors.push(e.message));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded', timeout: 30000 });
  const start = Date.now();
  while (Date.now() - start < 180000) { const s = await page.$eval('#status', el => el.textContent).catch(() => ''); if (s.includes('Ready')) break; await new Promise(r => setTimeout(r, 500)); }
  source.backend = await page.evaluate(async () => { const a = await navigator.gpu?.requestAdapter(); const info = a?.info; return { ua: navigator.userAgent, vendor: info?.vendor, arch: info?.architecture }; });
  lastEvidence = { source };

  const imageB64 = fs.readFileSync(IMAGE).toString('base64');
  await page.evaluate(async (b64) => { await new Promise((res, rej) => { const img = new Image(); img.onload = () => { window._img = img; res(); }; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; }); }, imageB64);

  const arms = await page.evaluate(async (BATCH) => {
    const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
    const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines, img = window._img;
    const toB64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
    let matWorker = null;
    const getMatWorker = () => (matWorker ||= new Worker(new URL('/src/lib/materialize_worker.js', location.origin), { type: 'module' }));

    async function arm(name, opts) {
      const frames = []; let on = true, last = null;
      const tick = (t) => { if (last != null) frames.push({ start: last, end: t, gap: t - last }); last = t; if (on) requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
      const out = await runFullPipelineToGlb(device, pipelines, weights, img, opts);
      on = false; await new Promise(r => setTimeout(r, 60));
      const span = (out.stageSpans || []).find(s => s.name === 'texture-bake');
      let bakeMaxGap = 0; if (span) for (const f of frames) { const mid = (f.start + f.end) / 2; if (mid >= span.start && mid < span.end) bakeMaxGap = Math.max(bakeMaxGap, f.gap); }
      const bake = out.cooperativeReports?.['texture-bake'] || null;
      const tel = bake?.textureBakeTelemetry || null;
      return {
        name, glbB64: toB64(new Uint8Array(out.glb)), glbBytes: out.glb.byteLength,
        bakeStageMs: span ? +(span.end - span.start).toFixed(1) : null,
        bakeMaxGapMs: +bakeMaxGap.toFixed(1),
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
    const results = [];
    results.push(await arm('monolithic', { ...base }));
    results.push(await arm('current-cooperative', { ...coop }));
    results.push(await arm('arena-only', { ...coop, decoderArena: true }));
    results.push(await arm('worker-only', { ...coop, materializeWorker: getMatWorker() }));
    results.push(await arm('arena-plus-worker', { ...coop, decoderArena: true, materializeWorker: getMatWorker() }));
    if (matWorker) matWorker.terminate();
    return results;
  }, BATCH);

  await browser.close().catch(() => {}); cleanup();
  if (pageErrors.length) fail(`page errors: ${pageErrors.join(' | ')}`, 'browser');

  const hash = (b64) => crypto.createHash('sha256').update(Buffer.from(b64, 'base64')).digest('hex');
  const withSha = arms.map(a => ({ ...a, glbSha: hash(a.glbB64), glbB64: undefined }));
  const shas = new Set(withSha.map(a => a.glbSha));
  const identical = shas.size === 1;

  // Acceptance: arena+worker (the winning arm) must (a) collapse scratch vs the
  // current-cooperative churn, and (b) collapse the foreground gap vs monolithic.
  const byName = Object.fromEntries(withSha.map(a => [a.name, a]));
  const cur = byName['current-cooperative'], mono = byName['monolithic'], best = byName['arena-plus-worker'];
  const scratchCollapsed = best.scratchAllocatedBytes != null && cur.scratchAllocatedBytes != null
    && best.scratchAllocatedBytes < cur.scratchAllocatedBytes * 0.2;
  const gapCollapsed = best.bakeMaxGapMs != null && mono.bakeMaxGapMs != null
    && best.bakeMaxGapMs < mono.bakeMaxGapMs * 0.5;

  const report = {
    ok: identical && scratchCollapsed && gapCollapsed,
    source,
    outputIdentical: identical,
    canonicalGlbSha: withSha[0].glbSha,
    distinctShas: [...shas],
    winning: {
      arm: 'arena-plus-worker',
      bakeMaxGapMs: best.bakeMaxGapMs, scratchAllocatedBytes: best.scratchAllocatedBytes,
      scratchCollapsedVsCurrent: scratchCollapsed, gapCollapsedVsMonolithic: gapCollapsed,
    },
    arms: withSha,
  };
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));

  console.log('\n=== SF3D five-arm texture-bake A/B ===');
  console.log(`source ${source.revision.slice(0, 10)} clean=${source.clean} ${source.backend.vendor}/${source.backend.arch} batch=${BATCH}`);
  console.log(`GLB identical across all 5 arms: ${identical} (sha ${withSha[0].glbSha.slice(0, 12)}…)`);
  console.log('arm'.padEnd(20), 'bakeWall'.padStart(9), 'bakeMaxGap'.padStart(11), 'scratchMB'.padStart(10), 'cpuMatMs'.padStart(9), 'xferMs'.padStart(7));
  for (const a of withSha) {
    console.log(
      a.name.padEnd(20),
      String(a.bakeStageMs).padStart(9),
      String(a.bakeMaxGapMs).padStart(11),
      String(a.scratchAllocatedBytes != null ? (a.scratchAllocatedBytes / 1e6).toFixed(1) : 'n/a').padStart(10),
      String(a.cpuMaterializationMs != null ? a.cpuMaterializationMs.toFixed(0) : 'n/a').padStart(9),
      String(a.workerTransferMs != null ? a.workerTransferMs.toFixed(0) : 'n/a').padStart(7),
    );
  }
  if (!identical) fail(`GLB DIFFERS across arms: ${[...shas].map(s => s.slice(0, 10)).join(', ')}`, 'output-identity');
  console.log(`\narena+worker: scratchCollapsed=${scratchCollapsed} gapCollapsed=${gapCollapsed}`);
  if (!scratchCollapsed) fail(`arena+worker did not collapse scratch: ${best.scratchAllocatedBytes} vs current ${cur.scratchAllocatedBytes}`, 'acceptance');
  if (!gapCollapsed) fail(`arena+worker did not collapse gap: ${best.bakeMaxGapMs}ms vs monolithic ${mono.bakeMaxGapMs}ms`, 'acceptance');
  console.log(`✓ SMOKE PASSED — five arms byte-identical; arena+worker collapses scratch AND foreground gap. report: ${REPORT_PATH}`);
  process.exit(0);
})().catch(e => fail(e.message, 'exception'));
