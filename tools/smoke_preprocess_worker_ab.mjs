#!/usr/bin/env node
/**
 * Image-preprocess worker-offload A/B smoke.
 *
 * Profiling flagged image-preprocess (Lanczos-3 resize, CPU) as the largest
 * foreground gap (~700ms) with zero GPU sync — so a Web Worker offload has no
 * per-duty fence floor (unlike cooperative GPU boundaries). This proves the
 * offload preserves exact output AND collapses the main-thread gap.
 *
 * Two arms over one loaded model:
 *   main-thread — preprocess on the main thread (control)
 *   worker      — preprocess on a Web Worker (candidate)
 * Asserts: CHW tensor + final GLB byte-identical; measures image-preprocess-
 * window rAF max gap per arm (expect worker arm's main-thread gap near zero).
 *
 * Usage: node tools/smoke_preprocess_worker_ab.mjs [--image PATH]
 */
import puppeteer from 'puppeteer-core';
import path from 'node:path';
import fs from 'node:fs';
import net from 'node:net';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const argVal = (f, d) => { const i = process.argv.indexOf(f); return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d; };
const IMAGE = path.resolve(argVal('--image', path.join(process.env.HOME, '.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png')));
const REPORT_PATH = path.resolve(argVal('--report', '/tmp/sf3d-preprocess-worker-ab.json'));

const allocatePort = () => new Promise((res, rej) => { const s = net.createServer(); s.unref(); s.on('error', rej); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); }); });
const procs = [];
const cleanup = () => { for (const p of procs) { try { p.kill(); } catch {} } };
const fail = (m) => { fs.writeFileSync(REPORT_PATH, JSON.stringify({ ok: false, error: m }, null, 2)); console.error(`\n✗ SMOKE FAILED: ${m}`); cleanup(); process.exit(1); };

(async () => {
  const port = await allocatePort();
  const vite = spawn('npx', ['vite', '--port', String(port), '--strictPort'], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(vite);
  await new Promise((res, rej) => { const to = setTimeout(() => rej(new Error('vite timeout')), 40000); vite.stdout.on('data', d => { if (/Local:|ready/.test(d.toString())) { clearTimeout(to); res(); } }); vite.on('error', e => { clearTimeout(to); rej(e); }); }).catch(e => fail(e.message));

  const browser = await puppeteer.launch({ executablePath: CHROME_PATH, headless: false, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
  const page = await browser.newPage();
  const pageErrors = []; page.on('pageerror', e => pageErrors.push(e.message));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded', timeout: 30000 });
  const start = Date.now();
  while (Date.now() - start < 180000) { const s = await page.$eval('#status', el => el.textContent).catch(() => ''); if (s.includes('Ready')) break; await new Promise(r => setTimeout(r, 500)); }

  const imageB64 = fs.readFileSync(IMAGE).toString('base64');
  await page.evaluate(async (b64) => { await new Promise((res, rej) => { const img = new Image(); img.onload = () => { window._img = img; res(); }; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; }); }, imageB64);

  const result = await page.evaluate(async () => {
    const { runFullPipelineToGlb, preprocessImage } = await import('/src/lib/full_pipeline.js').then(async m => ({
      runFullPipelineToGlb: m.runFullPipelineToGlb,
      preprocessImage: (await import('/src/lib/inference.js')).preprocessImage,
    }));
    const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines;
    const img = window._img;
    const toB64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
    const sha = async (buf) => { const d = await crypto.subtle.digest('SHA-256', buf); return [...new Uint8Array(d)].map(b => b.toString(16).padStart(2, '0')).join(''); };

    // Create the preprocess worker (Vite module worker).
    const worker = new Worker(new URL('/src/lib/preprocess_worker.js', location.origin), { type: 'module' });

    // First: direct byte-identity check of preprocessImage output (main vs worker).
    const chwMain = await preprocessImage(img, img.naturalWidth, img.naturalHeight);
    const chwWorker = await preprocessImage(img, img.naturalWidth, img.naturalHeight, { preprocessWorker: worker });
    const chwMainSha = await sha(chwMain.buffer.slice(0));
    const chwWorkerSha = await sha(chwWorker.buffer.slice(0));

    // Then: full-pipeline A/B with rAF gap attribution to the image-preprocess span.
    async function arm(opts) {
      const frames = []; let on = true, last = null;
      const tick = (t) => { if (last != null) frames.push({ start: last, end: t, gap: t - last }); last = t; if (on) requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
      const out = await runFullPipelineToGlb(device, pipelines, weights, img, opts);
      on = false; await new Promise(r => setTimeout(r, 60));
      const span = (out.stageSpans || []).find(s => s.name === 'image-preprocess');
      let ppMaxGap = 0;
      if (span) for (const f of frames) { const mid = (f.start + f.end) / 2; if (mid >= span.start && mid < span.end) ppMaxGap = Math.max(ppMaxGap, f.gap); }
      return { glbB64: toB64(new Uint8Array(out.glb)), glbBytes: out.glb.byteLength, ppMaxGap, ppStageMs: span ? +(span.end - span.start).toFixed(1) : null };
    }
    const mainArm = await arm({ cooperativeDino: false });
    const workerArm = await arm({ cooperativeDino: false, preprocessWorker: worker });
    worker.terminate();
    return { chwMainSha, chwWorkerSha, chwLen: chwMain.length, mainArm, workerArm };
  });

  await browser.close().catch(() => {}); cleanup();
  if (pageErrors.length) fail(`page errors: ${pageErrors.join(' | ')}`);

  const hash = (b64) => crypto.createHash('sha256').update(Buffer.from(b64, 'base64')).digest('hex');
  const glbMain = hash(result.mainArm.glbB64), glbWorker = hash(result.workerArm.glbB64);
  const chwIdentical = result.chwMainSha === result.chwWorkerSha;
  const glbIdentical = glbMain === glbWorker;

  const report = {
    ok: chwIdentical && glbIdentical && result.workerArm.ppMaxGap < result.mainArm.ppMaxGap,
    chwIdentical, glbIdentical,
    chwSha: result.chwMainSha, glbSha: glbMain,
    imagePreprocess: {
      mainThread: { maxGapMs: result.mainArm.ppMaxGap, stageMs: result.mainArm.ppStageMs },
      worker: { maxGapMs: result.workerArm.ppMaxGap, stageMs: result.workerArm.ppStageMs },
    },
  };
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));

  console.log('\n=== Image-preprocess worker-offload A/B ===');
  console.log(`CHW tensor identical (main vs worker): ${chwIdentical} (sha ${result.chwMainSha.slice(0, 12)}…)`);
  console.log(`final GLB identical:                   ${glbIdentical} (sha ${glbMain.slice(0, 12)}…)`);
  console.log(`image-preprocess main-thread max gap:  main=${result.mainArm.ppMaxGap.toFixed(1)}ms  worker=${result.workerArm.ppMaxGap.toFixed(1)}ms`);
  if (!report.ok) fail(`assertions failed: chwIdentical=${chwIdentical} glbIdentical=${glbIdentical} workerGap=${result.workerArm.ppMaxGap} mainGap=${result.mainArm.ppMaxGap}`);
  console.log(`\n✓ SMOKE PASSED — worker offload preserves output and reduces the main-thread gap. report: ${REPORT_PATH}`);
  process.exit(0);
})().catch(e => fail(e.message));
