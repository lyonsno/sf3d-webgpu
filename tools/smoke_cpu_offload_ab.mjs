#!/usr/bin/env node
/**
 * CPU worker-offload A/B smoke — image-preprocess + uv-unwrap.
 *
 * Offloads the two largest CPU foreground gaps profiling found (image-preprocess
 * ~700ms, uv-unwrap ~216ms) to Web Workers and proves byte-identical output plus
 * aggregate main-thread gap collapse. Unlike the fence-bound GPU cooperative
 * boundary, CPU offload has no per-duty fence floor.
 *
 * Two arms over one loaded model:
 *   baseline — both CPU stages on the main thread (control)
 *   offload  — both on Web Workers (candidate)
 * Asserts: final GLB byte-identical; measures per-stage main-thread max gap.
 *
 * Usage: node tools/smoke_cpu_offload_ab.mjs [--image PATH]
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
const REPORT_PATH = path.resolve(argVal('--report', '/tmp/sf3d-cpu-offload-ab.json'));

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
    const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
    const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines;
    const img = window._img;
    const toB64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
    const ppWorker = new Worker(new URL('/src/lib/preprocess_worker.js', location.origin), { type: 'module' });
    const uvWorker = new Worker(new URL('/src/lib/uv_unwrap_worker.js', location.origin), { type: 'module' });

    async function arm(opts) {
      const frames = []; let on = true, last = null;
      const tick = (t) => { if (last != null) frames.push({ start: last, end: t, gap: t - last }); last = t; if (on) requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
      const out = await runFullPipelineToGlb(device, pipelines, weights, img, opts);
      on = false; await new Promise(r => setTimeout(r, 60));
      const gapFor = (name) => { const s = (out.stageSpans || []).find(x => x.name === name); let m = 0; if (s) for (const f of frames) { const mid = (f.start + f.end) / 2; if (mid >= s.start && mid < s.end) m = Math.max(m, f.gap); } return { maxGap: m, stageMs: s ? +(s.end - s.start).toFixed(1) : null }; };
      const allGaps = frames.map(f => f.gap).sort((a, b) => a - b);
      return { glbB64: toB64(new Uint8Array(out.glb)), preprocess: gapFor('image-preprocess'), uvUnwrap: gapFor('uv-unwrap'), overallMax: allGaps.at(-1) || 0 };
    }

    const baseline = await arm({ cooperativeDino: false });
    const offload = await arm({ cooperativeDino: false, preprocessWorker: ppWorker, uvUnwrapWorker: uvWorker });
    ppWorker.terminate(); uvWorker.terminate();
    return { baseline, offload };
  });

  await browser.close().catch(() => {}); cleanup();
  if (pageErrors.length) fail(`page errors: ${pageErrors.join(' | ')}`);

  const hash = (b64) => crypto.createHash('sha256').update(Buffer.from(b64, 'base64')).digest('hex');
  const gB = hash(result.baseline.glbB64), gO = hash(result.offload.glbB64);
  const identical = gB === gO;
  const b = result.baseline, o = result.offload;

  const report = {
    ok: identical && o.preprocess.maxGap < b.preprocess.maxGap && o.uvUnwrap.maxGap <= b.uvUnwrap.maxGap,
    glbIdentical: identical, glbSha: gB,
    imagePreprocess: { baselineMaxGapMs: b.preprocess.maxGap, offloadMaxGapMs: o.preprocess.maxGap },
    uvUnwrap: { baselineMaxGapMs: b.uvUnwrap.maxGap, offloadMaxGapMs: o.uvUnwrap.maxGap },
    overallMaxGapMs: { baseline: b.overallMax, offload: o.overallMax },
  };
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));

  console.log('\n=== CPU worker-offload A/B (image-preprocess + uv-unwrap) ===');
  console.log(`final GLB identical: ${identical} (sha ${gB.slice(0, 12)}…)`);
  console.log(`image-preprocess max gap:  baseline=${b.preprocess.maxGap.toFixed(1)}ms  offload=${o.preprocess.maxGap.toFixed(1)}ms`);
  console.log(`uv-unwrap max gap:         baseline=${b.uvUnwrap.maxGap.toFixed(1)}ms  offload=${o.uvUnwrap.maxGap.toFixed(1)}ms`);
  console.log(`overall pipeline max gap:  baseline=${b.overallMax.toFixed(1)}ms  offload=${o.overallMax.toFixed(1)}ms`);
  if (!report.ok) fail(`assertions failed: identical=${identical} pp ${b.preprocess.maxGap}->${o.preprocess.maxGap} uv ${b.uvUnwrap.maxGap}->${o.uvUnwrap.maxGap}`);
  console.log(`\n✓ SMOKE PASSED — both CPU stages offloaded, output preserved. report: ${REPORT_PATH}`);
  process.exit(0);
})().catch(e => fail(e.message));
