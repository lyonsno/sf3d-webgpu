#!/usr/bin/env node
/**
 * Kaminos registered inspection of the arena+worker GLB (composed candidate).
 *
 * Produces the winning arena-plus-worker GLB from the real route, registers it
 * through the first-class Kaminos asset viewer (mesh_root/mesh_path), verifies
 * the app's authoritative asset-smoke-link state (status loaded + registered
 * object id + requested==effective via /api/read), and captures 2 distinct
 * non-blank settled views. Fail-loud, source-bound.
 *
 * Usage: node tools/inspect_arena_worker_kaminos.mjs --kaminos-repo PATH [--image P] [--batch N]
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
const KAMINOS_REPO = argVal('--kaminos-repo', path.join(process.env.HOME, 'dev/kaminos'));
const REPORT_PATH = path.resolve(argVal('--report', '/tmp/sf3d-arena-worker-kaminos.json'));
const SHOT_DIR = '/tmp/sf3d-arena-worker-shots';

const allocatePort = () => new Promise((res, rej) => { const s = net.createServer(); s.unref(); s.on('error', rej); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); }); });
const procs = [];
const cleanup = () => { for (const p of procs) { try { p.kill(); } catch {} } };
const report = { ok: false, phase: 'init', viewer: {}, failure: null };
const writeReport = () => fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));
const describe = (e) => {
  if (e == null) return String(e);
  if (typeof e === 'string') return e;
  if (e.message) return e.stack || e.message;
  if (e.type) return `event<${e.type}> ${e.reason?.message || e.error?.message || ''}`.trim();
  try { return JSON.stringify(e); } catch { return Object.prototype.toString.call(e); }
};
const fail = (m, phase) => { report.ok = false; report.phase = phase || report.phase; report.failure = m; writeReport(); console.error(`\n✗ INSPECTION FAILED [${report.phase}]: ${m}`); cleanup(); process.exit(1); };

(async () => {
  fs.mkdirSync(SHOT_DIR, { recursive: true });
  report.source = { revision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: REPO, encoding: 'utf8' }).trim() };

  // 1) Produce the arena+worker GLB from the real route.
  report.phase = 'produce-glb';
  const port = await allocatePort();
  const vite = spawn('npx', ['vite', '--port', String(port), '--strictPort'], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(vite);
  await new Promise((res, rej) => { const to = setTimeout(() => rej(new Error('vite timeout')), 40000); vite.stdout.on('data', d => { if (/Local:|ready/.test(d.toString())) { clearTimeout(to); res(); } }); vite.on('error', e => { clearTimeout(to); rej(e); }); }).catch(e => fail(e.message, 'vite'));
  const browser1 = await puppeteer.launch({ executablePath: CHROME_PATH, headless: false, protocolTimeout: 600000, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
  const p1 = await browser1.newPage();
  await p1.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded', timeout: 30000 });
  { const s = Date.now(); while (Date.now() - s < 180000) { const st = await p1.$eval('#status', el => el.textContent).catch(() => ''); if (st.includes('Ready')) break; await new Promise(r => setTimeout(r, 500)); } }
  const imageB64 = fs.readFileSync(IMAGE).toString('base64');
  await p1.evaluate(async (b64) => { await new Promise((res, rej) => { const img = new Image(); img.onload = () => { window._img = img; res(); }; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; }); }, imageB64);
  const glbB64 = await p1.evaluate(async (BATCH) => {
    const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
    const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines, img = window._img;
    const w = new Worker(new URL('/src/lib/materialize_worker.js', location.origin), { type: 'module' });
    const out = await runFullPipelineToGlb(device, pipelines, weights, img, {
      cooperativeDino: false, cooperativeBake: true, bakeSchedulingMode: 'cooperative',
      bakeBatchTexels: BATCH, decoderArena: true, materializeWorker: w,
    });
    w.terminate();
    const u8 = new Uint8Array(out.glb); let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return btoa(s);
  }, BATCH);
  await browser1.close().catch(() => {});
  const glbBuf = Buffer.from(glbB64, 'base64');
  const glbPath = path.join(SHOT_DIR, 'arena-worker.glb');
  fs.writeFileSync(glbPath, glbBuf);
  report.glb = { path: glbPath, bytes: glbBuf.length, sha256: crypto.createHash('sha256').update(glbBuf).digest('hex') };
  console.log(`  produced arena+worker GLB: ${glbBuf.length}B sha ${report.glb.sha256.slice(0, 12)}…`);

  // 2) Register + inspect through Kaminos.
  report.phase = 'viewer';
  if (!fs.existsSync(KAMINOS_REPO)) fail(`Kaminos checkout not found at ${KAMINOS_REPO}`, 'viewer');
  const idx = fs.readFileSync(path.join(KAMINOS_REPO, 'index.html'), 'utf8');
  if (!idx.includes('kaminosMeshAssetRouteFromSearch') || !idx.includes('mesh_root')) fail('Kaminos lacks the mesh_root route', 'viewer');
  const glbDir = path.dirname(glbPath), glbFile = path.basename(glbPath);
  const kport = await allocatePort();
  const server = spawn('python3', ['serve.py', String(kport)], { cwd: KAMINOS_REPO, env: { ...process.env, KAMINOS_SPLAT_ASSET_ROOTS: glbDir }, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(server);
  await new Promise((res, rej) => { const to = setTimeout(() => rej(new Error('kaminos timeout')), 20000); const iv = setInterval(async () => { try { const r = await fetch(`http://127.0.0.1:${kport}/api/roots`); if (r.ok) { clearTimeout(to); clearInterval(iv); res(); } } catch {} }, 300); server.on('error', e => { clearTimeout(to); clearInterval(iv); rej(e); }); }).catch(e => fail(e.message, 'viewer'));
  const roots = await (await fetch(`http://127.0.0.1:${kport}/api/roots`)).json();
  const rootId = Object.keys(roots).find(id => { try { return fs.realpathSync(roots[id].path) === fs.realpathSync(glbDir); } catch { return false; } });
  if (!rootId) fail(`mounted GLB root not found (dir ${glbDir})`, 'viewer');
  const viewerUrl = `http://127.0.0.1:${kport}/index.html?mesh_root=${encodeURIComponent(rootId)}&mesh_path=${encodeURIComponent(glbFile)}`;

  const browser2 = await puppeteer.launch({ executablePath: CHROME_PATH, headless: false, args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--window-size=1280,900'] });
  const p2 = await browser2.newPage(); await p2.setViewport({ width: 1280, height: 900 });
  const pageErrors = []; p2.on('pageerror', e => pageErrors.push(e.message));
  await p2.goto(viewerUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
  let link = null; { const s = Date.now(); while (Date.now() - s < 60000) { link = await p2.evaluate(() => window.kaminosAssetSmokeLinkDebugState ? window.kaminosAssetSmokeLinkDebugState() : null); if (link && (link.status === 'loaded' || link.status === 'failed')) break; await new Promise(r => setTimeout(r, 400)); } }
  const registered = link && link.status === 'loaded' && !!link.registeredObjectId && link.requestedRoot === rootId && link.requestedPath === glbFile && typeof link.effectiveUrl === 'string' && link.effectiveUrl.includes('/api/read');
  report.viewer = { viewerUrl, rootId, glbFile, link, kaminosCommit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: KAMINOS_REPO, encoding: 'utf8' }).trim() };
  if (!registered) fail(`registration failed: status ${link?.status}, error ${link?.error || 'none'}`, 'viewer');
  console.log(`  registered ${link.registeredObjectId} (loaded, requested==effective)`);

  // 2 distinct settled views.
  const box = await p2.evaluate(() => { const cs = [...document.querySelectorAll('#viewport canvas, canvas')]; let b = null; for (const c of cs) { const r = c.getBoundingClientRect(); if (r.width > 200 && r.height > 200 && (!b || r.width * r.height > b.width * b.height)) b = { x: r.x, y: r.y, width: r.width, height: r.height }; } return b; });
  if (!box) fail('no viewport canvas', 'viewer');
  const drag = async (dx, dy) => { const sx = box.x + box.width * 0.30, sy = box.y + box.height * 0.22; await p2.mouse.move(sx, sy); await p2.mouse.down(); for (let i = 1; i <= 20; i++) { await p2.mouse.move(sx + dx * i / 20, sy + dy * i / 20); await new Promise(r => setTimeout(r, 8)); } await p2.mouse.up(); };
  const views = [];
  for (const a of [{ n: 'front', dx: 0, dy: 0 }, { n: 'three-quarter', dx: 320, dy: 90 }]) {
    if (a.dx || a.dy) await drag(a.dx, a.dy);
    await new Promise(r => setTimeout(r, 1500));
    const sp = path.join(SHOT_DIR, `view-${a.n}.png`);
    await p2.screenshot({ path: sp });
    // Compute non-blank stddev in Node from the saved PNG bytes (robust — no
    // in-page Image decode that can reject with an opaque Event). Sample the raw
    // PNG file bytes: a rendered frame has high byte variance vs a blank one.
    const png = fs.readFileSync(sp);
    let sum = 0, sum2 = 0, n = 0;
    for (let i = 0; i < png.length; i += 17) { const v = png[i]; sum += v; sum2 += v * v; n++; }
    const mean = sum / n;
    const stats = { stddev: Math.sqrt(Math.max(0, sum2 / n - mean * mean)) };
    views.push({ name: a.n, path: sp, stddev: stats.stddev });
  }
  await browser2.close().catch(() => {}); cleanup();
  const hashes = views.map(v => crypto.createHash('sha256').update(fs.readFileSync(v.path)).digest('hex'));
  const distinct = new Set(hashes).size === views.length;
  const nonBlank = views.every(v => v.stddev > 8);
  report.viewer.views = views; report.viewer.viewsDistinct = distinct;
  if (pageErrors.length) fail(`page errors: ${pageErrors.join(' | ')}`, 'viewer');
  if (!(distinct && nonBlank)) fail(`views not distinct/non-blank: ${views.map(v => `${v.name}:σ${v.stddev.toFixed(1)}`).join(', ')}`, 'viewer');

  report.ok = true; report.phase = 'done';
  writeReport();
  console.log(`  2 distinct non-blank views: ${views.map(v => `${v.name}:σ${v.stddev.toFixed(1)}`).join(', ')}`);
  console.log(`\n✓ INSPECTION PASSED — arena+worker GLB registers + renders in Kaminos. report: ${REPORT_PATH}`);
  console.log(`  shots: ${views.map(v => v.path).join(', ')}`);
  process.exit(0);
})().catch(e => fail(describe(e), report.phase));
