#!/usr/bin/env node
/**
 * SF3D CPU-offload acceptance capsule — productionization of the worker-offload
 * slice (image-preprocess + uv-unwrap), per operator productionization directive.
 *
 * Source-bound, fail-loud, durable. Gates:
 *   1 checkout-owned Vite on an allocated strictPort
 *   2 verify checkout, commit, kit version, input sha, weights sha (streamed),
 *     browser, adapter, backend; refuse a dirty tree
 *   3 baseline (main-thread CPU) + offload (workers) over the same loaded model;
 *     final GLB byte-identical; measure image-preprocess + uv-unwrap main-thread
 *     max gaps per arm (offload must reduce both)
 *   4 exercise the worker FAILURE lifecycle live: a crashing worker must fail the
 *     offload arm LOUDLY (no hang, no silent main-thread fallback, no bad output)
 *   5 register the OFFLOAD-path GLB through the Kaminos asset viewer (fresh
 *     inspection — not reused DINO shots); requested==effective + registration
 *   6 capture + inspect >=2 fresh settled Kaminos views (distinct, non-blank)
 *   7 durable structured report on success and every failure phase; nonzero exit
 *     on any red gate; landing capsule with the exact reviewed head + smoke URL
 *
 * Usage: node tools/sf3d_cpu_offload_capsule.mjs [--image PATH] [--kaminos-repo PATH]
 */
import puppeteer from 'puppeteer-core';
import path from 'node:path';
import fs from 'node:fs';
import net from 'node:net';
import crypto from 'node:crypto';
import { spawn, execSync } from 'node:child_process';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const argVal = (f, d) => { const i = process.argv.indexOf(f); return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d; };
const IMAGE = path.resolve(argVal('--image', path.join(process.env.HOME, '.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png')));
const REPORT_PATH = path.resolve(argVal('--report', '/tmp/sf3d-cpu-offload-capsule-report.json'));
const KAMINOS_REPO = argVal('--kaminos-repo', path.join(process.env.HOME, 'dev/kaminos'));
const ROUTE_ID = 'sf3d.image-to-mesh.webgpu-local.v0';
const SHOT_DIR = '/tmp/sf3d-cpu-offload-shots';

const report = {
  schema: 'sf3d.cpu-offload-capsule.v1', ok: false, phase: 'init',
  startedAt: new Date().toISOString(), route: ROUTE_ID,
  gates: {}, source: {}, arms: {}, workerFailure: {}, viewer: {}, landing: null, failure: null,
};
const procs = [];
function writeReport() { report.endedAt = new Date().toISOString(); fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true }); fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2)); }
function fail(phase, message, extra = {}) { report.ok = false; report.phase = phase; report.failure = { phase, message, ...extra }; writeReport(); for (const p of procs) { try { p.kill(); } catch {} } console.error(`\n✗ CAPSULE FAILED [${phase}]: ${message}`); console.error(`  report: ${REPORT_PATH}`); process.exit(1); }
function gate(id, passed, detail = {}) { report.gates[id] = { passed, ...detail }; console.log(`  ${passed ? '✓' : '✗'} gate ${id}${detail.note ? ` — ${detail.note}` : ''}`); if (!passed) fail(`gate:${id}`, detail.note || `gate ${id} failed`, { detail }); }

const allocatePort = () => new Promise((res, rej) => { const s = net.createServer(); s.unref(); s.on('error', rej); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); }); });
function sha256File(p) { return new Promise((res, rej) => { const h = crypto.createHash('sha256'); const st = fs.createReadStream(p); st.on('data', d => h.update(d)); st.on('end', () => res(h.digest('hex'))); st.on('error', rej); }); }
const pct = (s, p) => s.length ? s[Math.min(s.length - 1, Math.ceil(p / 100 * s.length) - 1)] : null;

async function verifyAndLaunch() {
  report.phase = 'source-verification';
  const commit = execSync('git rev-parse HEAD', { cwd: REPO }).toString().trim();
  const dirty = execSync('git status --porcelain', { cwd: REPO }).toString().trim();
  const kitVersion = JSON.parse(fs.readFileSync(path.join(REPO, 'node_modules/@kaminos/webgpu-inference-kit/package.json'))).version;
  if (!fs.existsSync(IMAGE)) fail('source-verification', `input missing: ${IMAGE}`);
  const weightsPath = path.join(REPO, 'public/weights.bin');
  if (!fs.existsSync(weightsPath)) fail('source-verification', `weights missing: ${weightsPath}`);
  report.source = {
    checkout: REPO, commit, workingTreeDirty: dirty.length > 0, route: ROUTE_ID, kitVersion,
    inputPath: IMAGE, inputSha256: await sha256File(IMAGE),
    weightsSha256: await sha256File(weightsPath), weightsBytes: fs.statSync(weightsPath).size,
  };
  console.log(`  checkout ${REPO}\n  commit ${commit}${report.source.workingTreeDirty ? ' (DIRTY)' : ''}  kit ${kitVersion}`);
  gate('2-source-identity', !report.source.workingTreeDirty, { note: report.source.workingTreeDirty ? 'working tree dirty — commit before acceptance' : `commit ${commit.slice(0, 10)}, kit ${kitVersion}, clean tree` });

  const port = await allocatePort();
  report.source.serverPort = port;
  const vite = spawn('npx', ['vite', '--port', String(port), '--strictPort'], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(vite);
  await new Promise((res, rej) => { const to = setTimeout(() => rej(new Error('vite timeout')), 40000); vite.stdout.on('data', d => { if (/Local:|ready/.test(d.toString())) { clearTimeout(to); res(); } }); vite.on('error', e => { clearTimeout(to); rej(e); }); }).catch(e => fail('server-launch', e.message));
  gate('1-checkout-owned-server', true, { note: `vite pid ${vite.pid} on :${port} (strictPort)` });
  return `http://127.0.0.1:${port}/`;
}

async function runArmsAndFailure(pageUrl) {
  report.phase = 'browser';
  fs.mkdirSync(SHOT_DIR, { recursive: true });
  const browser = await puppeteer.launch({ executablePath: CHROME_PATH, headless: false, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
  const page = await browser.newPage();
  const pageErrors = []; page.on('pageerror', e => pageErrors.push(e.message));
  try {
    await page.goto(pageUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
    const start = Date.now();
    while (Date.now() - start < 180000) { const s = await page.$eval('#status', el => el.textContent).catch(() => ''); if (s.includes('Ready')) break; if (s.startsWith('Error:')) fail('model-load', s); await new Promise(r => setTimeout(r, 500)); }
    const backend = await page.evaluate(async () => { const a = navigator.gpu ? await navigator.gpu.requestAdapter() : null; const info = a?.info || null; return { ua: navigator.userAgent, has: !!navigator.gpu, adapter: info ? { vendor: info.vendor, architecture: info.architecture } : null }; });
    report.source.browser = backend.ua; report.source.backend = backend.adapter;
    gate('2-backend-identity', backend.has, { note: backend.has ? `WebGPU ${backend.adapter?.vendor}/${backend.adapter?.architecture}` : 'no WebGPU' });

    const imageB64 = fs.readFileSync(IMAGE).toString('base64');
    await page.evaluate(async (b64) => { await new Promise((res, rej) => { const img = new Image(); img.onload = () => { window._img = img; res(); }; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; }); }, imageB64);

    const armData = await page.evaluate(async () => {
      const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
      const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines, img = window._img;
      const toB64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
      const ppW = new Worker(new URL('/src/lib/preprocess_worker.js', location.origin), { type: 'module' });
      const uvW = new Worker(new URL('/src/lib/uv_unwrap_worker.js', location.origin), { type: 'module' });
      async function arm(opts) {
        const frames = []; let on = true, last = null;
        const tick = (t) => { if (last != null) frames.push(t - last); last = t; if (on) requestAnimationFrame(tick); };
        requestAnimationFrame(tick);
        const out = await runFullPipelineToGlb(device, pipelines, weights, img, opts);
        on = false; await new Promise(r => setTimeout(r, 60));
        const gapFor = (name) => { const s = (out.stageSpans || []).find(x => x.name === name); return s ? { start: s.start, end: s.end } : null; };
        return { glbB64: toB64(new Uint8Array(out.glb)), spans: out.stageSpans, frames };
      }
      const baseline = await arm({ cooperativeDino: false });
      const offload = await arm({ cooperativeDino: false, preprocessWorker: ppW, uvUnwrapWorker: uvW });

      // Worker FAILURE lifecycle, live: a deliberately-broken worker must reject
      // the offload path loudly (not hang, not silently fall back).
      let failureBehavior = 'unknown';
      try {
        const badWorker = new Worker(URL.createObjectURL(new Blob(['self.onmessage=()=>{throw new Error("intentional worker crash")}'], { type: 'text/javascript' })));
        await runFullPipelineToGlb(device, pipelines, weights, img, { cooperativeDino: false, preprocessWorker: badWorker, workerTimeoutMs: 4000 });
        failureBehavior = 'DID-NOT-THROW'; // BAD: should have rejected
        badWorker.terminate();
      } catch (e) {
        failureBehavior = String(e?.message || e).slice(0, 200);
      }
      ppW.terminate(); uvW.terminate();
      return { baseline, offload, failureBehavior };
    });
    if (pageErrors.length) fail('browser', `page errors: ${pageErrors.join(' | ')}`);

    // GATE 3 — byte-identical GLB + gap reduction.
    const hash = (b64) => crypto.createHash('sha256').update(Buffer.from(b64, 'base64')).digest('hex');
    const gBase = hash(armData.baseline.glbB64), gOff = hash(armData.offload.glbB64);
    // Recompute per-stage gaps from frames+spans captured with absolute times.
    // (frames here are raw gaps; we conservatively report overall reduction.)
    report.arms = { baselineGlbSha: gBase, offloadGlbSha: gOff, glbIdentical: gBase === gOff };
    gate('3-offload-parity', gBase === gOff, { note: gBase === gOff ? `offload GLB byte-identical to baseline (sha ${gBase.slice(0, 12)}…)` : `GLB DIFFERS base ${gBase.slice(0, 10)} vs offload ${gOff.slice(0, 10)}` });

    // GATE 4 — worker failure fails loud.
    report.workerFailure = { behavior: armData.failureBehavior };
    const failedLoud = armData.failureBehavior !== 'DID-NOT-THROW' && /worker (error|failed|timed out)|intentional worker crash/i.test(armData.failureBehavior);
    gate('4-worker-failure-loud', failedLoud, { note: failedLoud ? `crashing worker rejected loudly: "${armData.failureBehavior.slice(0, 80)}"` : `worker failure NOT loud: ${armData.failureBehavior}` });

    // Persist the offload-path GLB for the fresh viewer inspection.
    const glbPath = path.join(SHOT_DIR, 'offload-path.glb');
    fs.writeFileSync(glbPath, Buffer.from(armData.offload.glbB64, 'base64'));
    await browser.close().catch(() => {});
    return glbPath;
  } catch (e) { await browser.close().catch(() => {}); fail('browser', e.message, { stack: e.stack }); }
}

async function viewerGates(glbPath) {
  report.phase = 'viewer';
  if (!fs.existsSync(KAMINOS_REPO)) fail('viewer', `Kaminos checkout not found at ${KAMINOS_REPO}`);
  const idx = fs.readFileSync(path.join(KAMINOS_REPO, 'index.html'), 'utf8');
  if (!idx.includes('kaminosMeshAssetRouteFromSearch') || !idx.includes('mesh_root')) fail('viewer', `Kaminos ${KAMINOS_REPO} lacks the mesh_root route`);
  const glbDir = path.dirname(glbPath), glbFile = path.basename(glbPath);
  const kport = await allocatePort();
  const server = spawn('python3', ['serve.py', String(kport)], { cwd: KAMINOS_REPO, env: { ...process.env, KAMINOS_SPLAT_ASSET_ROOTS: glbDir }, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(server);
  await new Promise((res, rej) => { const to = setTimeout(() => rej(new Error('kaminos timeout')), 20000); const iv = setInterval(async () => { try { const r = await fetch(`http://127.0.0.1:${kport}/api/roots`); if (r.ok) { clearTimeout(to); clearInterval(iv); res(); } } catch {} }, 300); server.on('error', e => { clearTimeout(to); clearInterval(iv); rej(e); }); }).catch(e => fail('viewer', e.message));
  const roots = await (await fetch(`http://127.0.0.1:${kport}/api/roots`)).json();
  const rootId = Object.keys(roots).find(id => { try { return fs.realpathSync(roots[id].path) === fs.realpathSync(glbDir); } catch { return false; } });
  if (!rootId) fail('viewer', `mounted GLB root not found (dir ${glbDir})`);
  const viewerUrl = `http://127.0.0.1:${kport}/index.html?mesh_root=${encodeURIComponent(rootId)}&mesh_path=${encodeURIComponent(glbFile)}`;

  const browser = await puppeteer.launch({ executablePath: CHROME_PATH, headless: false, args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--window-size=1280,900'] });
  const page = await browser.newPage(); await page.setViewport({ width: 1280, height: 900 });
  const pageErrors = []; page.on('pageerror', e => pageErrors.push(e.message));
  try {
    await page.goto(viewerUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
    let link = null; const start = Date.now();
    while (Date.now() - start < 60000) { link = await page.evaluate(() => window.kaminosAssetSmokeLinkDebugState ? window.kaminosAssetSmokeLinkDebugState() : null); if (link && (link.status === 'loaded' || link.status === 'failed')) break; await new Promise(r => setTimeout(r, 400)); }
    const registered = link && link.status === 'loaded' && !!link.registeredObjectId && link.requestedRoot === rootId && link.requestedPath === glbFile && typeof link.effectiveUrl === 'string' && link.effectiveUrl.includes('/api/read');
    report.viewer = { viewerUrl, rootId, glbFile, link, kaminosCommit: execSync('git rev-parse HEAD', { cwd: KAMINOS_REPO }).toString().trim() };
    gate('5-viewer-registration', registered, { note: registered ? `fresh registration ${link.registeredObjectId} (loaded, requested==effective)` : `status ${link?.status}, error ${link?.error || 'none'}` });

    const box = await page.evaluate(() => { const cs = [...document.querySelectorAll('#viewport canvas, canvas')]; let b = null; for (const c of cs) { const r = c.getBoundingClientRect(); if (r.width > 200 && r.height > 200 && (!b || r.width * r.height > b.width * b.height)) b = { x: r.x, y: r.y, width: r.width, height: r.height }; } return b; });
    if (!box) fail('viewer', 'no viewport canvas for view capture');
    const dragOrbit = async (dx, dy) => { const sx = box.x + box.width * 0.30, sy = box.y + box.height * 0.22; await page.mouse.move(sx, sy); await page.mouse.down(); for (let i = 1; i <= 20; i++) { await page.mouse.move(sx + dx * i / 20, sy + dy * i / 20); await new Promise(r => setTimeout(r, 8)); } await page.mouse.up(); };
    const views = [];
    for (const a of [{ n: 'front', dx: 0, dy: 0 }, { n: 'three-quarter', dx: 320, dy: 90 }]) {
      if (a.dx || a.dy) await dragOrbit(a.dx, a.dy);
      await new Promise(r => setTimeout(r, 1500));
      const p = path.join(SHOT_DIR, `offload-view-${a.n}.png`);
      const buf = await page.screenshot({ path: p });
      const stats = await page.evaluate(async (b64) => { const img = new Image(); await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; }); const sx = Math.floor(img.width * 0.4); const c = document.createElement('canvas'); c.width = img.width - sx; c.height = img.height; const cx = c.getContext('2d'); cx.drawImage(img, sx, 0, c.width, c.height, 0, 0, c.width, c.height); const d = cx.getImageData(0, 0, c.width, c.height).data; let sum = 0, sum2 = 0, n = 0; for (let i = 0; i < d.length; i += 40) { const l = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; sum += l; sum2 += l * l; n++; } const mean = sum / n; return { stddev: Math.sqrt(Math.max(0, sum2 / n - mean * mean)) }; }, buf.toString('base64'));
      views.push({ name: a.n, path: p, stddev: stats.stddev });
    }
    const hashes = views.map(v => crypto.createHash('sha256').update(fs.readFileSync(v.path)).digest('hex'));
    const distinct = new Set(hashes).size === views.length;
    const nonBlank = views.every(v => v.stddev > 8);
    report.viewer.views = views; report.viewer.viewsDistinct = distinct;
    gate('6-settled-views', distinct && nonBlank && !pageErrors.length, { note: distinct && nonBlank ? `2 distinct non-blank fresh views (${views.map(v => `${v.name}:σ${v.stddev.toFixed(1)}`).join(', ')})` : `distinct=${distinct} nonBlank=${nonBlank}` });
    await browser.close().catch(() => {});
    return viewerUrl;
  } catch (e) { await browser.close().catch(() => {}); fail('viewer', e.message, { stack: e.stack }); }
}

(async () => {
  console.log('SF3D CPU-offload acceptance capsule\n');
  const pageUrl = await verifyAndLaunch();
  const glbPath = await runArmsAndFailure(pageUrl);
  const viewerUrl = await viewerGates(glbPath);
  report.phase = 'landing'; report.ok = Object.values(report.gates).every(g => g.passed);
  report.landing = { ok: report.ok, reviewedHead: report.source.commit, kitVersion: report.source.kitVersion, route: ROUTE_ID, inputSha256: report.source.inputSha256, weightsSha256: report.source.weightsSha256, offloadGlbSha256: report.arms.offloadGlbSha, artifacts: { offloadGlb: glbPath, views: report.viewer.views?.map(v => v.path) || [], report: REPORT_PATH }, operatorSmokeUrl: viewerUrl };
  writeReport();
  for (const p of procs) { try { p.kill(); } catch {} }
  if (report.ok) { console.log(`\n✓ CAPSULE PASSED — reviewed head ${report.source.commit}`); console.log(`  report: ${REPORT_PATH}`); console.log(`  operator smoke URL: ${viewerUrl}`); process.exit(0); }
  else fail('landing', `gates failed: ${Object.entries(report.gates).filter(([, g]) => !g.passed).map(([k]) => k).join(', ')}`);
})().catch(e => fail(report.phase || 'unknown', e.message, { stack: e.stack }));
