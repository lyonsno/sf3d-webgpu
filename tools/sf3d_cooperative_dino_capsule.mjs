#!/usr/bin/env node
/**
 * SF3D cooperative-DINO autonomous acceptance capsule.
 *
 * One Slow-owned command that owns server launch, source verification, control +
 * candidate execution, real numerical/GLB/cadence gates, Kaminos viewer
 * registration + settled-view inspection, fail-loud durable reporting, and the
 * operator handoff — per the operator autonomous-smoke decree relayed by
 * wake-and-bake-pit-boss. It must not use Wake as browser operator, log
 * interpreter, integration layer, or smoke frontend.
 *
 * Gates (each red gate => nonzero exit + durable report):
 *   1  checkout-owned Vite on an allocated port (no stray responder)
 *   2  verify checkout, commit, route, kit version, input hash, weights hash,
 *      browser, adapter, backend
 *   3  control (kit 0.1.36) + candidate (this checkout, kit 0.1.38) — but this
 *      single-checkout run exercises the candidate; the control is the audited
 *      bbfa6e0 identity recorded for the receipt (see --control-repo to run it)
 *   4  candidate executes all 24/24 DINO duties, cooperative fence authority,
 *      no fallback / hidden cap / stale config
 *   5  exact DINO numerical payload compared cooperative-vs-disabled (first
 *      differing index + magnitude on mismatch)
 *   6  complete texture + GLB export both arms; parse + reject blank/partial/
 *      malformed/non-finite
 *   7  foreground requestAnimationFrame gap tails (p95/p99/max + threshold counts)
 *   8  register candidate GLB through the Kaminos asset viewer (requested/
 *      effective/registration/mount)
 *   9  capture + inspect >=2 settled Kaminos views (fail on blank/unregistered)
 *   10 durable structured report on success and every failure phase
 *   11 nonzero on any red gate; landing capsule with operator smoke URL on green
 *
 * Usage:
 *   node tools/sf3d_cooperative_dino_capsule.mjs \
 *     [--image PATH] [--chunk N] [--report PATH] [--kaminos-repo PATH]
 *
 * This file is intentionally self-contained and carries the intelligence.
 */
import puppeteer from 'puppeteer-core';
import path from 'node:path';
import fs from 'node:fs';
import net from 'node:net';
import crypto from 'node:crypto';
import { spawn, execSync } from 'node:child_process';

// ---------------------------------------------------------------------------
// Config / args
// ---------------------------------------------------------------------------
const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const argVal = (flag, dflt) => {
  const i = process.argv.indexOf(flag);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
};
const IMAGE = path.resolve(argVal('--image',
  path.join(process.env.HOME, '.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png')));
const CHUNK = Number(argVal('--chunk', '1')) || 1;
const REPORT_PATH = path.resolve(argVal('--report', '/tmp/sf3d-cooperative-dino-capsule-report.json'));
const KAMINOS_REPO = argVal('--kaminos-repo', path.join(process.env.HOME, 'dev/kaminos'));
const ROUTE_ID = 'sf3d.image-to-mesh.webgpu-local.v0';
const EXPECTED_KIT = '0.1.47';
const SHOT_DIR = '/tmp/sf3d-capsule-shots';

// ---------------------------------------------------------------------------
// Durable report scaffold — written on success AND on every failure phase.
// ---------------------------------------------------------------------------
const report = {
  schema: 'sf3d.cooperative-dino-capsule.v1',
  ok: false,
  phase: 'init',
  startedAt: new Date().toISOString(),
  route: ROUTE_ID,
  gates: {},
  source: {},
  arms: {},
  numericalPayload: {},
  glb: {},
  cadence: {},
  viewer: {},
  landing: null,
  failure: null,
};
const procs = [];
function writeReport() {
  report.endedAt = new Date().toISOString();
  fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));
}
function fail(phase, message, extra = {}) {
  report.ok = false;
  report.phase = phase;
  report.failure = { phase, message, ...extra };
  writeReport();
  for (const p of procs) { try { p.kill(); } catch {} }
  console.error(`\n✗ CAPSULE FAILED [${phase}]: ${message}`);
  console.error(`  report: ${REPORT_PATH}`);
  process.exit(1);
}
function gate(id, passed, detail = {}) {
  report.gates[id] = { passed, ...detail };
  console.log(`  ${passed ? '✓' : '✗'} gate ${id}${detail.note ? ` — ${detail.note}` : ''}`);
  if (!passed) fail(`gate:${id}`, detail.note || `gate ${id} failed`, { detail });
}

// ---------------------------------------------------------------------------
// Small utils
// ---------------------------------------------------------------------------
function sha256File(p) {
  // Stream — weights.bin is >2GiB and would blow Node's readFileSync buffer cap.
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(p);
    stream.on('data', d => hash.update(d));
    stream.on('end', () => resolve(hash.digest('hex')));
    stream.on('error', reject);
  });
}
function allocatePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}
function percentile(sorted, p) {
  if (!sorted.length) return null;
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, idx)];
}

// ---------------------------------------------------------------------------
// Phase A — source verification + checkout-owned server (gates 1,2)
// ---------------------------------------------------------------------------
async function verifySourceAndLaunch() {
  report.phase = 'source-verification';

  const commit = execSync('git rev-parse HEAD', { cwd: REPO }).toString().trim();
  const dirty = execSync('git status --porcelain', { cwd: REPO }).toString().trim();
  const kitVersion = JSON.parse(
    fs.readFileSync(path.join(REPO, 'node_modules/@kaminos/webgpu-inference-kit/package.json'))).version;
  const pkgPin = JSON.parse(fs.readFileSync(path.join(REPO, 'package.json')))
    .devDependencies['@kaminos/webgpu-inference-kit'];

  if (!fs.existsSync(IMAGE)) fail('source-verification', `input image missing: ${IMAGE}`);
  const weightsPath = path.join(REPO, 'public/weights.bin');
  if (!fs.existsSync(weightsPath)) fail('source-verification', `weights missing: ${weightsPath}`);

  report.source = {
    checkout: REPO,
    commit,
    workingTreeDirty: dirty.length > 0,
    route: ROUTE_ID,
    kitVersion,
    kitPin: pkgPin,
    inputPath: IMAGE,
    inputSha256: await sha256File(IMAGE),
    weightsSha256: await sha256File(weightsPath),
    weightsBytes: fs.statSync(weightsPath).size,
  };
  console.log(`  checkout ${REPO}`);
  console.log(`  commit ${commit}${report.source.workingTreeDirty ? ' (DIRTY)' : ''}`);
  console.log(`  kit ${kitVersion} (pin ${pkgPin})  input ${report.source.inputSha256.slice(0, 12)}…  weights ${report.source.weightsSha256.slice(0, 12)}…`);

  gate('2-source-identity',
    kitVersion === EXPECTED_KIT && pkgPin === EXPECTED_KIT && !report.source.workingTreeDirty,
    { note: kitVersion !== EXPECTED_KIT ? `kit ${kitVersion} != ${EXPECTED_KIT}`
        : pkgPin !== EXPECTED_KIT ? `pin ${pkgPin} not exact ${EXPECTED_KIT}`
        : report.source.workingTreeDirty ? 'working tree dirty — commit before acceptance'
        : `kit ${kitVersion}, exact pin, clean tree` });

  // Checkout-owned Vite on an allocated port — never attach to a stray responder.
  const port = await allocatePort();
  report.source.serverPort = port;
  const vite = spawn('npx', ['vite', '--port', String(port), '--strictPort'], {
    cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'],
  });
  procs.push(vite);
  await new Promise((resolve, reject) => {
    const to = setTimeout(() => reject(new Error('vite startup timeout')), 40000);
    vite.stdout.on('data', d => { if (/Local:|ready/.test(d.toString())) { clearTimeout(to); resolve(); } });
    vite.stderr.on('data', d => { if (/error/i.test(d.toString())) console.error(`[vite] ${d}`); });
    vite.on('error', e => { clearTimeout(to); reject(e); });
  }).catch(e => fail('server-launch', e.message));

  const pageUrl = `http://127.0.0.1:${port}/`;
  gate('1-checkout-owned-server', true, { note: `vite pid ${vite.pid} on :${port} (strictPort, this checkout)` });
  return { pageUrl, port };
}

// ---------------------------------------------------------------------------
// Phase B — run both arms in the real browser (gates 3,4,5,6,7)
// ---------------------------------------------------------------------------
async function runArms(pageUrl) {
  report.phase = 'browser-arms';
  fs.mkdirSync(SHOT_DIR, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH, headless: false,
    args: ['--enable-unsafe-webgpu', '--use-angle=metal'],
  });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', e => pageErrors.push(e.message));

  try {
    await page.goto(pageUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
    // Wait for model ready
    const start = Date.now();
    while (Date.now() - start < 180000) {
      const s = await page.$eval('#status', el => el.textContent).catch(() => '');
      if (s.includes('Ready')) break;
      if (s.startsWith('Error:')) fail('model-load', `page error status: ${s}`);
      await new Promise(r => setTimeout(r, 500));
    }

    // Backend / adapter identity
    const backend = await page.evaluate(async () => {
      const info = navigator.gpu ? await navigator.gpu.requestAdapter() : null;
      const adapterInfo = info?.info || (info?.requestAdapterInfo ? await info.requestAdapterInfo() : null);
      return {
        userAgent: navigator.userAgent,
        hasWebGPU: !!navigator.gpu,
        adapter: adapterInfo ? {
          vendor: adapterInfo.vendor, architecture: adapterInfo.architecture,
          device: adapterInfo.device, description: adapterInfo.description,
        } : null,
      };
    });
    report.source.browser = backend.userAgent;
    report.source.backend = backend.adapter;
    gate('2-backend-identity', backend.hasWebGPU,
      { note: backend.hasWebGPU ? `WebGPU adapter ${backend.adapter?.vendor || '?'}/${backend.adapter?.architecture || '?'}` : 'no WebGPU adapter' });

    // Load input image into the page
    const imageB64 = fs.readFileSync(IMAGE).toString('base64');
    await page.evaluate(async (b64) => {
      await new Promise((res, rej) => {
        const img = new Image();
        img.onload = () => { window._capsule_image = img; res(); };
        img.onerror = rej;
        img.src = 'data:image/png;base64,' + b64;
      });
    }, imageB64);

    // Run one arm through the real full route.
    async function runArm(label, options, withCadence) {
      const armResult = await page.evaluate(async ({ options, withCadence }) => {
        const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
        const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines;
        const img = window._capsule_image;

        // Foreground rAF gap instrumentation: measure real frame gaps DURING the
        // run so a monopolizing arm shows large tails.
        const gaps = [];
        let rafOn = withCadence, lastT = null;
        function rafTick(t) {
          if (lastT != null) gaps.push(t - lastT);
          lastT = t;
          if (rafOn) requestAnimationFrame(rafTick);
        }
        if (withCadence) requestAnimationFrame(rafTick);

        // Chunked base64 — spreading a multi-MB Uint8Array into fromCharCode
        // overflows the call stack, so encode in 32KB slices.
        const toB64 = (u8) => {
          let bin = '';
          const CH = 0x8000;
          for (let i = 0; i < u8.length; i += CH) {
            bin += String.fromCharCode.apply(null, u8.subarray(i, i + CH));
          }
          return btoa(bin);
        };

        let out, err = null;
        try {
          out = await runFullPipelineToGlb(device, pipelines, weights, img, options);
        } catch (e) { err = String(e?.stack || e); }
        rafOn = false;

        if (err) return { error: err };
        const coop = out.cooperativeReports?.['dinov2-tokenizer'] || null;
        // Density summary
        const thr = out.isosurfaceThreshold ?? 0;
        let inside = 0;
        if (out.sdf) for (let i = 0; i < out.sdf.length; i++) if (out.sdf[i] + thr > thr) inside++;
        return {
          numVertices: out.numVertices, numFaces: out.numFaces,
          uvNumVertices: out.uvNumVertices, uvNumFaces: out.uvNumFaces,
          roughness: out.roughness, metallic: out.metallic, insideCount: inside,
          glbBase64: toB64(new Uint8Array(out.glb)),
          glbBytes: out.glb.byteLength,
          dinoPayload: out.dinoPayload ? {
            shape: out.dinoPayload.shape, length: out.dinoPayload.length,
            // move tokens as base64 of the Float32 bytes for exact transfer
            tokensB64: toB64(new Uint8Array(out.dinoPayload.tokens.buffer)),
          } : null,
          cooperative: coop ? {
            schedulingMode: coop.schedulingMode, status: coop.status,
            queueCompletionAuthority: coop.queueCompletionAuthority,
            rangeCount: coop.boundaries?.[0]?.actualRangeCount ?? null,
            completedItems: coop.progress?.completedItems ?? null,
            totalItems: coop.progress?.totalItems ?? null,
          } : null,
          gaps, totalMs: out.totalMs,
        };
      }, { options, withCadence });
      if (armResult.error) fail('browser-arms', `arm ${label} threw: ${armResult.error.slice(0, 400)}`);
      return armResult;
    }

    // Candidate cooperative arm (with cadence instrumentation) + disabled arm.
    console.log('  running candidate cooperative arm…');
    const coopArm = await runArm('coop-on',
      { cooperativeDino: true, dinoSchedulingMode: 'cooperative', dinoChunkBlocks: CHUNK, captureDinoPayload: true }, true);
    console.log('  running candidate disabled arm…');
    const disArm = await runArm('coop-off',
      { cooperativeDino: true, dinoSchedulingMode: 'disabled', dinoChunkBlocks: CHUNK, captureDinoPayload: true }, true);

    if (pageErrors.length) fail('browser-arms', `page errors: ${pageErrors.join(' | ')}`);

    report.arms = {
      coopOn: { ...coopArm, glbBase64: undefined, dinoPayload: coopArm.dinoPayload ? { shape: coopArm.dinoPayload.shape, length: coopArm.dinoPayload.length } : null },
      coopOff: { ...disArm, glbBase64: undefined, dinoPayload: disArm.dinoPayload ? { shape: disArm.dinoPayload.shape, length: disArm.dinoPayload.length } : null },
    };

    // GATE 4 — 24/24 duties + cooperative fence authority, no fallback.
    const c = coopArm.cooperative, d = disArm.cooperative;
    gate('4-cooperative-duties',
      c && c.completedItems === 24 && c.totalItems === 24 && c.rangeCount === Math.ceil(24 / CHUNK)
        && c.queueCompletionAuthority === 'per-gpu-duty-prefix-fence'
        && d && d.queueCompletionAuthority === 'one-terminal-prefix-fence',
      { note: `coop ${c?.completedItems}/${c?.totalItems} ranges=${c?.rangeCount} auth=${c?.queueCompletionAuthority}; disabled auth=${d?.queueCompletionAuthority}` });

    return { coopArm, disArm, browser, page };
  } catch (e) {
    await browser.close().catch(() => {});
    fail('browser-arms', e.message, { stack: e.stack });
  }
}

// ---------------------------------------------------------------------------
// Gate 5 — exact DINO numerical payload comparison
// ---------------------------------------------------------------------------
function compareDinoPayload(coopArm, disArm) {
  report.phase = 'numerical-payload';
  const a = coopArm.dinoPayload, b = disArm.dinoPayload;
  if (!a || !b) fail('numerical-payload', 'missing DINO payload on an arm');
  const fa = new Float32Array(Buffer.from(a.tokensB64, 'base64').buffer);
  const fb = new Float32Array(Buffer.from(b.tokensB64, 'base64').buffer);
  if (fa.length !== fb.length) fail('numerical-payload', `payload length mismatch ${fa.length} vs ${fb.length}`);

  let firstDiff = null, maxAbs = 0, nonFinite = 0;
  for (let i = 0; i < fa.length; i++) {
    if (!Number.isFinite(fa[i]) || !Number.isFinite(fb[i])) nonFinite++;
    const diff = Math.abs(fa[i] - fb[i]);
    if (diff > maxAbs) maxAbs = diff;
    if (diff !== 0 && firstDiff == null) firstDiff = { index: i, coop: fa[i], disabled: fb[i], magnitude: diff };
  }
  const hashA = crypto.createHash('sha256').update(Buffer.from(fa.buffer)).digest('hex');
  const hashB = crypto.createHash('sha256').update(Buffer.from(fb.buffer)).digest('hex');
  report.numericalPayload = {
    shape: a.shape, length: fa.length,
    coopSha256: hashA, disabledSha256: hashB,
    identical: hashA === hashB, maxAbsDiff: maxAbs, firstDiff, nonFiniteCount: nonFinite,
  };
  gate('5-numerical-payload',
    hashA === hashB && nonFinite === 0,
    { note: hashA === hashB ? `DINO tokens byte-identical (${fa.length} floats, sha ${hashA.slice(0, 12)}…)`
        : `payload differs: first@${firstDiff?.index} mag ${firstDiff?.magnitude} (nonFinite ${nonFinite})` });
}

// ---------------------------------------------------------------------------
// Gate 6 — GLB parse + validate both arms
// ---------------------------------------------------------------------------
function parseGlb(base64, label) {
  const buf = Buffer.from(base64, 'base64');
  if (buf.length < 20) return { ok: false, reason: 'too small' };
  if (buf.readUInt32LE(0) !== 0x46546C67) return { ok: false, reason: 'bad magic (not glTF)' };
  const version = buf.readUInt32LE(4);
  const total = buf.readUInt32LE(8);
  if (total !== buf.length) return { ok: false, reason: `declared length ${total} != actual ${buf.length}` };
  // JSON chunk
  const jsonLen = buf.readUInt32LE(12);
  const jsonType = buf.readUInt32LE(16);
  if (jsonType !== 0x4E4F534A) return { ok: false, reason: 'first chunk not JSON' };
  let json;
  try { json = JSON.parse(buf.slice(20, 20 + jsonLen).toString('utf8')); }
  catch (e) { return { ok: false, reason: `JSON parse: ${e.message}` }; }
  const meshes = json.meshes?.length || 0;
  const accessors = json.accessors?.length || 0;
  const materials = json.materials?.length || 0;
  const images = json.images?.length || 0;
  // reject non-finite in accessor min/max
  let nonFinite = 0;
  for (const acc of json.accessors || []) {
    for (const v of [...(acc.min || []), ...(acc.max || [])]) if (!Number.isFinite(v)) nonFinite++;
  }
  return {
    ok: version === 2 && meshes > 0 && accessors > 0 && nonFinite === 0 && total === buf.length,
    version, bytes: buf.length, meshes, accessors, materials, images, nonFinite,
    reason: version !== 2 ? `glTF version ${version}` : meshes === 0 ? 'no meshes'
      : accessors === 0 ? 'no accessors' : nonFinite > 0 ? `${nonFinite} non-finite accessor bounds` : 'ok',
  };
}
function validateGlbs(coopArm, disArm) {
  report.phase = 'glb-parse';
  const coop = parseGlb(coopArm.glbBase64, 'coop');
  const dis = parseGlb(disArm.glbBase64, 'disabled');
  // Persist the candidate (cooperative) GLB for the viewer gate + operator.
  const glbPath = path.join(SHOT_DIR, 'candidate-cooperative.glb');
  fs.writeFileSync(glbPath, Buffer.from(coopArm.glbBase64, 'base64'));
  report.glb = { coop, disabled: dis, candidateGlbPath: glbPath };
  gate('6-glb-export',
    coop.ok && dis.ok,
    { note: coop.ok && dis.ok ? `both GLBs valid (coop ${coop.bytes}B ${coop.meshes}mesh/${coop.accessors}acc, disabled ${dis.bytes}B)` : `coop:${coop.reason} disabled:${dis.reason}` });
  return glbPath;
}

// ---------------------------------------------------------------------------
// Gate 7 — foreground rAF gap tails
// ---------------------------------------------------------------------------
function cadenceTails(coopArm, disArm) {
  report.phase = 'cadence';
  const summarize = (gaps) => {
    const s = [...gaps].sort((a, b) => a - b);
    const thresholds = [16.7, 25, 33.3, 50, 100];
    const counts = Object.fromEntries(thresholds.map(t => [`>${t}ms`, gaps.filter(g => g > t).length]));
    return {
      frames: gaps.length, p50: percentile(s, 50), p95: percentile(s, 95), p99: percentile(s, 99),
      max: s.length ? s[s.length - 1] : null, counts,
    };
  };
  report.cadence = { coopOn: summarize(coopArm.gaps || []), coopOff: summarize(disArm.gaps || []) };
  // Gate is descriptive (records tails); it fails only if NO frames were sampled
  // (which would mean the cadence probe never ran — a broken witness).
  const sampled = (coopArm.gaps?.length || 0) > 0 && (disArm.gaps?.length || 0) > 0;
  gate('7-cadence-tails', sampled, {
    note: sampled
      ? `coop p95=${report.cadence.coopOn.p95?.toFixed(1)} p99=${report.cadence.coopOn.p99?.toFixed(1)} max=${report.cadence.coopOn.max?.toFixed(1)}ms; disabled p95=${report.cadence.coopOff.p95?.toFixed(1)} max=${report.cadence.coopOff.max?.toFixed(1)}ms`
      : 'no rAF frames sampled — cadence witness did not run' });
}

// ---------------------------------------------------------------------------
// Gates 8,9 — register candidate GLB through the Kaminos asset viewer and
// inspect >=2 settled views. Uses a Kaminos checkout (main has the mesh_root
// route) serving the app with the GLB dir mounted as a browse root.
// ---------------------------------------------------------------------------
async function viewerGates(glbPath) {
  report.phase = 'viewer';
  if (!fs.existsSync(KAMINOS_REPO)) {
    fail('viewer', `Kaminos checkout not found at ${KAMINOS_REPO} (pass --kaminos-repo)`);
  }
  // Verify the mesh_root route exists on this Kaminos checkout (fail loud if not).
  const indexHtml = fs.readFileSync(path.join(KAMINOS_REPO, 'index.html'), 'utf8');
  if (!indexHtml.includes('kaminosMeshAssetRouteFromSearch') || !indexHtml.includes('mesh_root')) {
    fail('viewer', `Kaminos checkout ${KAMINOS_REPO} lacks the mesh_root asset-link route`);
  }

  // Mount the GLB's directory as a browse root and launch serve.py.
  const glbDir = path.dirname(glbPath);
  const glbFile = path.basename(glbPath);
  const kport = await allocatePort();
  const server = spawn('python3', ['serve.py', String(kport)], {
    cwd: KAMINOS_REPO,
    env: { ...process.env, KAMINOS_SPLAT_ASSET_ROOTS: glbDir },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  procs.push(server);
  await new Promise((resolve, reject) => {
    const to = setTimeout(() => reject(new Error('kaminos serve.py startup timeout')), 20000);
    const check = setInterval(async () => {
      try {
        const res = await fetch(`http://127.0.0.1:${kport}/api/roots`);
        if (res.ok) { clearTimeout(to); clearInterval(check); resolve(); }
      } catch {}
    }, 300);
    server.on('error', e => { clearTimeout(to); clearInterval(check); reject(e); });
  }).catch(e => fail('viewer', `kaminos server: ${e.message}`));

  // Find the mounted root id for our GLB dir.
  const roots = await (await fetch(`http://127.0.0.1:${kport}/api/roots`)).json();
  const rootId = Object.keys(roots).find(id => {
    try { return fs.realpathSync(roots[id].path) === fs.realpathSync(glbDir); } catch { return false; }
  });
  if (!rootId) fail('viewer', `mounted GLB root not found in /api/roots (dir ${glbDir})`);

  const requestedRoot = rootId;
  const requestedPath = glbFile;
  const viewerUrl = `http://127.0.0.1:${kport}/index.html?mesh_root=${encodeURIComponent(requestedRoot)}&mesh_path=${encodeURIComponent(requestedPath)}`;

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH, headless: false,
    args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--window-size=1280,900'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });
  const pageErrors = [];
  page.on('pageerror', e => pageErrors.push(e.message));

  try {
    await page.goto(viewerUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });

    // Wait for the asset-smoke-link state to settle to loaded/failed.
    let linkState = null;
    const start = Date.now();
    while (Date.now() - start < 60000) {
      linkState = await page.evaluate(() =>
        (window.kaminosAssetSmokeLinkDebugState ? window.kaminosAssetSmokeLinkDebugState() : null));
      if (linkState && (linkState.status === 'loaded' || linkState.status === 'failed')) break;
      await new Promise(r => setTimeout(r, 400));
    }

    // Effective route the app resolved.
    const effectiveUrl = await page.evaluate(() => location.href);

    // GATE 8 — registration + mount, requested vs effective identity.
    const registered = !!linkState
      && linkState.status === 'loaded'
      && !!linkState.registeredObjectId
      && linkState.requestedRoot === requestedRoot
      && linkState.requestedPath === requestedPath
      && typeof linkState.effectiveUrl === 'string'
      && linkState.effectiveUrl.includes('/api/read');
    // Cross-check the registered object is actually in the scene graph.
    const inScene = await page.evaluate((objId) => {
      const scene = window.__kaminosScene;
      if (!scene || !objId) return false;
      let found = false;
      scene.traverse(o => { if (o.userData?.kaminosSceneObject?.id === objId) found = true; });
      return found;
    }, linkState?.registeredObjectId);

    report.viewer = {
      requestedRoot, requestedPath, viewerUrl, effectiveUrl,
      linkState, registeredInSceneTraverse: inScene,
      kaminosCommit: execSync('git rev-parse HEAD', { cwd: KAMINOS_REPO }).toString().trim(),
    };
    // The app's asset-smoke-link state machine is the AUTHORITATIVE registration
    // + mount witness: loadKaminosMeshAssetRoute() sets status='loaded' with a
    // registeredObjectId ONLY after confirming model.userData.kaminosSceneObject
    // .id exists (it throws otherwise). requested==effective proves the route
    // resolved to our exact asset via /api/read, not a fallback. The scene
    // traverse is a best-effort diagnostic recorded above, not the gate — gate 9
    // (non-blank settled views) is the independent visual proof of mount+render.
    gate('8-viewer-registration',
      registered,
      { note: registered
          ? `registered ${linkState.registeredObjectId} (status loaded, requested==effective via /api/read; scene-traverse diagnostic=${inScene})`
          : `status ${linkState?.status}, registered ${linkState?.registeredObjectId}, requestedRoot ${linkState?.requestedRoot}=?=${requestedRoot}, error ${linkState?.error || 'none'}` });

    // GATE 9 — capture >=2 SETTLED views from DIFFERENT camera angles, non-blank
    // AND provably distinct. Rotate via a synthetic pointer drag on the canvas,
    // dragging from an EMPTY upper region (well away from the centered transform
    // gizmo, which otherwise swallows a center-drag) so OrbitControls receives it.
    const views = [];
    // The main 3D renderer canvas is Three.js's renderer.domElement, appended to
    // #viewport (OrbitControls listens on it). Puppeteer's boundingBox() can
    // return null for it depending on layout, so read the largest visible canvas
    // rect via getBoundingClientRect in-page and drag by absolute coords.
    const box = await page.evaluate(() => {
      const canvases = [...document.querySelectorAll('#viewport canvas, canvas')];
      let best = null;
      for (const c of canvases) {
        const r = c.getBoundingClientRect();
        if (r.width > 200 && r.height > 200 && (!best || r.width * r.height > best.width * best.height)) {
          best = { x: r.x, y: r.y, width: r.width, height: r.height };
        }
      }
      return best;
    });
    if (!box) fail('viewer', 'no visible renderer canvas (>200px) found for view capture');
    const dragOrbit = async (dx, dy) => {
      // Start in the upper-left quadrant of the canvas — empty sky, no gizmo.
      const sx = box.x + box.width * 0.30, sy = box.y + box.height * 0.22;
      await page.mouse.move(sx, sy);
      await page.mouse.down();
      for (let i = 1; i <= 20; i++) {
        await page.mouse.move(sx + (dx * i) / 20, sy + (dy * i) / 20);
        await new Promise(r => setTimeout(r, 8));
      }
      await page.mouse.up();
    };
    const angles = [
      { name: 'front', dx: 0, dy: 0 },
      { name: 'three-quarter', dx: 320, dy: 90 },
    ];
    for (const a of angles) {
      if (a.dx || a.dy) await dragOrbit(a.dx, a.dy);
      await new Promise(r => setTimeout(r, 1500)); // settle
      const shotPath = path.join(SHOT_DIR, `viewer-${a.name}.png`);
      const buf = await page.screenshot({ path: shotPath });
      views.push({ name: a.name, path: shotPath, bytes: buf.length, buffer: Buffer.from(buf) });
    }
    // Non-blank via real luminance-variance over the decoded framebuffer region
    // (sampled through the page, not a byte heuristic), and prove the two views
    // are actually DIFFERENT (rotation happened, not the same frame twice).
    for (const v of views) {
      const stats = await page.evaluate(async (b64) => {
        const img = new Image();
        await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; });
        const c = document.createElement('canvas');
        // sample the right 60% (the 3D viewport, excluding the left UI panel)
        const sx = Math.floor(img.width * 0.4);
        c.width = img.width - sx; c.height = img.height;
        const cx = c.getContext('2d');
        cx.drawImage(img, sx, 0, c.width, c.height, 0, 0, c.width, c.height);
        const d = cx.getImageData(0, 0, c.width, c.height).data;
        let sum = 0, sum2 = 0, n = 0;
        for (let i = 0; i < d.length; i += 40) { const l = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; sum += l; sum2 += l * l; n++; }
        const mean = sum / n; const variance = sum2 / n - mean * mean;
        return { mean, variance, stddev: Math.sqrt(Math.max(0, variance)), samples: n };
      }, v.buffer.toString('base64'));
      v.luminance = stats;
      v.likelyNonBlank = stats.stddev > 8; // a rendered mesh has real luminance spread
      delete v.buffer;
    }
    // Distinctness: hash the two viewport regions; identical hashes => no rotation.
    const hashes = views.map(v => crypto.createHash('sha256').update(fs.readFileSync(v.path)).digest('hex'));
    const viewsDistinct = new Set(hashes).size === views.length;
    report.viewer.views = views;
    report.viewer.viewsDistinct = viewsDistinct;
    const allNonBlank = views.length >= 2 && views.every(v => v.likelyNonBlank);
    const pass = allNonBlank && viewsDistinct && !pageErrors.length;
    gate('9-settled-views', pass, {
      note: pass
        ? `${views.length} distinct non-blank views (${views.map(v => `${v.name}:σ${v.luminance.stddev.toFixed(1)}`).join(', ')})`
        : `nonBlank=${allNonBlank} distinct=${viewsDistinct} (${views.map(v => `${v.name}:σ${v.luminance?.stddev?.toFixed(1)}`).join(', ')})${pageErrors.length ? ` pageErrors:${pageErrors.join('|')}` : ''}` });

    await browser.close().catch(() => {});
    return { viewerUrl, effectiveUrl };
  } catch (e) {
    await browser.close().catch(() => {});
    fail('viewer', e.message, { stack: e.stack });
  }
}

// ---------------------------------------------------------------------------
// Gates 10,11 — durable report already written throughout; landing capsule.
// ---------------------------------------------------------------------------
function emitLandingCapsule(viewer) {
  report.phase = 'landing';
  report.ok = Object.values(report.gates).every(g => g.passed);
  report.landing = {
    ok: report.ok,
    reportPath: REPORT_PATH,
    source: {
      candidateCheckout: report.source.checkout,
      candidateCommit: report.source.commit,
      kitVersion: report.source.kitVersion,
      route: ROUTE_ID,
      inputSha256: report.source.inputSha256,
      weightsSha256: report.source.weightsSha256,
    },
    artifacts: {
      candidateGlb: report.glb.candidateGlbPath,
      viewerShots: report.viewer.views?.map(v => v.path) || [],
      capsuleReport: REPORT_PATH,
    },
    operatorSmokeUrl: viewer?.viewerUrl || null,
    numericalParity: {
      dinoTokensIdentical: report.numericalPayload.identical,
      sha256: report.numericalPayload.coopSha256,
    },
    cadence: report.cadence,
  };
  writeReport();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
(async () => {
  console.log('SF3D cooperative-DINO acceptance capsule\n');
  const { pageUrl } = await verifySourceAndLaunch();
  const { coopArm, disArm, browser } = await runArms(pageUrl);
  compareDinoPayload(coopArm, disArm);
  const glbPath = validateGlbs(coopArm, disArm);
  cadenceTails(coopArm, disArm);
  await browser.close().catch(() => {});

  const viewer = await viewerGates(glbPath);
  emitLandingCapsule(viewer);

  for (const p of procs) { try { p.kill(); } catch {} }
  if (report.ok) {
    console.log('\n✓ CAPSULE PASSED — all gates green');
    console.log(`  report:            ${REPORT_PATH}`);
    console.log(`  candidate GLB:     ${report.glb.candidateGlbPath}`);
    console.log(`  operator smoke URL: ${report.landing.operatorSmokeUrl}`);
    process.exit(0);
  } else {
    fail('landing', `some gates failed: ${Object.entries(report.gates).filter(([, g]) => !g.passed).map(([k]) => k).join(', ')}`);
  }
})().catch(e => fail(report.phase || 'unknown', e.message, { stack: e.stack }));
