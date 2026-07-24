#!/usr/bin/env node
/**
 * Cooperative DINO A/B smoke — first-consumer exercise of the Kaminos WebGPU
 * Inference Kit cooperative porting spine in the real sf3d.image-to-mesh
 * .webgpu-local.v0 browser route.
 *
 * Runs the full pipeline three times against ONE loaded model, varying only the
 * DINO encoder execution:
 *   1. legacy      — single command buffer, one submit (pre-cooperative path)
 *   2. coop-off    — cooperative facade, schedulingMode 'disabled' (A/B control)
 *   3. coop-on     — cooperative facade, schedulingMode 'cooperative' (per-block
 *                    submit + browser yield + per-duty queue fence)
 *
 * Reports per arm: mesh vertex/face counts, density stats, DINO wall time, and
 * the kit cooperative execution report (queueCompletionAuthority, boundary
 * range count, terminal progress). Asserts mesh parity across all three arms so
 * a numerical regression from the cooperative wiring fails loud.
 *
 * This is a witness harness, not just a screenshot: it records effective route
 * identity, the cooperative report schema, and cross-arm parity, and fails
 * closed if any arm errors or drifts.
 *
 * Usage: node tools/smoke_cooperative_dino_ab.mjs [--image path] [--chunk N] [--report path]
 */
import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import http from 'http';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const DEFAULT_IMAGE = path.join(process.env.HOME, '.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png');
const REPORT_PATH = argVal('--report', '/tmp/sf3d-cooperative-dino-ab-report.json');
const CHUNK_BLOCKS = Number(argVal('--chunk', '1')) || 1;
const imagePath = argVal('--image', DEFAULT_IMAGE);

function argVal(flag, dflt) {
  const i = process.argv.indexOf(flag);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}

async function ensureVite() {
  for (const port of [5177, 5176, 5178]) {
    try {
      await new Promise((resolve, reject) => {
        const req = http.get(`http://localhost:${port}/`, (res) => { res.resume(); resolve(); });
        req.on('error', reject);
        req.setTimeout(1000, () => { req.destroy(); reject(new Error('timeout')); });
      });
      return { port, proc: null };
    } catch {}
  }
  const proc = spawn('npx', ['vite', '--port', '5177'], {
    cwd: path.dirname(new URL(import.meta.url).pathname).replace(/\/tools$/, ''),
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('vite startup timeout')), 30000);
    proc.stdout.on('data', (d) => {
      if (/ready|Local:/.test(d.toString())) { clearTimeout(timeout); resolve(); }
    });
    proc.on('error', (e) => { clearTimeout(timeout); reject(e); });
  });
  return { port: 5177, proc };
}

let viteProc = null;
const fail = (msg, extra = {}) => {
  const out = { ok: false, phase: 'cooperative-dino-ab', error: msg, ...extra };
  fs.writeFileSync(REPORT_PATH, JSON.stringify(out, null, 2));
  console.error(`\nSMOKE FAILED: ${msg}`);
  if (viteProc) viteProc.kill();
  process.exit(1);
};

const { port, proc } = await ensureVite();
viteProc = proc;
const PAGE_URL = `http://localhost:${port}/`;

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=metal'],
});
const page = await browser.newPage();
const pageErrors = [];
page.on('pageerror', (err) => pageErrors.push(err.message));
page.on('console', (msg) => {
  const t = msg.text();
  if (msg.type() === 'error') console.log(`[ERR] ${t.slice(0, 200)}`);
  else if (/DINOv2 blocks|backbone|Mesh extracted/.test(t)) console.log(`[LOG] ${t.slice(0, 160)}`);
});

try {
  console.log(`Navigating ${PAGE_URL} ...`);
  await page.goto(PAGE_URL, { waitUntil: 'domcontentloaded', timeout: 30000 });

  // Wait for weights + pipelines (status "Ready")
  console.log('Waiting for model load (Ready)...');
  const start = Date.now();
  while (Date.now() - start < 180000) {
    const s = await page.$eval('#status', el => el.textContent).catch(() => '');
    if (s.includes('Ready')) break;
    if (s.startsWith('Error:')) fail(`page reached error status before ready: ${s}`);
    await new Promise(r => setTimeout(r, 500));
  }

  // Load the input image into the page as an HTMLImageElement.
  const imageB64 = fs.readFileSync(imagePath).toString('base64');
  await page.evaluate((b64) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => { window._smoke_image = img; resolve(); };
      img.onerror = reject;
      img.src = `data:image/png;base64,${b64}`;
    });
  }, imageB64);

  // Run the three arms in-page, reusing the one loaded model.
  const result = await page.evaluate(async (chunkBlocks) => {
    const { runInference } = await import('/src/lib/inference.js');
    const device = window._sf3d_device;
    const weights = window._sf3d_weights;
    const pipelines = window._sf3d_pipelines;
    if (!device || !weights || !pipelines) throw new Error('model globals not present');
    const img = window._smoke_image;

    async function arm(label, options) {
      const t0 = performance.now();
      const r = await runInference(device, pipelines, weights, img, () => {}, options);
      const t1 = performance.now();
      const coop = r._cooperativeReports?.['dinov2-tokenizer'] || null;
      // Density summary from _sdf (density - threshold), add threshold back.
      const thr = r._isosurfaceThreshold ?? 0;
      let dMin = Infinity, dMax = -Infinity, dSum = 0, dPos = 0, n = 0;
      if (r._sdf) {
        for (let i = 0; i < r._sdf.length; i++) {
          const d = r._sdf[i] + thr;
          if (Number.isNaN(d)) continue;
          if (d < dMin) dMin = d; if (d > dMax) dMax = d;
          dSum += d; if (d > thr) dPos++; n++;
        }
      }
      return {
        label,
        numVertices: r.numVertices,
        numFaces: r.numFaces,
        density: { min: dMin, max: dMax, mean: n ? dSum / n : null, insideCount: dPos },
        dinoWallMs: coop ? coop.durationMs : (r._stageTimings?.['dinov2-tokenizer'] ?? null),
        fullWallMs: t1 - t0,
        cooperative: coop ? {
          schema: coop.schema,
          schedulingMode: coop.schedulingMode,
          status: coop.status,
          queueCompletionAuthority: coop.queueCompletionAuthority,
          boundaryRangeCount: coop.boundaries?.[0]?.actualRangeCount ?? null,
          completedItems: coop.progress?.completedItems ?? null,
          totalItems: coop.progress?.totalItems ?? null,
          percent: coop.progress?.percent ?? null,
        } : null,
      };
    }

    const legacy = await arm('legacy', { cooperativeDino: false });
    const coopOff = await arm('coop-off', { cooperativeDino: true, dinoSchedulingMode: 'disabled', dinoChunkBlocks: chunkBlocks });
    const coopOn = await arm('coop-on', { cooperativeDino: true, dinoSchedulingMode: 'cooperative', dinoChunkBlocks: chunkBlocks });

    return { legacy, coopOff, coopOn, routeReceipt: window._lastRouteReceipt || null };
  }, CHUNK_BLOCKS);

  if (pageErrors.length) fail(`page errors during A/B: ${pageErrors.join(' | ')}`, { result });

  // --- Cross-arm parity assertions (fail loud on numerical regression) ---
  const arms = [result.legacy, result.coopOff, result.coopOn];
  const ref = result.legacy;
  const parityIssues = [];
  for (const a of arms) {
    if (a.numVertices !== ref.numVertices) parityIssues.push(`${a.label} vertices ${a.numVertices} != legacy ${ref.numVertices}`);
    if (a.numFaces !== ref.numFaces) parityIssues.push(`${a.label} faces ${a.numFaces} != legacy ${ref.numFaces}`);
    if (a.density.insideCount !== ref.density.insideCount) parityIssues.push(`${a.label} insideCount ${a.density.insideCount} != legacy ${ref.density.insideCount}`);
  }

  // --- Cooperative-report assertions ---
  const coopIssues = [];
  if (result.coopOn.cooperative?.queueCompletionAuthority !== 'per-gpu-duty-prefix-fence')
    coopIssues.push(`coop-on authority ${result.coopOn.cooperative?.queueCompletionAuthority}`);
  if (result.coopOff.cooperative?.queueCompletionAuthority !== 'one-terminal-prefix-fence')
    coopIssues.push(`coop-off authority ${result.coopOff.cooperative?.queueCompletionAuthority}`);
  const expectedRanges = Math.ceil(24 / CHUNK_BLOCKS);
  if (result.coopOn.cooperative?.boundaryRangeCount !== expectedRanges)
    coopIssues.push(`coop-on ranges ${result.coopOn.cooperative?.boundaryRangeCount} != ${expectedRanges}`);
  if (result.coopOn.cooperative?.completedItems !== 24 || result.coopOn.cooperative?.totalItems !== 24)
    coopIssues.push(`coop-on progress ${result.coopOn.cooperative?.completedItems}/${result.coopOn.cooperative?.totalItems}`);

  const report = {
    ok: parityIssues.length === 0 && coopIssues.length === 0,
    phase: 'cooperative-dino-ab',
    route: 'sf3d.image-to-mesh.webgpu-local.v0',
    effectiveRouteReceiptId: result.routeReceipt?.routeId ?? null,
    chunkBlocks: CHUNK_BLOCKS,
    image: imagePath,
    arms: { legacy: result.legacy, coopOff: result.coopOff, coopOn: result.coopOn },
    parityIssues,
    coopIssues,
    pageErrors,
  };
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));

  console.log('\n=== Cooperative DINO A/B ===');
  for (const a of arms) {
    console.log(`  ${a.label.padEnd(9)} v=${a.numVertices} f=${a.numFaces} inside=${a.density.insideCount} ` +
      `dinoWall=${a.dinoWallMs?.toFixed?.(1) ?? a.dinoWallMs}ms full=${a.fullWallMs.toFixed(0)}ms ` +
      `${a.cooperative ? `[${a.cooperative.schedulingMode} ${a.cooperative.queueCompletionAuthority} ranges=${a.cooperative.boundaryRangeCount}]` : ''}`);
  }
  if (!report.ok) fail(`A/B assertions failed: ${[...parityIssues, ...coopIssues].join('; ')}`, { reportPath: REPORT_PATH });

  console.log(`\nSMOKE PASSED — report: ${REPORT_PATH}`);
  await browser.close();
  if (viteProc) viteProc.kill();
  process.exit(0);
} catch (err) {
  await browser.close().catch(() => {});
  fail(err.message, { stack: err.stack });
}
