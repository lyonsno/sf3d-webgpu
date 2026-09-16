#!/usr/bin/env node
/**
 * smoke_live_flame_composition.mjs — witness for SF3D composed into the live
 * Kaminos app route (the `#composition_module_url=` seam, Kaminos branch
 * cc/slow-sf3d-live-flame-0916, entry /sf3d-elfinblue.html).
 *
 * Opens the entry page in real Chrome WebGPU, waits for the app's fire route
 * and the SF3D composition module to come up, triggers one SF3D run through
 * the HUD button, and records what the page itself measured: fire backend
 * status, frames observed during inference and their gap tail (the page's
 * requestAnimationFrame monitor — the independent liveness witness), the GLB
 * hash against the canonical demo-chair hash, the measured cooperative duty
 * counts, and the producer's identity. A failure at any phase still writes a
 * report naming the phase.
 *
 * Usage:
 *   node tools/smoke_live_flame_composition.mjs --url http://127.0.0.1:8095/sf3d-elfinblue.html
 *     [--report P] [--settle-ms 8000] [--timeout-ms 600000]
 * Device topology on this route is same-GPU-two-devices (the app does not
 * expose its device); the report says so.
 */
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const CANONICAL_DEMO_CHAIR_GLB_SHA256 = 'e1f70de3407df24d571bf68f70fac2b59373bdd948075a2387f1834e4faff8b7';
const argv = process.argv.slice(2);
const argVal = (flag, fallback) => { const i = argv.indexOf(flag); return i >= 0 && argv[i + 1] != null ? argv[i + 1] : fallback; };
const URL_ = argVal('--url', 'http://127.0.0.1:8095/sf3d-elfinblue.html');
const REPORT_PATH = path.resolve(argVal('--report', '/tmp/sf3d-live-flame-composition-witness.json'));
const SETTLE_MS = Number(argVal('--settle-ms', 8000));
const TIMEOUT_MS = Number(argVal('--timeout-ms', 900000));   // SF3D under the live flame runs several minutes

let phase = 'startup';
let browser = null;
const pageErrors = [];
const consoleErrors = [];
const events = [];           // crash / navigation / close events with timestamps
const hudSamples = [];       // periodic HUD text during the run
let lastScreenshot = null;
let baselineFrames = null;
const note = (kind, detail) => events.push({ atMs: Date.now(), kind, detail: String(detail ?? '').slice(0, 300) });
function writeReport(body) {
  fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  fs.writeFileSync(REPORT_PATH, JSON.stringify(body, null, 2));
}
function fail(error) {
  writeReport({ schema: 'sf3d.live-flame-composition-witness-failure.v0', ok: false, failurePhase: phase, url: URL_,
    error: { message: error?.message || String(error), stack: error?.stack || null }, events, hudSamples, lastScreenshot, baselineFrames,
    pageErrors, consoleErrors: consoleErrors.slice(0, 20), generatedAt: new Date().toISOString() });
  console.error(`\nCOMPOSITION WITNESS FAILED at ${phase}: ${error?.message || error}\nreport: ${REPORT_PATH}`);
}

try {
  phase = 'browser-launch';
  browser = await puppeteer.launch({
    executablePath: CHROME_PATH, headless: false,
    protocolTimeout: TIMEOUT_MS + 60000,   // puppeteer's default 180s protocol timeout aborted the run wait ("Waiting failed")
    args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding', '--no-first-run', '--no-default-browser-check', '--window-size=1400,1000'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 1000 });
  page.on('pageerror', e => pageErrors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text().slice(0, 300)); });
  page.on('error', e => note('page-crash', e.message));                 // renderer crash ("Page crashed!")
  page.on('framenavigated', f => { if (f === page.mainFrame()) note('navigated', f.url()); });
  page.on('close', () => note('page-closed', ''));
  browser.on('disconnected', () => note('browser-disconnected', ''));
  const hudSampler = setInterval(async () => {
    try {
      const sample = await page.evaluate(() => ({
        infer: document.getElementById('sf3d-infer')?.textContent ?? null,
        fire: document.getElementById('sf3d-fire')?.textContent ?? null,
        frames: window.__sf3dLiveFlame?.framesDuringInference ?? null,
        worst: window.__sf3dLiveFlame?.worstGapDuringInference ?? null,
        heap: performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1048576) : null,
      }));
      hudSamples.push({ atMs: Date.now(), ...sample });
    } catch (e) { hudSamples.push({ atMs: Date.now(), error: String(e.message).slice(0, 120) }); }
  }, 2000);
  const snap = async () => { try { lastScreenshot = REPORT_PATH.replace(/\.json$/, '') + '.png'; await page.screenshot({ path: lastScreenshot }); } catch { lastScreenshot = null; } };
  process.on('exit', () => clearInterval(hudSampler));

  phase = 'page-load';
  await page.goto(URL_, { waitUntil: 'domcontentloaded', timeout: 60000 });
  // The entry page meta-refreshes into the app route; wait for the composition module.
  phase = 'composition-mount';
  const t0 = Date.now();
  await page.waitForFunction(() => window.__sf3dLiveFlameReady === true, { timeout: TIMEOUT_MS, polling: 500 });
  const mountMs = Date.now() - t0;
  const fireStatusAtMount = await page.$eval('#sf3d-fire', el => el.textContent).catch(() => null);
  const weights = await page.$eval('#sf3d-weights', el => el.textContent).catch(() => null);

  phase = 'settle';
  await new Promise(r => setTimeout(r, SETTLE_MS));   // let the flame reach steady state before measuring
  baselineFrames = await page.evaluate(() => {
    const xs = [...window.__sf3dLiveFlame.frameIntervals].sort((a, b) => a - b);
    const pick = q => xs[Math.min(xs.length - 1, Math.floor(q * xs.length))];
    return { count: xs.length, p50Ms: pick(0.5), p95Ms: pick(0.95), maxMs: xs[xs.length - 1] };
  });
  const baseline = baselineFrames;

  phase = 'sf3d-run';
  await page.click('#sf3d-run');
  const runStart = Date.now();
  try {
    await page.waitForFunction(() => window.__sf3dLiveFlame.lastResult != null || window.__sf3dLiveFlame.lastError != null, { timeout: TIMEOUT_MS, polling: 500 });
  } catch (e) {
    await snap();
    throw new Error(`run wait aborted after ${Date.now() - runStart}ms: ${e.message}; last events: ${events.slice(-3).map(x => `${x.kind}:${x.detail}`).join(' | ') || 'none'}; last HUD: ${JSON.stringify(hudSamples.slice(-1)[0] ?? null)}`);
  }
  await snap();
  const raw = await page.evaluate(() => {
    const s = window.__sf3dLiveFlame;
    const r = s.lastResult;
    return {
      error: s.lastError,
      result: r ? { ...r, glb: undefined } : null,
      fireStatus: document.getElementById('sf3d-fire')?.textContent ?? null,
      volumeBackend: document.getElementById('volume-backend')?.textContent?.trim() ?? null,
      visibility: document.visibilityState,
      compositionRoute: window.__compositionRoute ?? null,
      producer: window.__sf3dProducer ? { kitVersion: window.__sf3dProducer.kitVersion, deviceInjected: window.__sf3dProducer.deviceInjected, resources: window.__sf3dProducer.resources, backend: window.__sf3dProducer.backend } : null,
    };
  });
  if (raw.error) throw new Error(`SF3D run failed: ${raw.error.message}`);
  const r = raw.result;

  phase = 'judge';
  const errors = [];
  if (r.glbSha256 !== CANONICAL_DEMO_CHAIR_GLB_SHA256) errors.push(`glb sha ${r.glbSha256} != canonical`);
  if (/error|unavailable/i.test(raw.fireStatus || '')) errors.push(`fire status during composition: ${raw.fireStatus}`);
  if (!(r.framesDuringInference > 100)) errors.push(`only ${r.framesDuringInference} frames observed during inference`);
  const expectedDuties = { 'dinov2-tokenizer': 24, 'two-stream-backbone': 2922, 'post-processor': 702, 'texture-bake': 61 };
  for (const [k, n] of Object.entries(expectedDuties)) if (r.duties[k] !== n) errors.push(`${k} submitted ${r.duties[k]} duties, expected ${n}`);
  for (const [k, st] of Object.entries(r.cooperativeStatuses)) if (st !== 'succeeded') errors.push(`${k} cooperative status ${st}`);
  if (r.receiptValidation?.ok !== true) errors.push('route receipt failed validation');
  if (raw.visibility !== 'visible') errors.push(`page visibility ${raw.visibility}`);
  if (pageErrors.length) errors.push(`page errors: ${pageErrors.slice(0, 3).join(' | ')}`);

  const report = {
    schema: 'sf3d.live-flame-composition-witness.v0',
    ok: errors.length === 0,
    errors,
    generatedAt: new Date().toISOString(),
    url: URL_,
    host: { hostname: os.hostname(), node: process.version },
    compositionRoute: raw.compositionRoute,
    deviceTopology: r.deviceTopology,
    producer: raw.producer,
    mount: { mountMs, weightsAtMount: weights, fireStatusAtMount },
    fire: { statusAfterRun: raw.fireStatus, volumeBackend: raw.volumeBackend },
    baselineFrames: baseline,
    events, hudSamples, screenshot: lastScreenshot,
    run: {
      runId: r.runId, wallMs: +r.wallMs.toFixed(1), harnessWallMs: Date.now() - runStart,
      glbSha256: r.glbSha256, canonical: r.canonical, glbBytes: r.glbBytes, numVertices: r.numVertices, numFaces: r.numFaces,
      duties: r.duties, cooperativeStatuses: r.cooperativeStatuses, offloads: r.offloads,
      framesDuringInference: r.framesDuringInference, inferenceGaps: r.inferenceGaps,
      foregroundOpportunityReport: r.foregroundOpportunityReport, identity: r.identity,
    },
    consoleErrors: consoleErrors.slice(0, 20),
  };
  writeReport(report);
  const g = r.inferenceGaps;
  console.log(`\n=== SF3D × live flame composition witness ===`);
  console.log(`fire: ${raw.fireStatus} | topology: ${r.deviceTopology} | kit ${raw.producer?.kitVersion}`);
  console.log(`baseline (flame only, ${baseline.count} frames): p50 ${baseline.p50Ms?.toFixed(1)} p95 ${baseline.p95Ms?.toFixed(1)} max ${baseline.maxMs?.toFixed(1)} ms`);
  console.log(`during SF3D (${r.framesDuringInference} frames, ${(r.wallMs / 1000).toFixed(1)}s): p50 ${g.p50Ms?.toFixed(1)} p95 ${g.p95Ms?.toFixed(1)} p99 ${g.p99Ms?.toFixed(1)} max ${g.maxMs?.toFixed(1)} ms  >33.3: ${g.over33_3}  >100: ${g.over100}`);
  console.log(`duties: ${Object.values(r.duties).join(' / ')} | GLB ${r.glbSha256.slice(0, 12)}… ${r.canonical ? '= canonical' : '≠ canonical'} (${r.numVertices}v/${r.numFaces}f)`);
  console.log(`report: ${REPORT_PATH}`);
  if (errors.length) { console.error(`\nWITNESS REJECTED:\n  - ${errors.join('\n  - ')}`); process.exitCode = 1; }
  else console.log('\nWITNESS ACCEPTED');
} catch (err) {
  fail(err);
  process.exitCode = 1;
} finally {
  if (browser) await browser.close().catch(() => {});
}
