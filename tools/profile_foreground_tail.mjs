#!/usr/bin/env node
/**
 * Foreground-tail stage attribution profiler.
 *
 * Per Wake's continuation directive: before choosing the next cooperative
 * boundary, prove by exact local profiling WHICH monolithic SF3D stage dominates
 * the ~258ms full-pipeline rAF max/p99 foreground tail. Stage total-time is NOT
 * the answer — a stage that runs long but yields often is cheap on the
 * foreground; a stage that blocks the main thread in one contiguous chunk owns a
 * big frame gap even if its total time is modest.
 *
 * Method: run the real full pipeline in the browser with (a) a requestAnimation
 * Frame probe recording every inter-frame gap with its start/end timestamps, and
 * (b) absolute-timestamp stage spans from full_pipeline/inference on the same
 * performance.now() clock. Attribute each gap to the stage whose span contains
 * the gap's midpoint. Report per-stage: gap count, summed gap time, and the
 * single largest gap — so the stage that OWNS the max/p99 tail is named.
 *
 * Usage: node tools/profile_foreground_tail.mjs [--image PATH] [--report PATH]
 */
import puppeteer from 'puppeteer-core';
import path from 'node:path';
import fs from 'node:fs';
import net from 'node:net';
import { spawn } from 'node:child_process';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const argVal = (f, d) => { const i = process.argv.indexOf(f); return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d; };
const IMAGE = path.resolve(argVal('--image', path.join(process.env.HOME, '.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png')));
const REPORT_PATH = path.resolve(argVal('--report', '/tmp/sf3d-foreground-tail-profile.json'));

function allocatePort() {
  return new Promise((res, rej) => { const s = net.createServer(); s.unref(); s.on('error', rej); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); }); });
}

const procs = [];
const cleanup = () => { for (const p of procs) { try { p.kill(); } catch {} } };

(async () => {
  const port = await allocatePort();
  const vite = spawn('npx', ['vite', '--port', String(port), '--strictPort'], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(vite);
  await new Promise((res, rej) => { const to = setTimeout(() => rej(new Error('vite timeout')), 40000); vite.stdout.on('data', d => { if (/Local:|ready/.test(d.toString())) { clearTimeout(to); res(); } }); vite.on('error', e => { clearTimeout(to); rej(e); }); });

  const browser = await puppeteer.launch({ executablePath: CHROME_PATH, headless: false, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
  const page = await browser.newPage();
  page.on('pageerror', e => console.error('[pageerror]', e.message));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded', timeout: 30000 });

  const start = Date.now();
  while (Date.now() - start < 180000) {
    const s = await page.$eval('#status', el => el.textContent).catch(() => '');
    if (s.includes('Ready')) break;
    await new Promise(r => setTimeout(r, 500));
  }

  const imageB64 = fs.readFileSync(IMAGE).toString('base64');
  await page.evaluate(async (b64) => {
    await new Promise((res, rej) => { const img = new Image(); img.onload = () => { window._prof_image = img; res(); }; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; });
  }, imageB64);

  // Run the full pipeline with rAF gap probe + absolute stage spans.
  const result = await page.evaluate(async () => {
    const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
    const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines;
    const img = window._prof_image;

    // rAF probe: record every inter-frame gap with absolute start/end.
    const frames = [];
    let on = true, lastT = null;
    const tick = (t) => { if (lastT != null) frames.push({ start: lastT, end: t, gap: t - lastT }); lastT = t; if (on) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);

    // Legacy DINO (monolithic) so we profile the CURRENT foreground-hostility
    // surface with all stages monolithic — the honest baseline to attribute.
    const out = await runFullPipelineToGlb(device, pipelines, weights, img, { cooperativeDino: false });
    on = false;
    await new Promise(r => setTimeout(r, 100));

    return { frames, stageSpans: out.stageSpans, stageTimings: out.stageTimings, totalMs: out.totalMs, numVertices: out.numVertices, numFaces: out.numFaces };
  });

  await browser.close().catch(() => {});
  cleanup();

  // Attribute each gap to the stage span containing its midpoint.
  const spans = result.stageSpans || [];
  const attribute = (mid) => {
    const hit = spans.find(s => mid >= s.start && mid < s.end);
    return hit ? hit.name : (mid < (spans[0]?.start ?? Infinity) ? 'before-pipeline' : 'after-pipeline');
  };
  const perStage = {};
  for (const f of result.frames) {
    const mid = (f.start + f.end) / 2;
    const name = attribute(mid);
    const e = perStage[name] || (perStage[name] = { name, gapCount: 0, summedGapMs: 0, maxGapMs: 0, gapsOver33: 0, gapsOver100: 0 });
    e.gapCount++; e.summedGapMs += f.gap; e.maxGapMs = Math.max(e.maxGapMs, f.gap);
    if (f.gap > 33.3) e.gapsOver33++; if (f.gap > 100) e.gapsOver100++;
  }
  const allGaps = result.frames.map(f => f.gap).sort((a, b) => a - b);
  const pct = (p) => allGaps.length ? allGaps[Math.min(allGaps.length - 1, Math.ceil(p / 100 * allGaps.length) - 1)] : null;

  // Rank stages by max gap (foreground-hostility), then by summed gap.
  const ranked = Object.values(perStage).sort((a, b) => b.maxGapMs - a.maxGapMs);
  const stageDurations = Object.fromEntries((spans).map(s => [s.name, +(s.end - s.start).toFixed(1)]));

  const report = {
    schema: 'sf3d.foreground-tail-profile.v0',
    generatedFor: 'wake continuation: attribute the ~258ms rAF tail to a stage',
    input: IMAGE,
    mesh: { numVertices: result.numVertices, numFaces: result.numFaces },
    totalMs: +result.totalMs.toFixed(1),
    overall: { frames: result.frames.length, p50: pct(50), p95: pct(95), p99: pct(99), max: allGaps.at(-1) },
    stageDurationsMs: stageDurations,
    perStageTailAttribution: ranked.map(s => ({ ...s, summedGapMs: +s.summedGapMs.toFixed(1), maxGapMs: +s.maxGapMs.toFixed(1) })),
    dominantStage: ranked[0]?.name || null,
    verdict: ranked[0] ? `${ranked[0].name} owns the largest single frame gap (${ranked[0].maxGapMs.toFixed(1)}ms)` : 'no gaps recorded',
  };
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));

  console.log('\n=== Foreground-tail stage attribution ===');
  console.log(`overall: frames=${report.overall.frames} p95=${report.overall.p95?.toFixed(1)} p99=${report.overall.p99?.toFixed(1)} max=${report.overall.max?.toFixed(1)}ms`);
  console.log('per-stage (ranked by max single gap):');
  for (const s of report.perStageTailAttribution) {
    console.log(`  ${s.name.padEnd(22)} maxGap=${s.maxGapMs.toFixed(1).padStart(7)}ms  summed=${s.summedGapMs.toFixed(0).padStart(6)}ms  >33ms×${s.gapsOver33}  >100ms×${s.gapsOver100}  (stageDur=${stageDurations[s.name] ?? '?'}ms)`);
  }
  console.log(`\nDOMINANT: ${report.verdict}`);
  console.log(`report: ${REPORT_PATH}`);
  process.exit(0);
})().catch(e => { console.error('profiler failed:', e.message); cleanup(); process.exit(1); });
