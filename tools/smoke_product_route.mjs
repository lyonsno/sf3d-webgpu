#!/usr/bin/env node
/**
 * smoke_product_route.mjs — the product-route foreground-liveness witness.
 *
 * Runs the exact route main.js runs (runFullPipelineToGlb + product_route
 * options) in real Chrome WebGPU, with:
 *   - a requestAnimationFrame probe recording every inter-frame interval,
 *     scoped to the inference window;
 *   - absolute stage spans so each frame gap is attributed to a stage;
 *   - optionally (--contend) a same-page WebGPU contender on a second device
 *     that submits compute work continuously, the way the kit's SHARP proof
 *     was measured, so "smooth" means smooth while sharing the GPU;
 *   - effective-route receipts: which CPU phases ran on workers, which
 *     cooperative mechanisms settled (kit reports), bounded-prefix validation
 *     for the post-processor, and the canonical GLB hash.
 * The report is assembled and judged by the pure module
 * tools/product_route_witness_report.mjs (deterministically falsifiable).
 * A failure at any phase still writes a report naming the phase.
 *
 * Usage:
 *   node tools/smoke_product_route.mjs [--arm product-default|no-workers|workers-only|monolithic|'{json overrides}']
 *     [--contend] [--image P] [--report P] [--expected-glb-sha SHA|none]
 *     [--max-gap-budget-ms N] [--allow-dirty] [--label TEXT]
 */
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import net from 'node:net';
import { spawn, execSync } from 'node:child_process';
import {
  CANONICAL_DEMO_CHAIR_GLB_SHA256,
  acceptProductRouteWitness,
  assembleProductRouteWitness,
} from './product_route_witness_report.mjs';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const argv = process.argv.slice(2);
const argVal = (flag, fallback) => { const i = argv.indexOf(flag); return i >= 0 && argv[i + 1] != null ? argv[i + 1] : fallback; };
const hasFlag = (flag) => argv.includes(flag);

const ARM = argVal('--arm', 'product-default');
const CONTEND = hasFlag('--contend');
const IMAGE = path.resolve(argVal('--image', path.join(REPO, 'public/demo_chair.png')));
const LABEL = argVal('--label', '');
const ARM_NAME = `${ARM.startsWith('{') ? 'custom' : ARM}${CONTEND ? '+contend' : ''}`;
const REPORT_PATH = path.resolve(argVal('--report', `/tmp/sf3d-product-route-witness-${ARM_NAME}.json`));
const expectedShaArg = argVal('--expected-glb-sha', CANONICAL_DEMO_CHAIR_GLB_SHA256);
const EXPECTED_GLB_SHA = expectedShaArg === 'none' ? null : expectedShaArg;
const MAX_GAP_BUDGET_MS = argVal('--max-gap-budget-ms', null) == null ? null : Number(argVal('--max-gap-budget-ms'));
const ALLOW_DIRTY = hasFlag('--allow-dirty');

const KNOWN_ARMS = ['product-default', 'no-workers', 'workers-only', 'monolithic'];
if (!KNOWN_ARMS.includes(ARM) && !ARM.startsWith('{')) {
  console.error(`unknown --arm ${ARM}; expected one of ${KNOWN_ARMS.join(', ')} or a JSON overrides object`);
  process.exit(2);
}
if (!fs.existsSync(IMAGE)) { console.error(`image not found: ${IMAGE}`); process.exit(2); }

function writeFailure(phase, error, partial = {}) {
  const failure = {
    schema: 'sf3d.product-route-witness-failure.v0',
    ok: false,
    arm: ARM_NAME,
    failurePhase: phase,
    error: { message: error?.message || String(error), stack: error?.stack || null },
    partial,
    generatedAt: new Date().toISOString(),
  };
  fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  fs.writeFileSync(REPORT_PATH, JSON.stringify(failure, null, 2));
  console.error(`\nWITNESS FAILED at ${phase}: ${failure.error.message}\nreport: ${REPORT_PATH}`);
}

function allocatePort() {
  return new Promise((res, rej) => {
    const s = net.createServer(); s.unref(); s.on('error', rej);
    s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); });
  });
}

// --- Source identity (effective, not requested) ---
const commit = execSync('git rev-parse HEAD', { cwd: REPO }).toString().trim();
const dirty = execSync('git status --porcelain', { cwd: REPO }).toString().trim().length > 0;
if (dirty && !ALLOW_DIRTY) {
  writeFailure('source-identity', new Error('worktree is dirty; commit first or pass --allow-dirty (the report will still record dirty=true)'));
  process.exit(1);
}
const kitVersion = JSON.parse(fs.readFileSync(path.join(REPO, 'node_modules/@kaminos/webgpu-inference-kit/package.json'), 'utf8')).version;
const source = { commit, dirty, kitVersion, hostname: os.hostname(), node: process.version, label: LABEL || null };

const procs = [];
const cleanup = () => { for (const p of procs) { try { p.kill(); } catch { /* gone */ } } };
let browser = null;

try {
  // --- Serve the checkout ---
  const port = await allocatePort();
  const vite = spawn('npx', ['vite', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(vite);
  await new Promise((res, rej) => {
    const to = setTimeout(() => rej(new Error('vite startup timeout')), 40000);
    vite.stdout.on('data', d => { if (/Local:|ready/.test(d.toString())) { clearTimeout(to); res(); } });
    vite.on('error', e => { clearTimeout(to); rej(e); });
  });

  browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: false,
    args: [
      '--enable-unsafe-webgpu', '--use-angle=metal',
      '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding',
      '--no-first-run', '--no-default-browser-check',
    ],
  });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', e => { pageErrors.push(e.message); console.error('[pageerror]', e.message); });
  page.on('console', m => { if (m.type() === 'error') console.error('[console.error]', m.text().slice(0, 300)); });

  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded', timeout: 30000 });
  const start = Date.now();
  let ready = false;
  while (Date.now() - start < 240000) {
    const s = await page.$eval('#status', el => el.textContent).catch(() => '');
    if (s.includes('Ready')) { ready = true; break; }
    if (s.startsWith('Error:')) throw new Error(`app failed to initialize: ${s}`);
    await new Promise(r => setTimeout(r, 500));
  }
  if (!ready) throw new Error('app did not reach Ready within 240s');

  const imageB64 = fs.readFileSync(IMAGE).toString('base64');
  await page.evaluate(async (b64) => {
    await new Promise((res, rej) => { const img = new Image(); img.onload = () => { window._witnessImage = img; res(); }; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; });
  }, imageB64);

  // --- The witnessed run ---
  const raw = await page.evaluate(async ({ armSpec, contend }) => {
    const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
    const {
      createProductRouteOptions, createProductRouteWorkers, describeProductRouteOptions, terminateProductRouteWorkers,
    } = await import('/src/lib/product_route.js');
    const { projectCooperativeReport } = await import('/tools/product_route_witness_report.mjs');
    const { acceptBoundedPrefixArm, expectedChannelDutyCount } = await import('/tools/bounded_prefix_acceptance.mjs');

    const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines;
    const img = window._witnessImage;
    if (!device || !weights || !pipelines || !img) throw new Error('page state missing (device/weights/pipelines/image)');
    if (document.visibilityState !== 'visible') throw new Error(`page visibility ${document.visibilityState} at start`);

    // Arm → options.
    let workers = null;
    let options;
    if (armSpec === 'monolithic') {
      options = Object.freeze({ cooperativeDino: false });
    } else if (armSpec === 'no-workers') {
      options = createProductRouteOptions({});
    } else if (armSpec === 'workers-only') {
      workers = createProductRouteWorkers();
      options = createProductRouteOptions({ workers, overrides: {
        cooperativeDino: false, cooperativeTwoStream: false, cooperativePostProcessor: false, cooperativeBake: false, decoderArena: false,
      } });
    } else {
      workers = createProductRouteWorkers();
      const overrides = armSpec.startsWith('{') ? JSON.parse(armSpec) : {};
      options = createProductRouteOptions({ workers, overrides });
    }

    // rAF probe.
    const frames = [];
    let on = true, lastT = null;
    const tick = (t) => { if (lastT != null) frames.push({ start: lastT, end: t }); lastT = t; if (on) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);

    // Same-page WebGPU contender on a second device (SHARP contention-witness shape).
    const contender = { enabled: contend, submitted: 0, completed: 0, errors: [] };
    let contenderDone = Promise.resolve();
    if (contend) {
      contenderDone = (async () => {
        try {
          const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
          if (!adapter) throw new Error('no contender adapter');
          const cdev = await adapter.requestDevice();
          const module = cdev.createShaderModule({ code: `
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
              var x = data[id.x];
              for (var i: u32 = 0u; i < 64u; i = i + 1u) { x = (x * 1.0001221) + f32((i & 7u) + 1u) * 0.00003125; }
              data[id.x] = x;
            }` });
          const pipeline = cdev.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
          const buffer = cdev.createBuffer({ size: 65536 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
          const bindGroup = cdev.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] });
          while (on) {
            const enc = cdev.createCommandEncoder();
            const pass = enc.beginComputePass();
            pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(1024); pass.end();
            cdev.queue.submit([enc.finish()]);
            contender.submitted += 1;
            await cdev.queue.onSubmittedWorkDone();
            contender.completed += 1;
            await new Promise(r => setTimeout(r, 0));
          }
          buffer.destroy();
        } catch (e) { contender.errors.push(e?.message || String(e)); }
      })();
      await new Promise(r => setTimeout(r, 200)); // let the contender reach steady state
    }

    const progress = [];
    const startMs = performance.now();
    let out, runError = null;
    try {
      out = await runFullPipelineToGlb(device, pipelines, weights, img, options, (m) => progress.push(String(m)));
    } catch (e) { runError = e; }
    const endMs = performance.now();
    on = false;
    await new Promise(r => setTimeout(r, 120));
    await contenderDone;
    const visibility = document.visibilityState;
    if (workers) terminateProductRouteWorkers(workers);
    if (runError) throw new Error(`route failed: ${runError.message}`);

    const digest = await crypto.subtle.digest('SHA-256', out.glb);
    const glbSha256 = [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');

    const cooperative = {};
    for (const [k, r] of Object.entries(out.cooperativeReports || {})) cooperative[k] = projectCooperativeReport(r);
    const cooperativeValidations = {};
    const ppReport = out.cooperativeReports?.['post-processor'];
    if (ppReport && options.cooperativePostProcessor && options.postProcessorDutyGranularity === 'channel-range'
        && options.postProcessorCompletionPolicy === 'bounded-prefix') {
      const v = acceptBoundedPrefixArm({
        report: ppReport, progressMessages: progress,
        expectedGpuDutyCount: expectedChannelDutyCount(options.postProcessorChannelsPerDuty),
        maxInFlightGpuDuties: options.postProcessorMaxInFlightGpuDuties,
      });
      cooperativeValidations['post-processor'] = { ok: v.ok, errors: [...v.errors], progressHonest: v.progressHonest };
    }
    const bakeTel = out.cooperativeReports?.['texture-bake']?.textureBakeTelemetry;
    const materializationOffloaded = bakeTel?.materializationOffloaded ?? bakeTel?.phases?.materializationOffloaded ?? null;

    const adapter = await navigator.gpu.requestAdapter();
    const info = adapter?.info || {};
    return {
      frames, stageSpans: out.stageSpans, stageTimings: out.stageTimings, totalMs: out.totalMs,
      inferenceWindow: { startMs, endMs }, visibility,
      offloads: out.offloads, requested: describeProductRouteOptions(options),
      cooperative, cooperativeValidations, materializationOffloaded,
      output: { glbSha256, glbBytes: out.glb.byteLength, numVertices: out.numVertices, numFaces: out.numFaces, roughness: out.roughness, metallic: out.metallic },
      contender, progressCount: progress.length,
      backend: { vendor: info.vendor ?? null, architecture: info.architecture ?? null, device: info.device ?? null, description: info.description ?? null, userAgent: navigator.userAgent },
    };
  }, { armSpec: ARM, contend: CONTEND });

  if (pageErrors.length) throw new Error(`page errors during run: ${pageErrors.join(' | ')}`);

  // --- Assemble + judge ---
  const report = assembleProductRouteWitness({
    arm: ARM_NAME,
    source: { ...source, backend: raw.backend },
    requestedOptions: raw.requested,
    offloads: raw.offloads,
    cooperativeReports: raw.cooperative,           // already projected in-page
    cooperativeValidations: raw.cooperativeValidations,
    materializationOffloaded: raw.materializationOffloaded,
    frames: raw.frames, stageSpans: raw.stageSpans, inferenceWindow: raw.inferenceWindow, visibility: raw.visibility,
    output: { ...raw.output, expectedGlbSha256: EXPECTED_GLB_SHA },
    contender: raw.contender, stageTimings: raw.stageTimings, totalMs: raw.totalMs,
  });
  const verdict = acceptProductRouteWitness(report, {
    expectedGlbSha: EXPECTED_GLB_SHA, requireContender: CONTEND, maxGapBudgetMs: MAX_GAP_BUDGET_MS,
  });
  const durable = { ...report, verdict: { ok: verdict.ok, errors: [...verdict.errors] } };
  fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  fs.writeFileSync(REPORT_PATH, JSON.stringify(durable, null, 2));

  // --- Summary ---
  const wr = report.cadence.wholeRoute;
  console.log(`\n=== SF3D product-route witness: ${ARM_NAME} ===`);
  console.log(`source ${commit.slice(0, 10)} dirty=${dirty} kit=${kitVersion} backend=${raw.backend.vendor}/${raw.backend.architecture}`);
  console.log(`route wall ${report.totalMs}ms  frames=${wr.frameIntervalCount}  p50/p95/p99/max = ${wr.p50Ms}/${wr.p95Ms}/${wr.p99Ms}/${wr.maxMs}ms  >16.7:${wr.over16_7} >33.3:${wr.over33_3} >100:${wr.over100}`);
  if (raw.contender.enabled) console.log(`contender: submitted=${raw.contender.submitted} completed=${raw.contender.completed} errors=${raw.contender.errors.length}`);
  console.log(`GLB sha ${report.output.glbSha256.slice(0, 12)}… (${report.output.glbBytes} B, ${report.output.numVertices}v/${report.output.numFaces}f) expected ${EXPECTED_GLB_SHA ? EXPECTED_GLB_SHA.slice(0, 12) + '…' : 'none'}`);
  console.log(`offloads: ${Object.entries(report.effective.offloads).map(([k, v]) => `${k}=${v}`).join(' ')}`);
  console.log(`cooperative: ${Object.entries(report.effective.cooperative).map(([k, c]) => `${k}=${c.status}(${c.progress?.completedItems ?? '?'}/${c.progress?.totalItems ?? '?'} duties${c.completionPolicy === 'bounded-prefix' ? `, bounded depth ${c.maxObservedInFlightGpuDuties}` : ''})`).join(' ') || 'none'}`);
  console.log('stages by max gap:');
  for (const name of report.cadence.rankedByMaxGap.slice(0, 8)) {
    const b = report.cadence.byStage[name];
    console.log(`  ${name.padEnd(24)} maxGap=${String(b.maxGapMs).padStart(7)}ms  >33.3×${b.over33_3}  summed=${String(b.summedGapMs).padStart(7)}ms  dur=${b.durationMs ?? '?'}ms`);
  }
  console.log(`report: ${REPORT_PATH}`);
  if (!verdict.ok) {
    console.error(`\nWITNESS REJECTED:\n  - ${verdict.errors.join('\n  - ')}`);
    process.exitCode = 1;
  } else {
    console.log('\nWITNESS ACCEPTED');
  }
} catch (err) {
  writeFailure('witness', err);
  process.exitCode = 1;
} finally {
  if (browser) await browser.close().catch(() => {});
  cleanup();
}
