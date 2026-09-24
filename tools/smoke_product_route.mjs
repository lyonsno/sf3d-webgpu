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
 *   - optionally (--contend-same-device) a host-frame contender on SF3D's OWN
 *     device, one frame per requestAnimationFrame, submitted through the
 *     producer's foreground-opportunity bridge (the kiln composition shape);
 *     the run then goes through createSf3dProducer().run and the report
 *     carries the producer's foreground-opportunity report;
 *   - effective-route receipts: which CPU phases ran on workers, which
 *     cooperative mechanisms settled (kit reports), bounded-prefix validation
 *     for the post-processor, and the canonical GLB hash.
 * The report is assembled and judged by the pure module
 * tools/product_route_witness_report.mjs (deterministically falsifiable).
 * A failure at any phase still writes a report naming the phase.
 *
 * Usage:
 *   node tools/smoke_product_route.mjs [--arm product-default|no-workers|workers-only|monolithic|'{json overrides}']
 *     [--contend | --contend-same-device] [--image P] [--report P] [--expected-glb-sha SHA|none]
 *     [--expected-commit SHA] [--expected-image-sha SHA] [--expected-weights-sha SHA]
 *     [--expected-kit-tree-sha256 SHA] [--protocol-timeout-ms N] [--max-gap-budget-ms N] [--allow-dirty] [--label TEXT]
 */
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import net from 'node:net';
import crypto from 'node:crypto';
import { spawn, execSync } from 'node:child_process';
import { createParentPhaseJournal, replayParentPhaseJournal } from './parent_phase_journal.mjs';
import { readImageInput, sha256Tree } from './witness_source_identity.mjs';
import {
  CANONICAL_DEMO_CHAIR_DUTY_COUNTS,
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
const CONTEND_SAME = hasFlag('--contend-same-device');
const IMAGE = path.resolve(argVal('--image', path.join(REPO, 'public/demo_chair.png')));
const LABEL = argVal('--label', '');
const ARM_NAME = `${ARM.startsWith('{') ? 'custom' : ARM}${CONTEND ? '+contend' : ''}${CONTEND_SAME ? '+contend-same-device' : ''}`;
const REPORT_PATH = path.resolve(argVal('--report', `/tmp/sf3d-product-route-witness-${ARM_NAME}.json`));
const INVOCATION_ID = `sf3d-witness-${Date.now()}-${process.pid}`;
const JOURNAL_PATH = path.resolve(argVal('--journal', path.join(
  os.homedir(), '.local/state/sf3d/parent-phase-journals', `${INVOCATION_ID}.jsonl`,
)));
const expectedShaArg = argVal('--expected-glb-sha', CANONICAL_DEMO_CHAIR_GLB_SHA256);
const EXPECTED_GLB_SHA = expectedShaArg === 'none' ? null : expectedShaArg;
const EXPECTED_WEIGHTS_SHA256 = argVal('--expected-weights-sha', null);
const EXPECTED_COMMIT = argVal('--expected-commit', null);
const EXPECTED_IMAGE_SHA256 = argVal('--expected-image-sha', null);
const EXPECTED_KIT_TREE_SHA256 = argVal('--expected-kit-tree-sha256', null);
// Puppeteer's 180 s protocol default is shorter than the authorized producer
// route. Zero explicitly disables that hidden deadline; callers may set a
// measured finite value when their execution contract has one.
const PROTOCOL_TIMEOUT_MS = Number(argVal('--protocol-timeout-ms', '0'));
// The product-default arm on the canonical image must also reproduce the
// measured cooperative duty counts (kit submittedGpuDutyCount), not just the GLB.
const EXPECTED_DUTY_COUNTS = (ARM === 'product-default' && EXPECTED_GLB_SHA === CANONICAL_DEMO_CHAIR_GLB_SHA256)
  ? CANONICAL_DEMO_CHAIR_DUTY_COUNTS : null;
const MAX_GAP_BUDGET_MS = argVal('--max-gap-budget-ms', null) == null ? null : Number(argVal('--max-gap-budget-ms'));
const ALLOW_DIRTY = hasFlag('--allow-dirty');

const KNOWN_ARMS = ['product-default', 'no-workers', 'workers-only', 'monolithic'];
// Deterministic failure injection for the failure-report contract
// (tools/test_witness_failure_report_contract.mjs): ordinary phases throw at
// entry; browser-evaluation injects a real page.evaluate() throw before app
// navigation, model loading, or GPU work.
const INJECT_FAILURE = process.env.SF3D_WITNESS_INJECT_FAILURE || null;
let phase = 'arguments';
let journal = null;
const memoryObservation = () => ({
  scope: 'witness-node-process-and-host',
  method: 'process.memoryUsage().rss and os.freemem/os.totalmem',
  units: 'bytes',
  processRssBytes: process.memoryUsage().rss,
  hostFreeBytes: os.freemem(),
  hostTotalBytes: os.totalmem(),
});
const enterPhase = (name, details = {}, { injectFailure = true } = {}) => {
  phase = name;
  journal?.append('phase-entered', { phase: name, ...details, memoryObservation: memoryObservation() });
  if (injectFailure && INJECT_FAILURE === name) throw new Error(`injected failure at ${name}`);
};
const completePhase = (name, details = {}) => journal?.append('phase-completed', { phase: name, ...details, memoryObservation: memoryObservation() });
function validateInvocation() {
  enterPhase('arguments');
  if (!KNOWN_ARMS.includes(ARM) && !ARM.startsWith('{')) {
    throw new Error(`unknown --arm ${ARM}; expected one of ${KNOWN_ARMS.join(', ')} or a JSON overrides object`);
  }
  if (CONTEND && CONTEND_SAME) throw new Error('--contend and --contend-same-device are separate arms; pick one');
  if (CONTEND_SAME && !(ARM === 'product-default' || ARM.startsWith('{'))) {
    throw new Error('--contend-same-device runs through the producer (product route); use --arm product-default or a JSON overrides object');
  }
  if (!Number.isSafeInteger(PROTOCOL_TIMEOUT_MS) || PROTOCOL_TIMEOUT_MS < 0) {
    throw new Error('--protocol-timeout-ms must be a non-negative integer (0 disables the protocol deadline)');
  }
  if (EXPECTED_WEIGHTS_SHA256 != null && !/^[0-9a-f]{64}$/i.test(EXPECTED_WEIGHTS_SHA256)) {
    throw new Error('--expected-weights-sha must be a 64-character SHA-256 hex digest');
  }
  if (EXPECTED_COMMIT != null && !/^[0-9a-f]{40}$/i.test(EXPECTED_COMMIT)) {
    throw new Error('--expected-commit must be a 40-character git commit SHA');
  }
  if (EXPECTED_IMAGE_SHA256 != null && !/^[0-9a-f]{64}$/i.test(EXPECTED_IMAGE_SHA256)) {
    throw new Error('--expected-image-sha must be a 64-character SHA-256 hex digest');
  }
  if (EXPECTED_KIT_TREE_SHA256 != null && !/^[0-9a-f]{64}$/i.test(EXPECTED_KIT_TREE_SHA256)) {
    throw new Error('--expected-kit-tree-sha256 must be a 64-character SHA-256 hex digest');
  }
  enterPhase('input');
  if (!fs.existsSync(IMAGE)) throw new Error(`image not found: ${IMAGE}`);
}

function writeFailure(failurePhase, error, partial = {}) {
  const failure = {
    schema: 'sf3d.product-route-witness-failure.v0',
    ok: false,
    arm: ARM_NAME,
    failurePhase,
    requested: { arm: ARM, contend: CONTEND, contendSameDevice: CONTEND_SAME, image: IMAGE, expectedCommit: EXPECTED_COMMIT, expectedImageSha256: EXPECTED_IMAGE_SHA256, expectedKitTreeSha256: EXPECTED_KIT_TREE_SHA256, expectedGlbSha: EXPECTED_GLB_SHA, expectedWeightsSha256: EXPECTED_WEIGHTS_SHA256, protocolTimeoutMs: PROTOCOL_TIMEOUT_MS, report: REPORT_PATH, journal: JOURNAL_PATH },
    // Effective identity as far as it was established when the run died.
    source: source ?? null,
    error: { message: error?.message || String(error), stack: error?.stack || null },
    partial,
    generatedAt: new Date().toISOString(),
  };
  failureDocument = failure;
  fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  fs.writeFileSync(REPORT_PATH, JSON.stringify(failure, null, 2));
  if (journal) {
    journal.append('failure-recorded', {
      failurePhase,
      errorName: error?.name || null,
      message: failure.error.message,
      stack: error?.stack || null,
      source: source ?? null,
      partial,
    });
  }
  console.error(`\nWITNESS FAILED at ${phase}: ${failure.error.message}\nreport: ${REPORT_PATH}`);
}

function allocatePort() {
  return new Promise((res, rej) => {
    const s = net.createServer(); s.unref(); s.on('error', rej);
    s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); });
  });
}

function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(filePath);
    stream.on('data', chunk => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

let source = null;
let imageInput = null;
let commit = null, dirty = null, kitVersion = null;
const procs = [];
const cleanup = () => { for (const p of procs) { try { p.kill(); } catch { /* gone */ } } };
let browser = null;
let terminalStatus = 'failed';
let terminalError = null;
let failureDocument = null;
let memoryHeartbeat = null;

try {
  journal = createParentPhaseJournal({
    journalPath: JOURNAL_PATH,
    invocationId: INVOCATION_ID,
    requested: {
      harnessRouteClass: 'sf3d.image-to-mesh.webgpu-local.v0',
      arm: ARM,
      contend: CONTEND,
      contendSameDevice: CONTEND_SAME,
      image: IMAGE,
      reportPath: REPORT_PATH,
      journalPath: JOURNAL_PATH,
      expectedCommit: EXPECTED_COMMIT,
      expectedImageSha256: EXPECTED_IMAGE_SHA256,
      expectedKitTreeSha256: EXPECTED_KIT_TREE_SHA256,
      expectedGlbSha256: EXPECTED_GLB_SHA,
      expectedWeightsSha256: EXPECTED_WEIGHTS_SHA256,
      protocolTimeoutMs: PROTOCOL_TIMEOUT_MS,
    },
  });
  validateInvocation();
  completePhase('arguments');
  completePhase('input', { imagePath: IMAGE, bytes: fs.statSync(IMAGE).size });

  // --- Source identity (effective, not requested) ---
  enterPhase('source-identity');
  commit = execSync('git rev-parse HEAD', { cwd: REPO }).toString().trim();
  dirty = execSync('git status --porcelain', { cwd: REPO }).toString().trim().length > 0;
  imageInput = readImageInput(IMAGE);
  const weightsPath = path.join(REPO, 'public/weights.bin');
  const weightsStat = fs.existsSync(weightsPath) ? fs.statSync(weightsPath) : null;
  source = {
    commit,
    dirty,
    kitVersion: null,
    hostname: os.hostname(),
    node: process.version,
    label: LABEL || null,
    input: { path: imageInput.path, bytes: imageInput.bytes.length, sha256: imageInput.sha256 },
    weightArtifact: weightsStat ? {
    path: fs.realpathSync(weightsPath),
    bytes: weightsStat.size,
    modifiedAtMs: weightsStat.mtimeMs,
    sha256: null,
    sha256Status: 'hashing',
    } : { path: weightsPath, status: 'missing-before-browser-load' },
    weightRepresentation: { status: 'not-exposed-by-current-loader', format: null },
  };
  if (weightsStat) {
    const sha256 = await sha256File(weightsPath);
    const afterHash = fs.statSync(weightsPath);
    if (afterHash.size !== weightsStat.size || afterHash.mtimeMs !== weightsStat.mtimeMs || afterHash.ino !== weightsStat.ino) {
      throw new Error('weights.bin changed while its source identity was being hashed');
    }
    source = {
      ...source,
      weightArtifact: { ...source.weightArtifact, sha256, sha256Status: 'computed' },
    };
  }
  journal.append('effective-identity', {
    ...source,
    harnessRouteClass: 'sf3d.image-to-mesh.webgpu-local.v0',
    effectiveProducerDeviceRoute: { status: 'unobserved', reason: 'pre-navigation identity checkpoint' },
  });
  if (EXPECTED_COMMIT && commit.toLowerCase() !== EXPECTED_COMMIT.toLowerCase()) {
    throw new Error(`source commit ${commit} does not match requested ${EXPECTED_COMMIT}`);
  }
  if (EXPECTED_IMAGE_SHA256 && source.input.sha256.toLowerCase() !== EXPECTED_IMAGE_SHA256.toLowerCase()) {
    throw new Error(`input image SHA-256 ${source.input.sha256} does not match requested ${EXPECTED_IMAGE_SHA256}`);
  }
  if (EXPECTED_WEIGHTS_SHA256 && !weightsStat) throw new Error('requested --expected-weights-sha but public/weights.bin is missing');
  if (EXPECTED_WEIGHTS_SHA256 && source.weightArtifact.sha256.toLowerCase() !== EXPECTED_WEIGHTS_SHA256.toLowerCase()) {
    throw new Error(`weights.bin SHA-256 ${source.weightArtifact.sha256} does not match requested ${EXPECTED_WEIGHTS_SHA256}`);
  }
  completePhase('source-identity', { commit, dirty });
  if (dirty && !ALLOW_DIRTY) {
    throw new Error('worktree is dirty; commit first or pass --allow-dirty (the report will still record dirty=true)');
  }
  enterPhase('kit-identity');
  const kitPath = path.join(REPO, 'node_modules/@kaminos/webgpu-inference-kit');
  const installedKit = JSON.parse(fs.readFileSync(path.join(kitPath, 'package.json'), 'utf8'));
  const lock = JSON.parse(fs.readFileSync(path.join(REPO, 'package-lock.json'), 'utf8'));
  const lockKit = lock.packages?.['node_modules/@kaminos/webgpu-inference-kit'] ?? null;
  const installedKitTree = await sha256Tree(kitPath);
  kitVersion = installedKit.version;
  source = {
    ...source,
    kitVersion,
    kitIdentity: {
      version: installedKit.version,
      resolved: lockKit?.resolved ?? null,
      integrity: lockKit?.integrity ?? null,
      installedTreeSha256: installedKitTree.sha256,
      installedFiles: installedKitTree.files,
    },
  };
  journal.append('effective-package-identity', source.kitIdentity);
  if (EXPECTED_KIT_TREE_SHA256 && installedKitTree.sha256.toLowerCase() !== EXPECTED_KIT_TREE_SHA256.toLowerCase()) {
    throw new Error(`installed kit tree SHA-256 ${installedKitTree.sha256} does not match requested ${EXPECTED_KIT_TREE_SHA256}`);
  }
  completePhase('kit-identity', source.kitIdentity);

  // --- Serve the checkout ---
  enterPhase('vite-start');
  const port = await allocatePort();
  const vite = spawn('npx', ['vite', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(vite);
  await new Promise((res, rej) => {
    const to = setTimeout(() => rej(new Error('vite startup timeout')), 40000);
    vite.stdout.on('data', d => { if (/Local:|ready/.test(d.toString())) { clearTimeout(to); res(); } });
    vite.on('error', e => { clearTimeout(to); rej(e); });
  });
  completePhase('vite-start', { port });

  enterPhase('browser-launch');
  browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: false,
    protocolTimeout: PROTOCOL_TIMEOUT_MS,
    args: [
      '--enable-unsafe-webgpu', '--use-angle=metal',
      '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding',
      '--no-first-run', '--no-default-browser-check',
    ],
  });
  completePhase('browser-launch', { browserPid: browser.process()?.pid ?? null });
  memoryHeartbeat = setInterval(() => {
    try {
      journal.append('resource-observation', {
        phase,
        observedAt: new Date().toISOString(),
        memoryObservation: memoryObservation(),
      });
    } catch (error) {
      console.error(`Parent memory observation failed: ${error.message}`);
    }
  }, 5000);
  memoryHeartbeat.unref();
  const page = await browser.newPage();
  await page.exposeFunction('__sf3dParentPhase', (event) => {
    if (!event || typeof event !== 'object' || Array.isArray(event)) {
      throw new TypeError('browser phase event must be an object');
    }
    const { type, ...payload } = event;
    const eventType = type
      || (event.state === 'entered' ? 'phase-entered' : null)
      || (event.state === 'completed' ? 'phase-completed' : null)
      || 'browser-phase-event';
    if (eventType === 'phase-entered' && typeof event.phase === 'string') phase = event.phase;
    if (event.phase === 'weight-representation' && event.state === 'observed') {
      source = { ...source, weightRepresentation: event };
    }
    journal.append(eventType, {
      ...payload,
      memoryObservation: memoryObservation(),
    });
  });
  // Exercise a real Puppeteer/CDP page evaluation before navigation can load
  // the model or submit any GPU work. This is a bounded harness-contract
  // failure point, not an inference run.
  enterPhase('browser-evaluation', {}, { injectFailure: false });
  await page.evaluate((inject) => {
    if (inject) throw new Error('injected failure at browser-evaluation');
    return true;
  }, INJECT_FAILURE === 'browser-evaluation');
  completePhase('browser-evaluation');
  const pageErrors = [];
  page.on('pageerror', e => { pageErrors.push(e.message); console.error('[pageerror]', e.message); });
  page.on('console', m => { if (m.type() === 'error') console.error('[console.error]', m.text().slice(0, 300)); });

  enterPhase('page-load');
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
  completePhase('page-load', { status: 'Ready' });
  await page.evaluate(() => {
    const device = window._sf3d_device;
    if (device?.lost && window.__sf3dParentPhase) {
      device.lost.then(info => window.__sf3dParentPhase({
        type: 'device-lost', reason: info.reason ?? null, message: info.message ?? null,
      }));
    }
  });

  const imageB64 = imageInput.bytes.toString('base64');
  enterPhase('image-submission');
  await page.evaluate(async (b64) => {
    await new Promise((res, rej) => { const img = new Image(); img.onload = () => { window._witnessImage = img; res(); }; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; });
  }, imageB64);
  completePhase('image-submission', { bytes: imageInput.bytes.length, sha256: imageInput.sha256, encoding: 'base64-data-url' });

  // --- The witnessed run ---
  enterPhase('witness');
  const raw = await page.evaluate(async ({ armSpec, contend, contendSame, expectedDutyCounts }) => {
    await window.__sf3dParentPhase({ type: 'phase-entered', phase: 'product-route' });
    const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
    const {
      createProductRouteOptions, createProductRouteWorkers, describeProductRouteOptions, terminateProductRouteWorkers,
    } = await import('/src/lib/product_route.js');
    const { projectCooperativeReport } = await import('/tools/product_route_witness_report.mjs');
    const { readBrowserKitIdentity } = await import('/tools/browser_kit_identity.js');
    const { acceptBoundedPrefixArm, expectedChannelDutyCount } = await import('/tools/bounded_prefix_acceptance.mjs');
    const { acceptCooperativeMechanismReport } = await import('/tools/cooperative_identity_acceptance.mjs');

    const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines;
    const producer = window._sf3d_producer;
    const img = window._witnessImage;
    if (!device || !weights || !pipelines || !img) throw new Error('page state missing (device/weights/pipelines/image)');
    if (!producer) throw new Error('page state missing (_sf3d_producer) for producer route identity');
    if (producer.device !== device) throw new Error('producer device is not window._sf3d_device; refusing route identity substitution');
    const browserKit = await readBrowserKitIdentity();
    const producerRoute = {
      invocation: contendSame ? 'producer.run' : 'direct-full-pipeline',
      deviceRelation: producer.device === device ? 'producer-device===window._sf3d_device' : 'mismatch',
      producerResourcesMatchApp: producer.weights === weights && producer.pipelines === pipelines,
      rendererDeviceRelationship: 'not-observed-by-this-witness',
      producer: {
        routeId: producer.routeId,
        commit: producer.commit,
        kitVersion: producer.kitVersion,
        deviceInjected: producer.deviceInjected,
        deviceTopology: producer.deviceInjected ? 'host-injected-device' : 'producer-owned-device',
        backend: producer.backend,
      },
      runIdentity: null,
      browserKit,
    };
    if (document.visibilityState !== 'visible') throw new Error(`page visibility ${document.visibilityState} at start`);

    // Arm → options.
    let workers = null;
    let options;
    let overrides = {};
    if (armSpec === 'monolithic') {
      options = Object.freeze({ cooperativeDino: false });
    } else if (armSpec === 'no-workers') {
      options = createProductRouteOptions({});
    } else if (armSpec === 'workers-only') {
      workers = createProductRouteWorkers();
      options = createProductRouteOptions({ workers, overrides: {
        cooperativeDino: false, cooperativeTwoStream: false, cooperativePostProcessor: false, cooperativeBake: false, decoderArena: false,
      } });
    } else if (contendSame) {
      // The producer builds the same options for its run; this copy is for
      // validation/description only (workers are the producer's long-lived ones).
      overrides = armSpec.startsWith('{') ? JSON.parse(armSpec) : {};
      options = createProductRouteOptions({ workers: producer.workers, overrides });
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

    const CONTENDER_WGSL = `
      @group(0) @binding(0) var<storage, read_write> data: array<f32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        var x = data[id.x];
        for (var i: u32 = 0u; i < 64u; i = i + 1u) { x = (x * 1.0001221) + f32((i & 7u) + 1u) * 0.00003125; }
        data[id.x] = x;
      }`;
    const contender = {
      enabled: contend || contendSame,
      mode: contendSame ? 'same-device-foreground-opportunity' : (contend ? 'second-device' : null),
      submitted: 0, completed: 0, errors: [],
      receipts: contendSame ? { completed: 0, failed: 0, canceled: 0, outsideRun: 0, schedulerBoundary: 0, foregroundWindow: 0, runFinish: 0 } : null,
    };
    let contenderDone = Promise.resolve();
    if (contendSame) {
      // Host-frame contender on SF3D's own device through the producer's
      // foreground-opportunity bridge: one frame per requestAnimationFrame, the
      // way a kiln frame loop would submit. Each receipt says where it was
      // serviced (scheduler duty boundary / idle drain / run finish / outside run).
      contenderDone = (async () => {
        try {
          const cdev = producer.device;
          const module = cdev.createShaderModule({ code: CONTENDER_WGSL });
          const pipeline = cdev.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
          const buffer = cdev.createBuffer({ size: 65536 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
          const bindGroup = cdev.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] });
          let n = 0;
          while (on) {
            n += 1;
            const requestId = `witness-host-frame:${n}`;
            const handle = producer.requestForegroundOpportunity({
              requestId,
              metadata: { kind: 'witness-host-frame', frame: n },
              run(ctx) {
                const enc = ctx.device.createCommandEncoder();
                const pass = enc.beginComputePass();
                pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(1024); pass.end();
                ctx.submit([enc.finish()], { submissionId: `${requestId}:1` });
                return { frame: n };
              },
            });
            contender.submitted += 1;
            const r = await handle.completion;
            if (r.status === 'completed') { contender.completed += 1; contender.receipts.completed += 1; }
            else if (String(r.status).startsWith('canceled')) contender.receipts.canceled += 1;
            else { contender.receipts.failed += 1; contender.errors.push(`${requestId}: ${r.failure?.error?.message || r.status}`); }
            if (r.runId == null || r.boundary?.phase === 'foreground-idle') contender.receipts.outsideRun += 1;
            else if (r.boundary?.phase === 'foreground-run-finish') contender.receipts.runFinish += 1;
            else if (['image-preprocess-worker', 'image-preprocess', 'marching-tet-worker', 'marching-tet', 'clip-prep-worker', 'uv-unwrap-worker', 'uv-unwrap', 'uv-rasterize', 'texture-materialize-worker', 'glb-export'].includes(r.boundary?.phase)) contender.receipts.foregroundWindow += 1;
            else contender.receipts.schedulerBoundary += 1;
            await new Promise(r => requestAnimationFrame(r));
          }
          buffer.destroy();
        } catch (e) { contender.errors.push(e?.message || String(e)); }
      })();
      await new Promise(r => setTimeout(r, 200));
    } else if (contend) {
      // Same-page WebGPU contender on a second device (SHARP contention-witness shape).
      contenderDone = (async () => {
        try {
          const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
          if (!adapter) throw new Error('no contender adapter');
          const cdev = await adapter.requestDevice();
          const module = cdev.createShaderModule({ code: CONTENDER_WGSL });
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
      await window.__sf3dParentPhase({
        type: 'phase-entered', phase: 'full-pipeline',
        requested: { arm: armSpec, contend, contendSameDevice: contendSame },
        resolvedRouteOptions: describeProductRouteOptions(options),
        effectiveProducerDeviceRoute: producerRoute,
      });
      out = contendSame
        ? await producer.run(img, { runId: `witness:${armSpec.startsWith('{') ? 'custom' : armSpec}`, onProgress: (m) => progress.push(String(m)), routeOverrides: overrides })
        : await runFullPipelineToGlb(device, pipelines, weights, img, {
          ...options,
          onPhase: (event) => window.__sf3dParentPhase(event),
        }, (m) => progress.push(String(m)));
      await window.__sf3dParentPhase({
        type: 'phase-completed', phase: 'full-pipeline',
        effectiveRouteOptions: describeProductRouteOptions(options),
      });
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
    // Kit-backed validation of EVERY requested cooperative mechanism's complete
    // report (the acceptor treats a missing record as a rejection). The
    // bounded-prefix post-processor keeps its validator-backed arm acceptance
    // (plus the progress-string check); the strict-prefix mechanisms go through
    // the same kit validator with their exact identity and, when pinned, the
    // measured duty count.
    const cooperativeValidations = {};
    const requestedDescription = describeProductRouteOptions(options);
    const ppReport = out.cooperativeReports?.['post-processor'];
    if (ppReport && options.cooperativePostProcessor && options.postProcessorDutyGranularity === 'channel-range'
        && options.postProcessorCompletionPolicy === 'bounded-prefix') {
      const v = acceptBoundedPrefixArm({
        report: ppReport, progressMessages: progress,
        expectedGpuDutyCount: expectedChannelDutyCount(options.postProcessorChannelsPerDuty),
        maxInFlightGpuDuties: options.postProcessorMaxInFlightGpuDuties,
      });
      cooperativeValidations['post-processor'] = { ok: v.ok, errors: [...v.errors], progressHonest: v.progressHonest };
    } else if (ppReport && options.cooperativePostProcessor) {
      const v = acceptCooperativeMechanismReport('post-processor', ppReport, requestedDescription, { expectedGpuDutyCount: expectedDutyCounts?.['post-processor'] ?? null });
      cooperativeValidations['post-processor'] = { ok: v.ok, errors: [...v.errors] };
    }
    for (const key of ['dinov2-tokenizer', 'two-stream-backbone', 'texture-bake']) {
      const r = out.cooperativeReports?.[key];
      if (!r) continue;
      const v = acceptCooperativeMechanismReport(key, r, requestedDescription, { expectedGpuDutyCount: expectedDutyCounts?.[key] ?? null });
      cooperativeValidations[key] = { ok: v.ok, errors: [...v.errors], expectations: v.expectations };
    }
    const bakeTel = out.cooperativeReports?.['texture-bake']?.textureBakeTelemetry;
    const materializationOffloaded = bakeTel?.materializationOffloaded ?? bakeTel?.phases?.materializationOffloaded ?? null;

    producerRoute.runIdentity = out.identity ?? null;
    return {
      frames, stageSpans: out.stageSpans, stageTimings: out.stageTimings, totalMs: out.totalMs,
      inferenceWindow: { startMs, endMs }, visibility,
      offloads: out.offloads, requested: describeProductRouteOptions(options),
      cooperative, cooperativeValidations, materializationOffloaded,
      output: { glbSha256, glbBytes: out.glb.byteLength, numVertices: out.numVertices, numFaces: out.numFaces, roughness: out.roughness, metallic: out.metallic },
      contender, progressCount: progress.length,
      foregroundOpportunities: out.foregroundOpportunityReport ?? null,
      routeReceiptValidation: out.receiptValidation ?? null,
      producerRoute,
      backend: {
        vendor: producer.adapterInfo?.vendor ?? null,
        architecture: producer.adapterInfo?.architecture ?? null,
        device: producer.adapterInfo?.device ?? null,
        description: producer.adapterInfo?.description ?? null,
        userAgent: navigator.userAgent,
      },
    };
  }, { armSpec: ARM, contend: CONTEND, contendSame: CONTEND_SAME, expectedDutyCounts: EXPECTED_DUTY_COUNTS });

  if (pageErrors.length) throw new Error(`page errors during run: ${pageErrors.join(' | ')}`);
  completePhase('witness', { outputSha256: raw.output.glbSha256, outputBytes: raw.output.glbBytes });

  // --- Assemble + judge ---
  enterPhase('assemble');
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
    foregroundOpportunities: raw.foregroundOpportunities,
    producerRoute: raw.producerRoute,
  });
  const verdict = acceptProductRouteWitness(report, {
    expectedGlbSha: EXPECTED_GLB_SHA, requireContender: CONTEND || CONTEND_SAME, maxGapBudgetMs: MAX_GAP_BUDGET_MS,
    expectedDutyCounts: EXPECTED_DUTY_COUNTS,
  });
  if (raw.routeReceiptValidation && raw.routeReceiptValidation.ok !== true) {
    throw new Error(`producer route receipt failed validation: ${(raw.routeReceiptValidation.errors || []).join('; ')}`);
  }
  const durable = { ...report, verdict: { ok: verdict.ok, errors: [...verdict.errors] } };
  durable.execution = {
    requestedProtocolTimeoutMs: PROTOCOL_TIMEOUT_MS,
    effectiveProtocolTimeout: PROTOCOL_TIMEOUT_MS === 0 ? 'disabled' : 'finite',
    expectedCommit: EXPECTED_COMMIT,
    expectedImageSha256: EXPECTED_IMAGE_SHA256,
    expectedKitTreeSha256: EXPECTED_KIT_TREE_SHA256,
    expectedWeightsSha256: EXPECTED_WEIGHTS_SHA256,
    effectiveWeightsSha256: source.weightArtifact?.sha256 ?? null,
  };
  fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  fs.writeFileSync(REPORT_PATH, JSON.stringify(durable, null, 2));
  completePhase('assemble', { verdict: verdict.ok ? 'accepted' : 'rejected' });
  terminalStatus = verdict.ok ? 'succeeded' : 'rejected';

  // --- Summary ---
  const wr = report.cadence.wholeRoute;
  console.log(`\n=== SF3D product-route witness: ${ARM_NAME} ===`);
  console.log(`source ${commit.slice(0, 10)} dirty=${dirty} kit=${kitVersion} backend=${raw.backend.vendor}/${raw.backend.architecture}`);
  console.log(`route wall ${report.totalMs}ms  frames=${wr.frameIntervalCount}  p50/p95/p99/max = ${wr.p50Ms}/${wr.p95Ms}/${wr.p99Ms}/${wr.maxMs}ms  >16.7:${wr.over16_7} >33.3:${wr.over33_3} >100:${wr.over100}`);
  if (raw.contender.enabled) console.log(`contender (${raw.contender.mode}): submitted=${raw.contender.submitted} completed=${raw.contender.completed} errors=${raw.contender.errors.length}`);
  if (raw.contender.receipts) {
    const r = raw.contender.receipts;
    console.log(`host frames serviced at: scheduler duty boundary=${r.schedulerBoundary} foreground window=${r.foregroundWindow} run finish=${r.runFinish} outside run=${r.outsideRun}  (failed=${r.failed} canceled=${r.canceled})`);
  }
  if (report.foregroundOpportunities) {
    const f = report.foregroundOpportunities;
    console.log(`foreground service: status=${f.status} requests=${f.requestCount} receipts=${f.receiptCount} pending=${f.pendingRequestCount} active=${f.activeRequestCount}`);
  }
  console.log(`GLB sha ${report.output.glbSha256.slice(0, 12)}… (${report.output.glbBytes} B, ${report.output.numVertices}v/${report.output.numFaces}f) expected ${EXPECTED_GLB_SHA ? EXPECTED_GLB_SHA.slice(0, 12) + '…' : 'none'}`);
  console.log(`offloads: ${Object.entries(report.effective.offloads).map(([k, v]) => `${k}=${v}`).join(' ')}`);
  console.log(`cooperative: ${Object.entries(report.effective.cooperative).map(([k, c]) => `${k}=${c.status}(${c.submittedGpuDutyCount ?? '?'} duties submitted, ${c.progress?.completedItems ?? '?'}/${c.progress?.totalItems ?? '?'} items${c.completionPolicy === 'bounded-prefix' ? `, bounded depth ${c.maxObservedInFlightGpuDuties}` : ''})`).join(' ') || 'none'}`);
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
  terminalError = err;
  writeFailure(phase, err);
  process.exitCode = 1;
} finally {
  if (memoryHeartbeat) clearInterval(memoryHeartbeat);
  if (browser) {
    try {
      await browser.close();
      journal?.append('browser-teardown', { status: 'closed' });
    } catch (error) {
      terminalStatus = 'failed';
      terminalError ??= error;
      journal?.append('browser-teardown', { status: 'failed', error: error?.message || String(error) });
    }
  }
  cleanup();
  if (journal) {
    journal.append('terminal', {
      status: terminalStatus,
      failurePhase: terminalError ? phase : null,
      errorName: terminalError?.name || null,
      message: terminalError?.message || null,
      reportPath: REPORT_PATH,
    });
    journal.close();
    try {
      const replay = replayParentPhaseJournal(JOURNAL_PATH);
      if (fs.existsSync(REPORT_PATH)) {
        const report = failureDocument ?? JSON.parse(fs.readFileSync(REPORT_PATH, 'utf8'));
        report.parentPhaseJournal = replay;
        fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));
      }
    } catch (error) {
      console.error(`Could not attach journal replay to final report: ${error.message}`);
    }
  }
}
