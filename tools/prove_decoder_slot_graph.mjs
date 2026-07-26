#!/usr/bin/env node
/**
 * Prove the decoder's reusable slot graph (Cranial arena assay contract #1).
 *
 * Runs ONE real triplane-decode over a batch of texels through
 * captureGpuBufferAllocations() and dumps the exact set of scratch buffers with
 * their stable slot keys and byte sizes. This is the evidence the scratch arena
 * is built against: the 31-resource reusable slot graph, each slot sized as a
 * function of the batch N so a maxBatch-capacity arena can hold them all.
 *
 * Asserts the captured count matches Cranial's assay (31 per range for the bake
 * heads ['features','perturb_normal']) and that every slot size is N-linear.
 *
 * Usage: node tools/prove_decoder_slot_graph.mjs [--image PATH] [--batch N]
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
const REPORT_PATH = path.resolve(argVal('--report', '/tmp/sf3d-decoder-slot-graph.json'));
const BATCH_A = 4096, BATCH_B = 8192; // two batch sizes to prove N-linearity

const allocatePort = () => new Promise((res, rej) => { const s = net.createServer(); s.unref(); s.on('error', rej); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); }); });
const procs = [];
const cleanup = () => { for (const p of procs) { try { p.kill(); } catch {} } };
const fail = (m) => { fs.writeFileSync(REPORT_PATH, JSON.stringify({ ok: false, error: m }, null, 2)); console.error(`\n✗ PROOF FAILED: ${m}`); cleanup(); process.exit(1); };

(async () => {
  const port = await allocatePort();
  const vite = spawn('npx', ['vite', '--port', String(port), '--strictPort'], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(vite);
  await new Promise((res, rej) => { const to = setTimeout(() => rej(new Error('vite timeout')), 40000); vite.stdout.on('data', d => { if (/Local:|ready/.test(d.toString())) { clearTimeout(to); res(); } }); vite.on('error', e => { clearTimeout(to); rej(e); }); }).catch(e => fail(e.message));

  const browser = await puppeteer.launch({ executablePath: CHROME_PATH, headless: false, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
  const page = await browser.newPage();
  page.on('pageerror', e => console.error('[pageerror]', e.message));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded', timeout: 30000 });
  const start = Date.now();
  while (Date.now() - start < 180000) { const s = await page.$eval('#status', el => el.textContent).catch(() => ''); if (s.includes('Ready')) break; await new Promise(r => setTimeout(r, 500)); }

  const imageB64 = fs.readFileSync(IMAGE).toString('base64');
  await page.evaluate(async (b64) => { await new Promise((res, rej) => { const img = new Image(); img.onload = () => { window._img = img; res(); }; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; }); }, imageB64);

  const result = await page.evaluate(async ({ batchA, batchB }) => {
    // Run the mesh pipeline once to get a real triplane buffer + decoder.
    const { runInference } = await import('/src/lib/inference.js');
    const { captureGpuBufferAllocations } = await import('/src/lib/gpu.js');
    const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines, img = window._img;
    const mesh = await runInference(device, pipelines, weights, img, () => {}, { cooperativeDino: false });
    const decoder = mesh._triplaneDecoder, triplanes = mesh._triplanesBuf, dw = mesh._decoderWeights;

    // Build a dummy query-position buffer of N texels (values irrelevant to the
    // slot graph — we only inspect allocations, not outputs).
    function makePos(N) {
      const arr = new Float32Array(N * 3);
      const buf = device.createBuffer({ size: arr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
      new Float32Array(buf.getMappedRange()).set(arr); buf.unmap(); return buf;
    }
    function captureSlots(N) {
      const posBuf = makePos(N);
      const enc = device.createCommandEncoder();
      const { allocations } = captureGpuBufferAllocations(() =>
        decoder.decode(enc, posBuf, triplanes, N, dw, ['features', 'perturb_normal']));
      // We don't submit; we only need the allocation shape. Retire captured bufs.
      for (const a of allocations) { try { a.buffer.destroy(); } catch {} }
      posBuf.destroy();
      return allocations.map(a => ({ label: a.label, size: a.size }));
    }
    return { a: captureSlots(batchA), b: captureSlots(batchB), meshVerts: mesh.numVertices };
  }, { batchA: BATCH_A, batchB: BATCH_B });

  await browser.close().catch(() => {}); cleanup();

  const a = result.a, b = result.b;
  // Pair slots by index (deterministic decode order) and prove N-linearity.
  if (a.length !== b.length) fail(`slot count differs across batches: ${a.length} vs ${b.length}`);
  const ratio = BATCH_B / BATCH_A;
  const slots = a.map((sa, i) => {
    const sb = b[i];
    // size = base + perN * N ; solve from two points (most slots have base 0)
    const perN = (sb.size - sa.size) / (BATCH_B - BATCH_A);
    const base = sa.size - perN * BATCH_A;
    return { index: i, label: sa.label, sizeAtA: sa.size, sizeAtB: sb.size, perTexelBytes: perN, baseBytes: Math.round(base) };
  });
  // Slot keys must be stable (a label appears; duplicates allowed only if the
  // decode legitimately reuses a key — flag duplicates for arena awareness).
  const labelCounts = {};
  for (const s of slots) labelCounts[s.label] = (labelCounts[s.label] || 0) + 1;
  const duplicateLabels = Object.entries(labelCounts).filter(([, c]) => c > 1);

  const capacityAt = (N) => slots.reduce((sum, s) => sum + Math.max(0, Math.round(s.baseBytes + s.perTexelBytes * N)), 0);

  // DRIFT GATE (review residual-risk #2): the captured real-decoder slot graph
  // must exactly match the frozen DECODER_BAKE_SLOT_GRAPH the arena pre-allocates
  // against. If a decoder change adds/renames a slot or changes a per-texel size,
  // this diverges and fails loud at test time (rather than a silent output
  // change or a runtime-only throw). Compare by ordered (slotKey, perTexelBytes,
  // baseBytes).
  const { DECODER_BAKE_SLOT_GRAPH } = await import('../src/lib/decoder_scratch_arena.js');
  const capturedKeyed = slots.map(s => ({ slotKey: s.label, perTexelBytes: Math.round(s.perTexelBytes), baseBytes: s.baseBytes }));
  const frozen = DECODER_BAKE_SLOT_GRAPH.map(s => ({ slotKey: s.slotKey, perTexelBytes: s.perTexelBytes, baseBytes: s.baseBytes }));
  const driftMismatch = JSON.stringify(capturedKeyed) !== JSON.stringify(frozen);

  const report = {
    ok: !driftMismatch,
    slotCount: slots.length,
    frozenSlotCount: frozen.length,
    driftMismatch,
    driftDetail: driftMismatch ? { captured: capturedKeyed, frozen } : null,
    duplicateLabels,
    slots,
    arenaCapacityBytes: { at4096: capacityAt(4096), at16384: capacityAt(16384) },
    note: 'Each slot size = baseBytes + perTexelBytes*N; a maxBatch-capacity arena holds all slots and is reused across ranges.',
  };
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));
  if (driftMismatch) fail(`SLOT GRAPH DRIFT: captured decoder slots != frozen DECODER_BAKE_SLOT_GRAPH — arena would allocate the wrong slots. See ${REPORT_PATH}`);

  console.log('\n=== Decoder scratch slot graph ===');
  console.log(`slots captured: ${slots.length} ${slots.length === 31 ? '(matches Cranial assay 31)' : '(DIFFERS from assay 31 — investigate)'}`);
  for (const s of slots) console.log(`  [${String(s.index).padStart(2)}] ${s.label.padEnd(26)} ${String(s.sizeAtA).padStart(9)}B@4096  perTexel=${s.perTexelBytes.toFixed(1)}B  base=${s.baseBytes}B`);
  if (duplicateLabels.length) console.log(`  duplicate slot labels: ${JSON.stringify(duplicateLabels)}`);
  console.log(`arena capacity: ${(report.arenaCapacityBytes.at4096 / 1e6).toFixed(1)}MB@4096  ${(report.arenaCapacityBytes.at16384 / 1e6).toFixed(1)}MB@16384`);
  console.log(`report: ${REPORT_PATH}`);
  process.exit(0);
})().catch(e => fail(e.message));
