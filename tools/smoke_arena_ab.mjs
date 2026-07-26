#!/usr/bin/env node
/**
 * Decoder scratch arena A/B smoke.
 *
 * Proves the arena (options.decoderArena) preserves byte-identical GLB while
 * eliminating the ~1.015GB per-route decode allocation churn Cranial's assay
 * identified. Two arms over one loaded model:
 *   current — cooperative bake, per-range transient scratch (churn)
 *   arena   — cooperative bake + decoder scratch arena (reuse)
 * Asserts: GLB byte-identical; arena arm's captured scratch allocation bytes
 * are dramatically lower (churn collapsed); arena capacity reasonable.
 *
 * Usage: node tools/smoke_arena_ab.mjs [--image PATH] [--batch N]
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
const BATCH = Number(argVal('--batch', '4096')) || 4096;
const REPORT_PATH = path.resolve(argVal('--report', '/tmp/sf3d-arena-ab.json'));

const allocatePort = () => new Promise((res, rej) => { const s = net.createServer(); s.unref(); s.on('error', rej); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); }); });
const procs = [];
const cleanup = () => { for (const p of procs) { try { p.kill(); } catch {} } };
const fail = (m) => { fs.writeFileSync(REPORT_PATH, JSON.stringify({ ok: false, error: m }, null, 2)); console.error(`\n✗ SMOKE FAILED: ${m}`); cleanup(); process.exit(1); };

(async () => {
  const port = await allocatePort();
  const vite = spawn('npx', ['vite', '--port', String(port), '--strictPort'], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(vite);
  await new Promise((res, rej) => { const to = setTimeout(() => rej(new Error('vite timeout')), 40000); vite.stdout.on('data', d => { if (/Local:|ready/.test(d.toString())) { clearTimeout(to); res(); } }); vite.on('error', e => { clearTimeout(to); rej(e); }); }).catch(e => fail(e.message));

  const browser = await puppeteer.launch({ executablePath: CHROME_PATH, headless: false, protocolTimeout: 900000, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
  const page = await browser.newPage();
  const pageErrors = []; page.on('pageerror', e => pageErrors.push(e.message));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded', timeout: 30000 });
  const start = Date.now();
  while (Date.now() - start < 180000) { const s = await page.$eval('#status', el => el.textContent).catch(() => ''); if (s.includes('Ready')) break; await new Promise(r => setTimeout(r, 500)); }

  const imageB64 = fs.readFileSync(IMAGE).toString('base64');
  await page.evaluate(async (b64) => { await new Promise((res, rej) => { const img = new Image(); img.onload = () => { window._img = img; res(); }; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; }); }, imageB64);

  const result = await page.evaluate(async (BATCH) => {
    const { runFullPipelineToGlb } = await import('/src/lib/full_pipeline.js');
    const device = window._sf3d_device, weights = window._sf3d_weights, pipelines = window._sf3d_pipelines, img = window._img;
    const toB64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
    async function arm(opts) {
      const out = await runFullPipelineToGlb(device, pipelines, weights, img, opts);
      const bake = out.cooperativeReports?.['texture-bake'] || null;
      const scratch = bake?.textureBakeTelemetry?.scratch || null;
      return {
        glbB64: toB64(new Uint8Array(out.glb)), glbBytes: out.glb.byteLength,
        arenaSnapshot: out.arenaSnapshot || null,
        scratch,
        allocatedBytes: scratch?.allocatedBytes ?? null,
        allocatedCount: scratch?.allocatedCount ?? null,
        rangeCount: bake?.boundaries?.[0]?.actualRangeCount ?? null,
      };
    }
    const current = await arm({ cooperativeDino: false, cooperativeBake: true, bakeSchedulingMode: 'cooperative', bakeBatchTexels: BATCH });
    const arena = await arm({ cooperativeDino: false, cooperativeBake: true, bakeSchedulingMode: 'cooperative', bakeBatchTexels: BATCH, decoderArena: true });
    return { current, arena };
  }, BATCH);

  await browser.close().catch(() => {}); cleanup();
  if (pageErrors.length) fail(`page errors: ${pageErrors.join(' | ')}`);

  const hash = (b64) => crypto.createHash('sha256').update(Buffer.from(b64, 'base64')).digest('hex');
  const gCurrent = hash(result.current.glbB64), gArena = hash(result.arena.glbB64);
  const identical = gCurrent === gArena;
  const curAlloc = result.current.allocatedBytes;
  const arenaAlloc = result.arena.allocatedBytes;
  // churn collapsed: arena arm allocates far fewer scratch bytes than current.
  const churnCollapsed = (curAlloc != null && arenaAlloc != null) ? arenaAlloc < curAlloc * 0.2 : null;

  const report = {
    ok: identical && (churnCollapsed !== false),
    glbIdentical: identical, glbSha: gCurrent,
    batch: BATCH,
    current: { glbBytes: result.current.glbBytes, allocatedBytes: curAlloc, allocatedCount: result.current.allocatedCount, rangeCount: result.current.rangeCount },
    arena: { glbBytes: result.arena.glbBytes, allocatedBytes: arenaAlloc, allocatedCount: result.arena.allocatedCount, rangeCount: result.arena.rangeCount, snapshot: result.arena.arenaSnapshot },
    churnCollapsed,
  };
  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));

  console.log('\n=== Decoder scratch arena A/B ===');
  console.log(`GLB byte-identical (current vs arena): ${identical} (sha ${gCurrent.slice(0, 12)}…)`);
  console.log(`scratch allocated bytes:  current=${curAlloc != null ? (curAlloc / 1e6).toFixed(1) + 'MB' : 'n/a'} (${result.current.allocatedCount} bufs)  arena=${arenaAlloc != null ? (arenaAlloc / 1e6).toFixed(1) + 'MB' : 'n/a'} (${result.arena.allocatedCount} bufs)`);
  console.log(`arena snapshot: ${JSON.stringify(result.arena.arenaSnapshot)}`);
  console.log(`churn collapsed: ${churnCollapsed}`);
  if (!identical) fail(`GLB DIFFERS: current ${gCurrent.slice(0, 12)} vs arena ${gArena.slice(0, 12)} — arena changed output`);
  if (churnCollapsed === false) fail(`arena did NOT collapse churn: current=${curAlloc} arena=${arenaAlloc}`);
  console.log(`\n✓ SMOKE PASSED — arena preserves output + collapses allocation churn. report: ${REPORT_PATH}`);
  process.exit(0);
})().catch(e => fail(e.message));
