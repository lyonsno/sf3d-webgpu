#!/usr/bin/env node
/**
 * Durable failure-report contract for the two browser harnesses
 * (tools/smoke_product_route.mjs, tools/smoke_parity.mjs).
 *
 * Review 2026-09-16 (MEDIUM): both harnesses could die before their primary
 * artifact — argument/input validation, source/kit identity, Vite start,
 * browser launch — without writing the failure report they promise. Every
 * phase before the primary output must now leave a report naming the phase.
 * Failures are injected deterministically through SF3D_WITNESS_INJECT_FAILURE
 * / SF3D_PARITY_INJECT_FAILURE, which throw at the start of the named phase.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'sf3d-failure-report-'));
const run = (script, args, env = {}) => spawnSync(process.execPath, [path.join(REPO, 'tools', script), ...args], {
  cwd: REPO, env: { ...process.env, ...env }, encoding: 'utf8', timeout: 120000,
});
const readReport = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));

// --- Product-route witness ---
const cases = [
  ['unknown arm', ['--arm', 'definitely-not-an-arm'], {}, 'arguments', /unknown --arm/],
  ['missing image', ['--image', '/nonexistent/image.png'], {}, 'input', /image not found/],
  ['source identity', ['--allow-dirty'], { SF3D_WITNESS_INJECT_FAILURE: 'source-identity' }, 'source-identity', /injected failure at source-identity/],
  ['kit identity', ['--allow-dirty'], { SF3D_WITNESS_INJECT_FAILURE: 'kit-identity' }, 'kit-identity', /injected failure at kit-identity/],
  ['vite start', ['--allow-dirty'], { SF3D_WITNESS_INJECT_FAILURE: 'vite-start' }, 'vite-start', /injected failure at vite-start/],
  ['browser launch', ['--allow-dirty'], { SF3D_WITNESS_INJECT_FAILURE: 'browser-launch' }, 'browser-launch', /injected failure at browser-launch/],
];
for (const [name, args, env, phase, re] of cases) {
  const report = path.join(tmp, `product-${phase}.json`);
  const r = run('smoke_product_route.mjs', [...args, '--report', report], env);
  assert.notEqual(r.status, 0, `${name}: harness must fail`);
  assert.ok(fs.existsSync(report), `${name}: failure report must exist (stderr: ${r.stderr.slice(0, 300)})`);
  const d = readReport(report);
  assert.equal(d.ok, false); assert.equal(d.schema, 'sf3d.product-route-witness-failure.v0');
  assert.equal(d.failurePhase, phase, `${name}: names its phase`);
  assert.match(d.error.message, re);
  assert.equal(d.requested.report, report, 'requested report path recorded');
  if (phase === 'kit-identity' || phase === 'vite-start' || phase === 'browser-launch') {
    assert.match(d.source.commit, /^[0-9a-f]{40}$/, `${name}: effective source identity recorded once established`);
  }
  console.log(`ok  product-route witness: ${name} → durable report at phase ${phase}`);
}

// --- Parity smoke ---
const demoImage = path.join(REPO, 'public/demo_chair.png');
{
  // Unbound reference directory: no manifest → refused at reference-provenance.
  const refDir = fs.mkdtempSync(path.join(tmp, 'ref-nomanifest-'));
  fs.writeFileSync(path.join(refDir, 'summary.json'), '{}');
  const report = path.join(tmp, 'parity-nomanifest.json');
  const r = run('smoke_parity.mjs', ['--reference', refDir, '--report', report], { IMAGE: demoImage });
  assert.notEqual(r.status, 0);
  const d = readReport(report);
  assert.equal(d.evidentiary, false); assert.equal(d.failure.phase, 'reference-provenance');
  assert.match(d.failure.message, /no manifest.json/);
  assert.equal(d.parity, null);
  console.log('ok  parity smoke: unbound reference → non-evidentiary report at reference-provenance');
}
{
  // Tampered artifact under a real manifest → refused by name.
  const refDir = fs.mkdtempSync(path.join(tmp, 'ref-tampered-'));
  const density = Buffer.from('density'); fs.writeFileSync(path.join(refDir, 'density.npy'), density);
  const sha = (b) => createHash('sha256').update(b).digest('hex');
  fs.writeFileSync(path.join(refDir, 'manifest.json'), JSON.stringify({
    schema: 'sf3d.parity-reference-manifest.v0', input: { sha256: sha(fs.readFileSync(demoImage)) },
    sf3d: { commit: 'cafebabe' }, model: { repo_id: 'stabilityai/stable-fast-3d' },
    artifacts: { 'density.npy': { sha256: sha(Buffer.from('other')) } },
  }));
  const report = path.join(tmp, 'parity-tampered.json');
  const r = run('smoke_parity.mjs', ['--reference', refDir, '--report', report], { IMAGE: demoImage });
  assert.notEqual(r.status, 0);
  const d = readReport(report);
  assert.equal(d.failure.phase, 'reference-provenance');
  assert.match(d.failure.message, /density.npy sha256/);
  console.log('ok  parity smoke: tampered reference artifact refused by name');
}
{
  // Injected failure at vite-start after provenance passed → report names vite-start.
  const refDir = fs.mkdtempSync(path.join(tmp, 'ref-ok-'));
  const density = Buffer.from('density'); fs.writeFileSync(path.join(refDir, 'density.npy'), density);
  const sha = (b) => createHash('sha256').update(b).digest('hex');
  fs.writeFileSync(path.join(refDir, 'manifest.json'), JSON.stringify({
    schema: 'sf3d.parity-reference-manifest.v0', input: { sha256: sha(fs.readFileSync(demoImage)) },
    sf3d: { commit: 'cafebabe' }, model: { repo_id: 'stabilityai/stable-fast-3d' },
    artifacts: { 'density.npy': { sha256: sha(density) } },
  }));
  for (const phase of ['vite-start', 'browser-launch']) {
    const report = path.join(tmp, `parity-${phase}.json`);
    const r = run('smoke_parity.mjs', ['--reference', refDir, '--report', report], { IMAGE: demoImage, SF3D_PARITY_INJECT_FAILURE: phase });
    assert.notEqual(r.status, 0);
    const d = readReport(report);
    assert.equal(d.failure.phase, phase, `parity ${phase} (stderr: ${r.stderr.slice(0, 300)})`);
    assert.match(d.failure.message, new RegExp(`injected failure at ${phase}`));
    console.log(`ok  parity smoke: injected ${phase} → durable report`);
  }
}

console.log('\nWITNESS FAILURE REPORT CONTRACT PASSED');
