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
import { readImageInput, sha256Tree } from './witness_source_identity.mjs';

const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'sf3d-failure-report-'));
const journalParent = path.join(os.homedir(), '.local/state/sf3d');
fs.mkdirSync(journalParent, { recursive: true });
const journalRoot = fs.mkdtempSync(path.join(journalParent, 'test-witness-journal-'));
const run = (script, args, env = {}) => spawnSync(process.execPath, [path.join(REPO, 'tools', script), ...args], {
  cwd: REPO, env: { ...process.env, ...env }, encoding: 'utf8', timeout: 120000,
});
const readReport = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
const sha256File = (filePath) => new Promise((resolve, reject) => {
  const hash = createHash('sha256');
  const stream = fs.createReadStream(filePath);
  stream.on('data', chunk => hash.update(chunk));
  stream.on('error', reject);
  stream.on('end', () => resolve(hash.digest('hex')));
});

// Source identity must stay bound to the exact captured bytes even if the
// pathname changes after capture, because those same bytes feed the browser.
{
  const imagePath = path.join(tmp, 'mutable-image.png');
  const capturedBytes = Buffer.from('image bytes used for identity and browser submission');
  fs.writeFileSync(imagePath, capturedBytes);
  const captured = readImageInput(imagePath);
  fs.writeFileSync(imagePath, Buffer.from('replacement bytes at same path'));
  assert.equal(captured.sha256, createHash('sha256').update(capturedBytes).digest('hex'));
  assert.equal(Buffer.from(captured.bytes.toString('base64'), 'base64').compare(capturedBytes), 0,
    'browser encoding uses the captured image bytes rather than rereading the path');
  assert.notEqual(captured.sha256, await sha256File(imagePath), 'later path contents cannot silently inherit the captured digest');
}

// A package-tree digest must distinguish same-version installed files, not
// merely package.json/version identity.
{
  const kitRoot = path.join(tmp, 'kit-tree-fixture');
  fs.mkdirSync(path.join(kitRoot, 'dist'), { recursive: true });
  fs.writeFileSync(path.join(kitRoot, 'package.json'), JSON.stringify({ name: '@kaminos/webgpu-inference-kit', version: '0.1.52' }));
  fs.writeFileSync(path.join(kitRoot, 'dist', 'entry.js'), 'export const mode = "before";');
  const original = await sha256Tree(kitRoot);
  fs.writeFileSync(path.join(kitRoot, 'dist', 'entry.js'), 'export const mode = "after";');
  const changed = await sha256Tree(kitRoot);
  assert.equal(JSON.parse(fs.readFileSync(path.join(kitRoot, 'package.json'), 'utf8')).version, '0.1.52');
  assert.notEqual(changed.sha256, original.sha256,
    'changed installed file changes tree identity while package version remains constant');
  assert.deepEqual(changed.files, original.files, 'the difference is file content, not package layout');
}

// --- Product-route witness ---
const cases = [
  ['unknown arm', ['--arm', 'definitely-not-an-arm'], {}, 'arguments', /unknown --arm/],
  ['missing image', ['--image', '/nonexistent/image.png'], {}, 'input', /image not found/],
  ['source identity', ['--allow-dirty'], { SF3D_WITNESS_INJECT_FAILURE: 'source-identity' }, 'source-identity', /injected failure at source-identity/],
  ['kit identity', ['--allow-dirty'], { SF3D_WITNESS_INJECT_FAILURE: 'kit-identity' }, 'kit-identity', /injected failure at kit-identity/],
  ['vite start', ['--allow-dirty'], { SF3D_WITNESS_INJECT_FAILURE: 'vite-start' }, 'vite-start', /injected failure at vite-start/],
  ['browser launch', ['--allow-dirty'], { SF3D_WITNESS_INJECT_FAILURE: 'browser-launch' }, 'browser-launch', /injected failure at browser-launch/],
  ['browser evaluation', ['--allow-dirty'], { SF3D_WITNESS_INJECT_FAILURE: 'browser-evaluation' }, 'browser-evaluation', /injected failure at browser-evaluation/],
  // Fail-first source/input/package identity assertions. The pre-fix harness
  // ignores these expectations and is stopped safely at vite-start, before
  // Chromium launches or any model/GPU work can begin.
  ['wrong expected source revision', ['--allow-dirty', '--expected-commit', '0'.repeat(40)], { SF3D_WITNESS_INJECT_FAILURE: 'vite-start' }, 'source-identity', /source commit .* does not match requested/],
  ['wrong expected input digest', ['--allow-dirty', '--expected-image-sha', '0'.repeat(64)], { SF3D_WITNESS_INJECT_FAILURE: 'vite-start' }, 'source-identity', /input image SHA-256 .* does not match requested/],
  ['wrong expected installed kit tree', ['--allow-dirty', '--expected-kit-tree-sha256', '0'.repeat(64)], { SF3D_WITNESS_INJECT_FAILURE: 'vite-start' }, 'kit-identity', /installed kit tree SHA-256 .* does not match requested/],
];
for (const [name, args, env, phase, re] of cases) {
  const caseSlug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
  const report = path.join(tmp, `product-${caseSlug}.json`);
  const journal = path.join(journalRoot, `product-${caseSlug}.jsonl`);
  const r = run('smoke_product_route.mjs', [...args, '--report', report, '--journal', journal], env);
  assert.notEqual(r.status, 0, `${name}: harness must fail`);
  assert.ok(fs.existsSync(report), `${name}: failure report must exist (stderr: ${r.stderr.slice(0, 300)})`);
  const d = readReport(report);
  assert.equal(d.ok, false); assert.equal(d.schema, 'sf3d.product-route-witness-failure.v0');
  assert.equal(d.failurePhase, phase, `${name}: names its phase (stderr: ${r.stderr.slice(0, 500)})`);
  assert.match(d.error.message, re, `${name}: reports its expected failure (stderr: ${r.stderr.slice(0, 500)})`);
  assert.equal(d.requested.report, report, 'requested report path recorded');
  if (phase === 'kit-identity' || phase === 'vite-start' || phase === 'browser-launch' || phase === 'browser-evaluation') {
    assert.match(d.source.commit, /^[0-9a-f]{40}$/, `${name}: effective source identity recorded once established`);
  }
  if (phase === 'browser-launch') {
    assert.equal(d.requested.protocolTimeoutMs, 0, 'long browser evaluation has an explicit disabled protocol deadline');
    assert.ok(fs.existsSync(journal), 'parent journal survives a browser-launch failure before primary output');
    const events = fs.readFileSync(journal, 'utf8').trimEnd().split('\n').map(line => JSON.parse(line));
    assert.equal(events[0].type, 'invocation-requested');
    assert.ok(events.some(event => event.type === 'effective-identity'));
    const effectiveKit = events.find(event => event.type === 'effective-package-identity');
    assert.match(effectiveKit.payload.installedTreeSha256, /^[0-9a-f]{64}$/);
    assert.equal(effectiveKit.payload.version, d.source.kitVersion);
    assert.ok(events.some(event => event.type === 'phase-completed' && event.payload.phase === 'kit-identity'));
    assert.ok(events.some(event => event.type === 'phase-entered' && event.payload.phase === 'browser-launch'));
    assert.ok(events.some(event => event.payload.phase === 'browser-launch' && event.payload.memoryObservation?.method));
    assert.equal(events.at(-1).type, 'terminal');
    assert.equal(events.at(-1).payload.failurePhase, 'browser-launch');
    assert.equal(events.at(-1).payload.status, 'failed');
    assert.equal(d.parentPhaseJournal.lastEnteredPhase, 'browser-launch');
    assert.equal(d.parentPhaseJournal.lastCompletedPhase, 'vite-start');
    assert.equal(d.parentPhaseJournal.integrityOk, true);
  }
  if (phase === 'browser-evaluation') {
    assert.equal(d.requested.protocolTimeoutMs, 0, 'browser evaluation uses explicit deadline configuration');
    assert.ok(fs.existsSync(journal), 'parent journal survives a browser evaluation failure before primary output');
    const replay = run('replay_parent_phase_journal.mjs', ['--journal', journal]);
    assert.equal(replay.status, 0, `browser-evaluation journal replays (stderr: ${replay.stderr})`);
    const replayed = JSON.parse(replay.stdout);
    assert.equal(replayed.integrityOk, true);
    assert.equal(replayed.lastEnteredPhase, 'browser-evaluation');
    assert.equal(replayed.lastCompletedPhase, 'browser-launch');
    assert.equal(d.parentPhaseJournal.integrityOk, true);
    assert.ok(d.parentPhaseJournal.eventCount >= replayed.eventCount, 'failure report includes the replayed journal summary');
    const events = fs.readFileSync(journal, 'utf8').trimEnd().split('\n').map(line => JSON.parse(line));
    const identity = events.find(event => event.type === 'effective-identity').payload;
    assert.equal(identity.harnessRouteClass, 'sf3d.image-to-mesh.webgpu-local.v0');
    assert.equal(identity.routeId, undefined, 'pre-navigation state must not assert an effective route id');
    assert.equal(identity.effectiveProducerDeviceRoute?.status, 'unobserved');
  }
  if (name === 'wrong expected source revision') {
    assert.equal(d.requested.expectedCommit, '0'.repeat(40));
    assert.match(d.source.commit, /^[0-9a-f]{40}$/);
    assert.notEqual(d.source.commit, d.requested.expectedCommit);
  }
  if (name === 'wrong expected input digest') {
    assert.equal(d.requested.expectedImageSha256, '0'.repeat(64));
    assert.match(d.source.input.sha256, /^[0-9a-f]{64}$/);
    assert.notEqual(d.source.input.sha256, d.requested.expectedImageSha256);
  }
  if (name === 'wrong expected installed kit tree') {
    assert.equal(d.requested.expectedKitTreeSha256, '0'.repeat(64));
    assert.match(d.source.kitIdentity.installedTreeSha256, /^[0-9a-f]{64}$/);
    assert.equal(d.source.kitIdentity.version, d.source.kitVersion, 'tree identity disambiguates same-version installs');
    assert.notEqual(d.source.kitIdentity.installedTreeSha256, d.requested.expectedKitTreeSha256);
  }
  console.log(`ok  product-route witness: ${name} → durable report at phase ${phase}`);
}

// The browser is deliberately not launched: this proves the exact mounted
// artifact is content-identified before any model work begins.
const weightPath = path.join(REPO, 'public/weights.bin');
const createdPublic = !fs.existsSync(path.dirname(weightPath));
fs.mkdirSync(path.dirname(weightPath), { recursive: true });
const createdFixtureWeights = !fs.existsSync(weightPath);
if (createdFixtureWeights) fs.writeFileSync(weightPath, Buffer.from('sf3d-weight-hash-fixture-v0'));
try {
  const report = path.join(tmp, 'product-weight-source-identity.json');
  const journal = path.join(journalRoot, 'product-weight-source-identity.jsonl');
  const expectedSha = await sha256File(weightPath);
  const r = run('smoke_product_route.mjs', ['--allow-dirty', '--report', report, '--journal', journal], { SF3D_WITNESS_INJECT_FAILURE: 'kit-identity' });
  assert.notEqual(r.status, 0, 'injected source-identity failure must fail');
  const d = readReport(report);
  assert.equal(d.failurePhase, 'kit-identity');
  assert.equal(d.source.weightArtifact.sha256, expectedSha, 'failure report binds the exact weight bytes');
  assert.equal(d.source.weightArtifact.sha256Status, 'computed');
  assert.equal(d.requested.expectedWeightsSha256, null);
  const mismatchReport = path.join(tmp, 'product-weight-digest-mismatch.json');
  const mismatchJournal = path.join(journalRoot, 'product-weight-digest-mismatch.jsonl');
  const mismatch = run('smoke_product_route.mjs', [
    '--allow-dirty', '--expected-weights-sha', '0'.repeat(64), '--report', mismatchReport, '--journal', mismatchJournal,
  ]);
  assert.notEqual(mismatch.status, 0, 'an incorrect expected weight identity must reject');
  const md = readReport(mismatchReport);
  assert.equal(md.failurePhase, 'source-identity');
  assert.equal(md.requested.expectedWeightsSha256, '0'.repeat(64));
  assert.equal(md.source.weightArtifact.sha256, expectedSha, 'mismatch report preserves the observed digest');
  assert.match(md.error.message, /does not match requested/);
} finally {
  if (createdFixtureWeights) fs.unlinkSync(weightPath);
  if (createdPublic) fs.rmdirSync(path.dirname(weightPath));
}

if (process.env.SF3D_FAILURE_REPORT_PRODUCT_ONLY === '1') {
  fs.rmSync(journalRoot, { recursive: true, force: true });
  console.log('\nPRODUCT WITNESS FAILURE-JOURNAL CONTRACT PASSED');
  process.exit(0);
}

// --- Parity smoke ---
const demoImage = path.join(REPO, 'public/demo_chair.png');
const REQUIRED = ['summary.json', 'density.npy', 'vertex_offset.npy', 'grid_positions.npy', 'camera_embed.npy', 'scene_codes.npy'];
function writeBoundReference(prefix, { tamper = null } = {}) {
  const refDir = fs.mkdtempSync(path.join(tmp, prefix));
  const sha = (b) => createHash('sha256').update(b).digest('hex');
  const artifacts = {};
  for (const name of REQUIRED) {
    const b = Buffer.from(name); fs.writeFileSync(path.join(refDir, name), b); artifacts[name] = { sha256: sha(b), bytes: b.length };
  }
  if (tamper) artifacts[tamper] = { sha256: sha(Buffer.from('other')), bytes: 5 };
  fs.writeFileSync(path.join(refDir, 'manifest.json'), JSON.stringify({
    schema: 'sf3d.parity-reference-manifest.v0', generated_at: '2026-09-16T00:00:00Z', input: { sha256: sha(fs.readFileSync(demoImage)) },
    sf3d: { commit: 'cafebabe' }, model: { repo_id: 'stabilityai/stable-fast-3d', snapshot_commit: 'snap', weights_sha256: 'w' },
    generator: { script: 'tools/dump_parity_reference.py', script_sha256: 's', sf3d_webgpu: { commit: 'g' } }, torch: { version: '2.x' }, artifacts,
  }));
  return refDir;
}
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
  const refDir = writeBoundReference('ref-tampered-', { tamper: 'density.npy' });
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
  const refDir = writeBoundReference('ref-ok-');
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

{
  // r2 MEDIUM (2026-09-16): a REAL Vite spawn failure (not an injected throw)
  // must be caught and reported at vite-start. PATH without npx → spawn ENOENT
  // arrives as an asynchronous child 'error' event; the harness must own it.
  const refDir = writeBoundReference('ref-spawnfail-');
  // A PATH that still resolves git (source identity runs first) but has no npx.
  const binDir = fs.mkdtempSync(path.join(tmp, 'bin-'));
  const gitPath = spawnSync('/usr/bin/which', ['git'], { encoding: 'utf8' }).stdout.trim() || '/usr/bin/git';
  fs.symlinkSync(gitPath, path.join(binDir, 'git'));
  const report = path.join(tmp, 'parity-spawnfail.json');
  const r = run('smoke_parity.mjs', ['--reference', refDir, '--report', report], { IMAGE: demoImage, PATH: binDir });
  assert.notEqual(r.status, 0);
  assert.ok(fs.existsSync(report), `real spawn failure must leave a report (stderr: ${r.stderr.slice(0, 300)})`);
  const d = readReport(report);
  assert.equal(d.failure.phase, 'vite-start');
  assert.match(d.failure.message, /ENOENT|spawn/);
  assert.ok(d.webgpuIdentity?.commit, 'established identities travel with the failure report');
  // r3: the verified reference provenance travels too.
  assert.equal(d.provenance?.ok, true);
  assert.equal(d.provenance.identities.generatedAt, '2026-09-16T00:00:00Z');
  assert.equal(d.provenance.identities.modelSnapshotCommit, 'snap');
  assert.equal(d.provenance.identities.modelWeightsSha256, 'w');
  assert.equal(d.provenance.identities.generatorScript, 'tools/dump_parity_reference.py');
  assert.equal(d.provenance.identities.generatorCommit, 'g');
  for (const name of REQUIRED) assert.ok(d.provenance.manifestArtifacts.includes(name), `${name} in manifestArtifacts`);
  console.log('ok  parity smoke: real Vite spawn failure → durable report at vite-start with identities');
}

console.log('\nWITNESS FAILURE REPORT CONTRACT PASSED');
fs.rmSync(journalRoot, { recursive: true, force: true });
