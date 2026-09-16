#!/usr/bin/env node
/**
 * Parity reference provenance contract (tools/parity_compare_core.mjs).
 *
 * Review 2026-09-16 (MEDIUM): the parity smoke accepted any directory with
 * the expected filenames as PyTorch parity evidence. A reference directory is
 * evidentiary only with a manifest that binds every artifact hash, the input
 * image hash, and the model/source identities; altered, missing, or
 * mismatched artifacts must be rejected by name.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import {
  PARITY_REFERENCE_MANIFEST_SCHEMA,
  loadReferenceManifest,
  sha256File,
  verifyReferenceProvenance,
} from './parity_compare_core.mjs';

const sha = (buf) => createHash('sha256').update(buf).digest('hex');
function makeReference() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sf3d-parity-ref-'));
  const input = Buffer.from('fake-png-bytes');
  const inputPath = path.join(dir, 'input.png'); fs.writeFileSync(inputPath, input);
  const density = Buffer.from('density-npy'); fs.writeFileSync(path.join(dir, 'density.npy'), density);
  const summary = Buffer.from('{"density":{}}'); fs.writeFileSync(path.join(dir, 'summary.json'), summary);
  const extra = {};
  for (const name of ['vertex_offset.npy', 'grid_positions.npy', 'camera_embed.npy', 'scene_codes.npy']) {
    const b = Buffer.from(name); fs.writeFileSync(path.join(dir, name), b); extra[name] = { sha256: sha(b), bytes: b.length };
  }
  const manifest = {
    schema: PARITY_REFERENCE_MANIFEST_SCHEMA,
    generated_at: '2026-09-16T00:00:00Z',
    input: { path: inputPath, sha256: sha(input), bytes: input.length },
    generator: { script: 'tools/dump_parity_reference.py', script_sha256: 'abc', argv: [], cwd: dir, sf3d_webgpu: { commit: 'deadbeef', dirty: false } },
    sf3d: { repo: '/x/sf3d', commit: 'cafebabe', dirty: false },
    model: { repo_id: 'stabilityai/stable-fast-3d', snapshot_commit: 'snap', weights_sha256: 'w'.repeat(64) },
    torch: { version: '2.x', device: 'mps' },
    artifacts: { 'density.npy': { sha256: sha(density), bytes: density.length }, 'summary.json': { sha256: sha(summary), bytes: summary.length }, ...extra },
  };
  fs.writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2));
  return { dir, inputSha256: sha(input), manifest };
}

assert.equal(PARITY_REFERENCE_MANIFEST_SCHEMA, 'sf3d.parity-reference-manifest.v0');

// 1. Matching reference → ok, identities surfaced.
{
  const { dir, inputSha256 } = makeReference();
  const m = loadReferenceManifest(dir);
  const v = verifyReferenceProvenance(dir, m, { inputSha256 });
  assert.deepEqual([...v.errors], []);
  assert.equal(v.ok, true);
  assert.equal(v.identities.sf3dCommit, 'cafebabe');
  assert.equal(v.identities.modelSnapshotCommit, 'snap');
  assert.equal(v.identities.artifactCount, 6);
  assert.equal(sha256File(path.join(dir, 'density.npy')), sha(Buffer.from('density-npy')));
  console.log('ok  matching manifest accepted with identities');
}
// 2. Missing manifest → throws (never a silent default).
{
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sf3d-parity-ref-'));
  fs.writeFileSync(path.join(dir, 'summary.json'), '{}');
  assert.throws(() => loadReferenceManifest(dir), /no manifest.json/);
  console.log('ok  missing manifest refused');
}
// 3. Wrong schema → throws.
{
  const { dir } = makeReference();
  fs.writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify({ schema: 'other', artifacts: {} }));
  assert.throws(() => loadReferenceManifest(dir), /schema/);
  console.log('ok  wrong manifest schema refused');
}
// 4. Altered artifact → rejected by name.
{
  const { dir, inputSha256, manifest } = makeReference();
  fs.writeFileSync(path.join(dir, 'density.npy'), 'tampered');
  const v = verifyReferenceProvenance(dir, manifest, { inputSha256 });
  assert.equal(v.ok, false);
  assert.match(v.errors.join('\n'), /density.npy sha256 .* != manifest/);
  console.log('ok  altered .npy rejected by name');
}
// 5. Missing artifact → rejected by name.
{
  const { dir, inputSha256, manifest } = makeReference();
  fs.unlinkSync(path.join(dir, 'summary.json'));
  const v = verifyReferenceProvenance(dir, manifest, { inputSha256 });
  assert.match(v.errors.join('\n'), /summary.json missing/);
  console.log('ok  missing artifact rejected by name');
}
// 6. Input image mismatch → rejected.
{
  const { dir, manifest } = makeReference();
  const v = verifyReferenceProvenance(dir, manifest, { inputSha256: 'f'.repeat(64) });
  assert.match(v.errors.join('\n'), /input image sha256 .* != manifest/);
  console.log('ok  input image mismatch rejected');
}
// 7. Manifest without model/source identity → rejected (not evidentiary).
{
  const { dir, inputSha256, manifest } = makeReference();
  const v = verifyReferenceProvenance(dir, { ...manifest, sf3d: { commit: null }, model: { repo_id: null } }, { inputSha256 });
  assert.match(v.errors.join('\n'), /sf3d source commit missing/);
  assert.match(v.errors.join('\n'), /model identity missing/);
  console.log('ok  manifest without source/model identity rejected');
}

// 8. Completeness (r2 MEDIUM, 2026-09-16): every identity the report advertises
//    is required, and the full artifact set the comparison needs must be listed
//    and hashed; a partial manifest is not evidentiary.
{
  const { PARITY_REQUIRED_ARTIFACTS } = await import('./parity_compare_core.mjs');
  assert.deepEqual([...PARITY_REQUIRED_ARTIFACTS], ['summary.json', 'density.npy', 'vertex_offset.npy', 'grid_positions.npy', 'camera_embed.npy', 'scene_codes.npy']);
  const { dir, inputSha256, manifest } = makeReference();
  const twoOnly = { ...manifest, artifacts: { 'density.npy': manifest.artifacts['density.npy'], 'summary.json': manifest.artifacts['summary.json'] } };
  const partial = verifyReferenceProvenance(dir, twoOnly, { inputSha256 });
  assert.equal(partial.ok, false);
  assert.match(partial.errors.join('\n'), /required artifacts missing from manifest: vertex_offset.npy, grid_positions.npy, camera_embed.npy, scene_codes.npy/);
  // Missing identities are each named.
  const stripped = { ...manifest, generated_at: null, generator: { ...manifest.generator, script_sha256: null, sf3d_webgpu: { commit: null } }, model: { repo_id: 'stabilityai/stable-fast-3d' }, torch: null };
  const v = verifyReferenceProvenance(dir, stripped, { inputSha256 });
  for (const re of [/generation timestamp missing/, /generator script hash missing/, /generator source commit missing/, /model snapshot commit missing/, /model weights hash missing/, /torch identity missing/]) {
    assert.match(v.errors.join('\n'), re);
  }
  console.log('ok  incomplete manifests (missing identities / required artifacts) are not evidentiary');
}

console.log('\nPARITY REFERENCE PROVENANCE CONTRACT PASSED');
