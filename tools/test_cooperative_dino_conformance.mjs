/**
 * SF3D cooperative-DINO adapter conformance test.
 *
 * Runs cranial's runWebGpuCooperativeAdapterConformance() (kit 0.1.38) over the
 * SF3D DINO boundary through all four scenarios, per directive
 * bind-sf3d-candidate-to-kit-0138-conformance. Also proves:
 *   - the shared recordDinoDispatchTrace produces the canonical 363-op /
 *     24-range trace the fingerprint hashes;
 *   - the orchestration fingerprint is stable and chunk-policy-bound;
 *   - the kit runtime version is exactly 0.1.38;
 *   - enabled==disabled output-fingerprint equivalence passes;
 *   - 24/24 coverage, no pending terminal ranges.
 *
 * Run: node --import ./tools/wgsl-raw-loader-register.mjs tools/test_cooperative_dino_conformance.mjs
 */
import assert from 'node:assert/strict';
import { execSync } from 'node:child_process';

import * as kit from '@kaminos/webgpu-inference-kit';
import { recordDinoDispatchTrace, VIT_NUM_BLOCKS } from '../src/lib/sf3d_backbone.js';
import {
  computeDinoOrchestrationFingerprint,
  runSf3dDinoConformance,
  REQUIRED_KIT_VERSION,
  SF3D_DINO_ADAPTER_ID,
} from '../src/lib/cooperative_dino_conformance.js';

const gitRev = (() => {
  try { return execSync('git rev-parse HEAD', { cwd: new URL('..', import.meta.url).pathname }).toString().trim(); }
  catch { return 'unknown'; }
})();

// --- 1. Kit is exactly 0.1.38 -------------------------------------------------
assert.equal(kit.WEBGPU_INFERENCE_KIT_VERSION, REQUIRED_KIT_VERSION,
  `kit must be exactly ${REQUIRED_KIT_VERSION}, got ${kit.WEBGPU_INFERENCE_KIT_VERSION}`);
assert.equal(typeof kit.runWebGpuCooperativeAdapterConformance, 'function',
  'kit 0.1.38 must export runWebGpuCooperativeAdapterConformance');
console.log(`ok  kit version exactly ${REQUIRED_KIT_VERSION}, conformance harness present`);

// --- 2. Shared trace is the canonical 363-op / 24-range orchestration --------
{
  const trace = recordDinoDispatchTrace({ numBlocks: VIT_NUM_BLOCKS, chunkBlocks: 1 });
  assert.equal(trace.chunks.length, 24, 'chunk=1 must produce 24 ranges');
  const opCount = trace.chunks.reduce((n, c) => n + c.ops.length, 0);
  assert.equal(opCount, 363, `canonical trace must be 363 ops, got ${opCount}`);
  assert.deepEqual(trace.output, { N: 1297, tokenH: 36, tokenW: 36, dim: 1024 },
    'output-shape identity must be exact');
  // Chunk sizes preserve the same total op count (only the cut points differ).
  for (const cb of [2, 3, 4, 8, 24]) {
    const t = recordDinoDispatchTrace({ numBlocks: VIT_NUM_BLOCKS, chunkBlocks: cb });
    const oc = t.chunks.reduce((n, c) => n + c.ops.length, 0);
    assert.equal(oc, 363, `chunk=${cb} op count must stay 363, got ${oc}`);
    assert.equal(t.chunks.length, Math.ceil(24 / cb), `chunk=${cb} range count`);
  }
  console.log('ok  shared trace: 363 ops, 24 ranges, exact output shape, stable across chunk sizes');
}

// --- 3. Fingerprint is stable + sensitive to orchestration drift -------------
{
  const a = await computeDinoOrchestrationFingerprint(1);
  const b = await computeDinoOrchestrationFingerprint(1);
  assert.equal(a.fingerprint, b.fingerprint, 'fingerprint must be deterministic');
  assert.equal(a.opCount, 363);
  assert.match(a.fingerprint, /^[0-9a-f]{64}$/, 'fingerprint must be SHA-256 hex');
  // Different chunk policy => different canonical trace => different fingerprint.
  const c4 = await computeDinoOrchestrationFingerprint(4);
  assert.notEqual(a.fingerprint, c4.fingerprint, 'chunk policy must change the fingerprint');
  console.log(`ok  fingerprint deterministic + chunk-sensitive: chunk1=${a.fingerprint.slice(0, 16)}…`);
}

// --- 4. Full adapter conformance: all four scenarios pass --------------------
{
  const report = await runSf3dDinoConformance(kit, {
    chunkBlocks: 1,
    packageVersion: '0.1.0-candidate',
    sourceRevision: gitRev,
  });

  assert.equal(report.status, 'passed', `conformance must pass; failed: ${JSON.stringify(report.summary?.failedCheckIds)}`);
  assert.equal(report.kitVersion, REQUIRED_KIT_VERSION, 'report kitVersion must be 0.1.38');
  assert.equal(report.adapterIdentity.adapterId, SF3D_DINO_ADAPTER_ID);
  assert.equal(report.scenarios.length, 4, 'four scenarios');

  const byId = Object.fromEntries(report.checks.map(c => [c.checkId, c.status]));
  for (const id of [
    'enabled-disabled-output-equivalence',
    'enabled-disabled-declared-work-equivalence',
    'no-pending-terminal-ranges',
    'resource-lifecycle',
  ]) {
    assert.equal(byId[id], 'passed', `check ${id} must pass (got ${byId[id]})`);
  }

  // Each scenario reached its expected terminal status.
  const expected = { 'cooperative-success': 'succeeded', 'disabled-success': 'succeeded', 'cancellation': 'cancelled', 'runtime-failure': 'failed' };
  for (const s of report.scenarios) {
    assert.equal(s.status, expected[s.scenario], `${s.scenario} expected ${expected[s.scenario]} got ${s.status}`);
  }

  // The two success scenarios must share the exact fingerprint (orchestration equivalence).
  const coop = report.scenarios.find(s => s.scenario === 'cooperative-success');
  const dis = report.scenarios.find(s => s.scenario === 'disabled-success');
  assert.ok(coop.outputFingerprint && coop.outputFingerprint === dis.outputFingerprint,
    'cooperative and disabled must share the orchestration fingerprint');

  console.log(`ok  conformance PASSED: 4/4 scenarios, ${report.summary.passedCheckCount}/${report.summary.checkCount} checks`);
  console.log(`    fingerprint (coop==disabled) = ${coop.outputFingerprint.slice(0, 24)}…`);
}

console.log('\nALL COOPERATIVE DINO CONFORMANCE CHECKS PASSED');
