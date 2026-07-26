/**
 * materialize_core contract (deterministic, no GPU).
 *
 * Proves the extracted materializeTextures + dilateTexture produce stable,
 * shape-correct output and that the extraction is self-consistent (same inputs
 * → identical bytes across runs). Guards the byte-identity the worker relies on.
 *
 * Run: node tools/test_materialize_core.mjs
 */
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { materializeTextures, dilateTexture } from '../src/lib/materialize_core.js';

const RES = 64; // small synthetic texture
const N = 200;

function synthetic(seedOffset = 0) {
  const featuresCPU = new Float32Array(N * 3);
  const normalsCPU = new Float32Array(N * 3);
  const occupiedIndices = new Uint32Array(N);
  const tbnData = new Float32Array(RES * RES * 9);
  const mask = new Uint8Array(RES * RES);
  // Deterministic pseudo-values (no Math.random — must be reproducible).
  for (let i = 0; i < N; i++) {
    const idx = (i * 7 + seedOffset) % (RES * RES);
    occupiedIndices[i] = idx;
    mask[idx] = 1;
    featuresCPU[i * 3] = ((i * 3) % 100) / 100;
    featuresCPU[i * 3 + 1] = ((i * 5) % 100) / 100;
    featuresCPU[i * 3 + 2] = ((i * 7) % 100) / 100;
    // unit-ish normal
    normalsCPU[i * 3] = 0.3; normalsCPU[i * 3 + 1] = 0.4; normalsCPU[i * 3 + 2] = 0.866;
    const b = idx * 9;
    tbnData[b] = 1; tbnData[b + 4] = 1; tbnData[b + 8] = 1; // identity TBN
  }
  return { featuresCPU, normalsCPU, occupiedIndices, tbnData, mask, resolution: RES, numOccupied: N };
}
const sha = (u8) => crypto.createHash('sha256').update(Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength)).digest('hex');

// 1. Output shapes exact.
{
  const { albedo, normalMap } = materializeTextures(synthetic());
  assert.equal(albedo.length, RES * RES * 4);
  assert.equal(normalMap.length, RES * RES * 4);
  console.log('ok  output shapes exact');
}

// 2. Determinism: same input → byte-identical output across runs.
{
  const a = materializeTextures(synthetic());
  const b = materializeTextures(synthetic());
  assert.equal(sha(a.albedo), sha(b.albedo), 'albedo deterministic');
  assert.equal(sha(a.normalMap), sha(b.normalMap), 'normalMap deterministic');
  console.log(`ok  deterministic (albedo sha ${sha(a.albedo).slice(0, 12)}…)`);
}

// 3. Albedo occupied texels reflect features; unoccupied are dilation-filled or 0.
{
  const input = synthetic();
  const { albedo } = materializeTextures(input);
  const idx0 = input.occupiedIndices[0];
  assert.equal(albedo[idx0 * 4 + 3], 255, 'occupied alpha = 255');
  console.log('ok  albedo occupied texels materialized');
}

// 4. Normal map identity-TBN encodes the world normal into [0,1].
{
  const input = synthetic();
  const { normalMap } = materializeTextures(input);
  const idx0 = input.occupiedIndices[0];
  // world normal (0.3,0.4,0.866), identity TBN → encoded round((n*0.5+0.5)*255)
  assert.equal(normalMap[idx0 * 4], Math.round((0.3 * 0.5 + 0.5) * 255));
  assert.equal(normalMap[idx0 * 4 + 1], Math.round((0.4 * 0.5 + 0.5) * 255));
  console.log('ok  normal-map TBN transform correct');
}

// 5. dilateTexture converges (empty run leaves texture unchanged after break).
{
  const tex = new Uint8Array(RES * RES * 4);
  const mask = new Uint8Array(RES * RES); // all empty
  const before = sha(tex);
  dilateTexture(tex, mask, RES, 7);
  assert.equal(sha(tex), before, 'all-empty dilation is a no-op (converges immediately)');
  console.log('ok  dilation converges on empty mask');
}

console.log('\nALL MATERIALIZE CORE CHECKS PASSED');
