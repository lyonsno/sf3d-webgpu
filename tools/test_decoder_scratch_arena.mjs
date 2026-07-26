/**
 * Decoder scratch arena contract (deterministic, no GPU).
 *
 * Fail-first: proves the arena pre-allocates one buffer per proven slot, reuses
 * the same physical buffer across acquires, fails loud on unknown slot and on
 * capacity overflow (never silent truncation), and retires all buffers on
 * destroy. Uses a fake device that records buffer creation/destruction.
 *
 * Run: node tools/test_decoder_scratch_arena.mjs
 */
import assert from 'node:assert/strict';
import {
  createDecoderScratchArena,
  decoderArenaCapacityBytes,
  DECODER_BAKE_SLOT_GRAPH,
} from '../src/lib/decoder_scratch_arena.js';

// gpu.js touches GPUBufferUsage inside createEmptyBuffer; shim it.
globalThis.GPUBufferUsage = globalThis.GPUBufferUsage || { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8, UNIFORM: 16 };
let created = 0, destroyed = 0;
const fakeDevice = {
  createBuffer({ size, label }) {
    created++;
    return { size, label, _destroyed: false, destroy() { this._destroyed = true; destroyed++; } };
  },
};

const MAX = 16384;

// 1. Capacity matches the proven assay numbers.
{
  const at4096 = decoderArenaCapacityBytes(4096);
  const at16384 = decoderArenaCapacityBytes(16384);
  assert.equal(DECODER_BAKE_SLOT_GRAPH.length, 30, 'proven graph is 30 slots');
  // ~16.9MB / ~67.4MB (allow small alignment slack)
  assert.ok(Math.abs(at4096 / 1e6 - 16.9) < 0.2, `4096 capacity ~16.9MB, got ${(at4096 / 1e6).toFixed(2)}MB`);
  assert.ok(Math.abs(at16384 / 1e6 - 67.4) < 0.4, `16384 capacity ~67.4MB, got ${(at16384 / 1e6).toFixed(2)}MB`);
  console.log(`ok  capacity: ${(at4096 / 1e6).toFixed(1)}MB@4096  ${(at16384 / 1e6).toFixed(1)}MB@16384`);
}

// 2. Pre-allocates exactly one buffer per slot.
{
  created = 0;
  const arena = createDecoderScratchArena(fakeDevice, { maxBatch: MAX });
  assert.equal(arena.slotCount, 30);
  assert.equal(created, 30, 'one buffer per slot pre-allocated');
  assert.equal(arena.totalBytes, decoderArenaCapacityBytes(MAX));
  console.log(`ok  pre-allocates 30 buffers, ${(arena.totalBytes / 1e6).toFixed(1)}MB total`);
}

// 3. acquire returns the SAME physical buffer across ranges (reuse, not realloc).
{
  created = 0;
  const arena = createDecoderScratchArena(fakeDevice, { maxBatch: MAX });
  const preCount = created;
  const b1 = arena.acquire('sampled:XY', 40 * 4 * 100);   // range 1
  const b2 = arena.acquire('sampled:XY', 40 * 4 * 200);   // range 2, smaller N still same slot
  assert.equal(b1, b2, 'same slot returns same physical buffer across ranges');
  assert.equal(created, preCount, 'acquire does NOT allocate new buffers');
  console.log('ok  acquire reuses the same physical buffer across ranges (no realloc)');
}

// 4. Fail loud: unknown slot.
{
  const arena = createDecoderScratchArena(fakeDevice, { maxBatch: MAX });
  assert.throws(() => arena.acquire('no-such-slot', 16), /no slot "no-such-slot"/);
  console.log('ok  unknown slot rejects');
}

// 5. Fail loud: capacity overflow (never silent truncation).
{
  const arena = createDecoderScratchArena(fakeDevice, { maxBatch: 4096 });
  // sampled:XY capacity at 4096 = 160*4096 = 655360; ask for more.
  assert.throws(() => arena.acquire('sampled:XY', 655360 + 4), /overflow: need 655364 > capacity 655360/);
  console.log('ok  capacity overflow rejects (no silent truncation)');
}

// 6. Exact fit at maxBatch is allowed.
{
  const arena = createDecoderScratchArena(fakeDevice, { maxBatch: 4096 });
  assert.doesNotThrow(() => arena.acquire('sampled:XY', 655360), 'exact-capacity acquire allowed');
  console.log('ok  exact-capacity acquire allowed');
}

// 7. destroy retires all buffers; post-destroy acquire rejects.
{
  destroyed = 0;
  const arena = createDecoderScratchArena(fakeDevice, { maxBatch: MAX });
  const r = arena.destroy();
  assert.equal(r.retired, 30, 'all 30 buffers retired');
  assert.equal(destroyed, 30);
  assert.throws(() => arena.acquire('scaledPos', 16), /destroyed/);
  assert.deepEqual(arena.destroy(), { retired: 0 }, 'double destroy is inert');
  console.log('ok  destroy retires all 30 buffers; post-destroy acquire rejects');
}

// 8. Bad maxBatch rejects.
{
  assert.throws(() => createDecoderScratchArena(fakeDevice, { maxBatch: 0 }), /positive safe integer/);
  assert.throws(() => createDecoderScratchArena(fakeDevice, { maxBatch: -1 }), /positive safe integer/);
  console.log('ok  invalid maxBatch rejects');
}

console.log('\nALL DECODER SCRATCH ARENA CHECKS PASSED');
