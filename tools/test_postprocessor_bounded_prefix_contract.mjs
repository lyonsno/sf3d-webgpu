/**
 * Bounded-prefix completion contract for the postprocessor channel-range
 * boundary (kit >=0.1.41). Fail-first: proves runCooperativePostProcessor's
 * bounded-prefix guardrails reject every unlawful configuration BEFORE any GPU
 * work, and accepts the one lawful shape. These validations throw synchronously
 * at the top of the function, so a dummy device (never used) suffices.
 *
 * Run: node --import ./tools/wgsl-raw-loader-register.mjs
 *      tools/test_postprocessor_bounded_prefix_contract.mjs
 */
import assert from 'node:assert/strict';
import { runCooperativePostProcessor } from '../src/lib/cooperative_post_processor.js';

// A device stub that throws if actually touched — proves rejection happens
// during validation, before any GPU use.
const trapDevice = new Proxy({}, {
  get() { throw new Error('device was touched — validation should have thrown first'); },
});
const base = { device: trapDevice, triplanesBuf: {}, weights: {} };

// 1. bounded-prefix WITHOUT channel-range → reject (adaptive/plane/layer stay strict).
for (const g of ['plane', 'layer']) {
  await assert.rejects(
    () => runCooperativePostProcessor({ ...base, dutyGranularity: g, completionPolicy: 'bounded-prefix', maxInFlightGpuDuties: 2 }),
    /bounded-prefix completion is only supported for channel-range/,
    `bounded-prefix must reject dutyGranularity=${g}`);
}
console.log('ok  bounded-prefix rejected for plane/layer granularity');

// 2. bounded-prefix WITHOUT cooperative scheduling → reject.
await assert.rejects(
  () => runCooperativePostProcessor({ ...base, dutyGranularity: 'channel-range', schedulingMode: 'disabled', completionPolicy: 'bounded-prefix', maxInFlightGpuDuties: 2 }),
  /bounded-prefix completion requires cooperative scheduling/);
console.log('ok  bounded-prefix rejected for disabled scheduling');

// 3. bounded-prefix with bad maxInFlightGpuDuties → reject (missing / <=0 / non-int).
for (const bad of [undefined, null, 0, -1, 1.5]) {
  await assert.rejects(
    () => runCooperativePostProcessor({ ...base, dutyGranularity: 'channel-range', completionPolicy: 'bounded-prefix', maxInFlightGpuDuties: bad }),
    /bounded-prefix completion requires a positive maxInFlightGpuDuties/,
    `maxInFlightGpuDuties=${bad} must reject`);
}
console.log('ok  bounded-prefix rejects missing/invalid maxInFlightGpuDuties');

// 4. maxInFlightGpuDuties WITHOUT bounded-prefix → reject.
await assert.rejects(
  () => runCooperativePostProcessor({ ...base, dutyGranularity: 'channel-range', completionPolicy: 'strict-prefix', maxInFlightGpuDuties: 2 }),
  /maxInFlightGpuDuties is available only with bounded-prefix/);
console.log('ok  maxInFlightGpuDuties rejected without bounded-prefix');

// 5. unknown completionPolicy → reject.
await assert.rejects(
  () => runCooperativePostProcessor({ ...base, dutyGranularity: 'channel-range', completionPolicy: 'nonsense' }),
  /completionPolicy must be strict-prefix or bounded-prefix/);
console.log('ok  unknown completionPolicy rejected');

// 6. LAWFUL shape (channel-range + cooperative + bounded-prefix + depth 2) must
//    pass validation and proceed PAST the guards (it will then fail on the
//    trapDevice — which proves validation accepted and it moved to GPU work).
await assert.rejects(
  () => runCooperativePostProcessor({ ...base, dutyGranularity: 'channel-range', schedulingMode: 'cooperative', completionPolicy: 'bounded-prefix', maxInFlightGpuDuties: 2 }),
  /device was touched/,
  'lawful bounded-prefix config must pass validation and reach GPU work');
console.log('ok  lawful bounded-prefix (channel-range, cooperative, depth 2) passes validation');

// 7. Default (no completionPolicy) is strict-prefix and reaches GPU work (no
//    policy rejection), for channel-range.
await assert.rejects(
  () => runCooperativePostProcessor({ ...base, dutyGranularity: 'channel-range', schedulingMode: 'cooperative' }),
  /device was touched/,
  'default strict-prefix must pass validation');
console.log('ok  default strict-prefix passes validation (unchanged behavior)');

console.log('\nALL POSTPROCESSOR BOUNDED-PREFIX CONTRACT CHECKS PASSED');
