/**
 * Contract test for dependency-safe postprocessor plane duties.
 *
 * The three triplane planes are independent but each plane's four convolutions
 * are sequential. This first cut therefore declares exactly three one-plane
 * GPU duties and proves that the shared cooperative facade visits each plane
 * once, in order, with no hidden work cap.
 *
 * Run: node tools/test_cooperative_post_processor_contract.mjs
 */
import assert from 'node:assert/strict';

import {
  POST_PROCESSOR_BOUNDARY_ID,
  definePostProcessorManifest,
  drivePostProcessorCooperativeBoundary,
} from '../src/lib/cooperative_post_processor.js';

const manifest = definePostProcessorManifest(3);
assert.equal(manifest.phases.length, 1);
const boundary = manifest.phases[0].boundaries[0];
assert.equal(boundary.boundaryId, POST_PROCESSOR_BOUNDARY_ID);
assert.equal(boundary.kind, 'gpu-command');
assert.equal(boundary.unit, 'triplane-plane');
assert.equal(boundary.totalItems, 3);
assert.equal(boundary.progressWeight, 3);
assert.deepEqual(boundary.chunking, { mode: 'fixed', chunkItems: 1 });
assert.throws(() => definePostProcessorManifest(0), /positive safe integer/);
assert.throws(() => definePostProcessorManifest(4), /exactly 3/);
console.log('ok  manifest declares exactly three one-plane duties');

const events = [];
let nextPlane = 0;
const ranges = Array.from({ length: 3 }, (_, rangeIndex) => ({
  rangeIndex,
  itemStart: rangeIndex,
  itemEnd: rangeIndex + 1,
  itemCount: 1,
}));
const cooperative = {
  startBoundary(boundaryId) {
    assert.equal(boundaryId, POST_PROCESSOR_BOUNDARY_ID);
    return {
      nextRange() {
        return ranges.shift() ?? null;
      },
      async runGpuDuty(range, duty) {
        events.push(`range:${range.itemStart}-${range.itemEnd}`);
        const commandBuffer = duty.encode();
        events.push(`encoded:${commandBuffer.plane}`);
        duty.submit(commandBuffer);
        events.push(`fenced:${range.itemStart}`);
      },
    };
  },
};

const result = await drivePostProcessorCooperativeBoundary(cooperative, {
  numPlanes: 3,
  encodePlane({ plane }) {
    assert.equal(plane, nextPlane++);
    return { plane };
  },
  submitPlane(commandBuffer) {
    events.push(`submitted:${commandBuffer.plane}`);
  },
});

assert.deepEqual(result, { completedPlanes: 3, totalPlanes: 3 });
assert.deepEqual(events, [
  'range:0-1', 'encoded:0', 'submitted:0', 'fenced:0',
  'range:1-2', 'encoded:1', 'submitted:1', 'fenced:1',
  'range:2-3', 'encoded:2', 'submitted:2', 'fenced:2',
]);
assert.equal(ranges.length, 0);
console.log('ok  driver encodes, submits, and settles every plane exactly once');

const extraRangeCooperative = {
  startBoundary() {
    let index = 0;
    return {
      nextRange() {
        if (index > 3) return null;
        const itemStart = index++;
        return { rangeIndex: itemStart, itemStart, itemEnd: itemStart + 1, itemCount: 1 };
      },
      async runGpuDuty(range, duty) {
        duty.submit(duty.encode());
      },
    };
  },
};
await assert.rejects(
  drivePostProcessorCooperativeBoundary(extraRangeCooperative, {
    numPlanes: 3,
    encodePlane: ({ plane }) => ({ plane }),
    submitPlane() {},
  }),
  /left ranges unconsumed/,
);
console.log('ok  driver fails loud when declared work is not fully consumed');

console.log('\nALL COOPERATIVE POSTPROCESSOR CONTRACT CHECKS PASSED');
