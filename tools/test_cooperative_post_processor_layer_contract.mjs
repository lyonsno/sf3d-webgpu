/**
 * Contract test for dependency-safe intra-plane postprocessor duties.
 *
 * Each triplane follows the same exact six-stage dependency chain:
 * gather, three Conv+ReLU stages, final Conv, and PixelShuffle+copy.
 * The driver must visit all 18 stages exactly once, in order, with uncapped
 * telemetry that distinguishes host encoding from the completed GPU duty.
 *
 * Run:
 *   node --import ./tools/wgsl-raw-loader-register.mjs \
 *     tools/test_cooperative_post_processor_layer_contract.mjs
 */
import assert from 'node:assert/strict';

import {
  POST_PROCESSOR_LAYER_BOUNDARY_ID,
  POST_PROCESSOR_LAYER_DUTY_COUNT,
  POST_PROCESSOR_PLANE_STAGE_IDS,
  definePostProcessorLayerManifest,
  drivePostProcessorLayerBoundary,
} from '../src/lib/cooperative_post_processor.js';

assert.deepEqual(POST_PROCESSOR_PLANE_STAGE_IDS, [
  'gather',
  'conv-0-relu',
  'conv-1-relu',
  'conv-2-relu',
  'conv-3',
  'pixel-shuffle-copy',
]);
assert.equal(POST_PROCESSOR_LAYER_DUTY_COUNT, 18);

const manifest = definePostProcessorLayerManifest();
const boundary = manifest.phases[0].boundaries[0];
assert.equal(boundary.boundaryId, POST_PROCESSOR_LAYER_BOUNDARY_ID);
assert.equal(boundary.kind, 'gpu-command');
assert.equal(boundary.unit, 'post-processor-stage');
assert.equal(boundary.totalItems, 18);
assert.equal(boundary.progressWeight, 18);
assert.deepEqual(boundary.chunking, { mode: 'fixed', chunkItems: 1 });
console.log('ok  layer manifest declares all eighteen dependency-safe duties');

const ranges = Array.from({ length: 18 }, (_, rangeIndex) => ({
  rangeIndex,
  itemStart: rangeIndex,
  itemEnd: rangeIndex + 1,
  itemCount: 1,
}));
const events = [];
let clock = 100;
const cooperative = {
  startBoundary(boundaryId) {
    assert.equal(boundaryId, POST_PROCESSOR_LAYER_BOUNDARY_ID);
    return {
      nextRange() {
        return ranges.shift() ?? null;
      },
      async runGpuDuty(range, duty) {
        events.push(`range:${range.itemStart}`);
        const commandBuffer = duty.encode();
        duty.submit(commandBuffer);
        clock += 5;
      },
    };
  },
};

const result = await drivePostProcessorLayerBoundary(cooperative, {
  encodeStage({ dutyIndex, plane, stageIndex, stageId }) {
    events.push(`encode:${dutyIndex}:${plane}:${stageIndex}:${stageId}`);
    clock += 2;
    return { dutyIndex, plane, stageIndex, stageId };
  },
  submitStage(commandBuffer) {
    events.push(`submit:${commandBuffer.dutyIndex}`);
    clock += 1;
  },
  now: () => clock,
});

assert.equal(result.completedDuties, 18);
assert.equal(result.totalDuties, 18);
assert.equal(result.telemetry.length, 18);
assert.equal(ranges.length, 0);
assert.deepEqual(
  result.telemetry.map(entry => ({
    dutyIndex: entry.dutyIndex,
    plane: entry.plane,
    stageIndex: entry.stageIndex,
    stageId: entry.stageId,
    encodeMs: entry.encodeMs,
    dutyMs: entry.dutyMs,
  })),
  Array.from({ length: 18 }, (_, dutyIndex) => {
    const plane = Math.floor(dutyIndex / 6);
    const stageIndex = dutyIndex % 6;
    return {
      dutyIndex,
      plane,
      stageIndex,
      stageId: POST_PROCESSOR_PLANE_STAGE_IDS[stageIndex],
      encodeMs: 2,
      dutyMs: 8,
    };
  }),
);
assert.equal(events.length, 54);
console.log('ok  driver encodes, submits, and times every stage exactly once');

const shortCooperative = {
  startBoundary() {
    let index = 0;
    return {
      nextRange() {
        if (index >= 17) return null;
        const itemStart = index++;
        return { itemStart, itemEnd: itemStart + 1, itemCount: 1 };
      },
      async runGpuDuty(_range, duty) {
        duty.submit(duty.encode());
      },
    };
  },
};
await assert.rejects(
  drivePostProcessorLayerBoundary(shortCooperative, {
    encodeStage: ({ dutyIndex }) => ({ dutyIndex }),
    submitStage() {},
  }),
  /exhausted ranges before duty 17/,
);
console.log('ok  driver fails loud on missing declared work');

const extraCooperative = {
  startBoundary() {
    let index = 0;
    return {
      nextRange() {
        if (index > 18) return null;
        const itemStart = index++;
        return { itemStart, itemEnd: itemStart + 1, itemCount: 1 };
      },
      async runGpuDuty(_range, duty) {
        duty.submit(duty.encode());
      },
    };
  },
};
await assert.rejects(
  drivePostProcessorLayerBoundary(extraCooperative, {
    encodeStage: ({ dutyIndex }) => ({ dutyIndex }),
    submitStage() {},
  }),
  /left ranges unconsumed/,
);
console.log('ok  driver fails loud on surplus declared work');

console.log('\nALL COOPERATIVE POSTPROCESSOR LAYER CONTRACT CHECKS PASSED');
