/**
 * Contract test for the dependency-safe SF3D two-stream execution graph.
 *
 * Run:
 *   node --import ./tools/wgsl-raw-loader-register.mjs \
 *     tools/test_cooperative_two_stream_contract.mjs
 */
import assert from 'node:assert/strict';

import {
  TWO_STREAM_BOUNDARY_ID,
  TWO_STREAM_DUTY_COUNT,
  TWO_STREAM_STAGE_IDS,
  defineTwoStreamManifest,
  driveTwoStreamBoundary,
} from '../src/lib/cooperative_two_stream.js';
import { TwoStreamBackbone } from '../src/lib/two_stream.js';

const expectedStageIds = [
  'setup',
  ...Array.from({ length: 4 }, (_, block) => [
    `block-${block}-fuse-in`,
    `block-${block}-basic-0`,
    `block-${block}-basic-1`,
    `block-${block}-basic-2`,
    `block-${block}-fuse-out`,
  ]).flat(),
  'final',
];

assert.deepEqual(TWO_STREAM_STAGE_IDS, expectedStageIds);
assert.equal(TWO_STREAM_DUTY_COUNT, 22);

const manifest = defineTwoStreamManifest();
const boundary = manifest.phases[0].boundaries[0];
assert.equal(boundary.boundaryId, TWO_STREAM_BOUNDARY_ID);
assert.equal(boundary.kind, 'gpu-command');
assert.equal(boundary.unit, 'two-stream-stage');
assert.equal(boundary.totalItems, 22);
assert.equal(boundary.progressWeight, 22);
assert.deepEqual(boundary.chunking, { mode: 'fixed', chunkItems: 1 });
console.log('ok  two-stream manifest declares the complete twenty-two-stage graph');

function makeCooperative(totalRanges = TWO_STREAM_DUTY_COUNT) {
  let index = 0;
  return {
    startBoundary(boundaryId) {
      assert.equal(boundaryId, TWO_STREAM_BOUNDARY_ID);
      return {
        nextRange() {
          if (index >= totalRanges) return null;
          const itemStart = index++;
          return { itemStart, itemEnd: itemStart + 1, itemCount: 1 };
        },
        async runGpuDuty(range, duty) {
          const commandBuffer = duty.encode();
          duty.submit(commandBuffer);
          assert.equal(commandBuffer.stageIndex, range.itemStart);
        },
      };
    },
  };
}

let clock = 100;
const events = [];
const driven = await driveTwoStreamBoundary(makeCooperative(), {
  encodeStage({ stageIndex, stageId }) {
    events.push(`encode:${stageIndex}:${stageId}`);
    clock += 2;
    return { stageIndex, stageId };
  },
  submitStage(commandBuffer) {
    events.push(`submit:${commandBuffer.stageIndex}`);
    clock += 1;
  },
  now: () => clock,
});

assert.equal(driven.completedDuties, 22);
assert.equal(driven.totalDuties, 22);
assert.equal(driven.telemetry.length, 22);
assert.deepEqual(
  driven.telemetry.map(entry => ({
    stageIndex: entry.stageIndex,
    stageId: entry.stageId,
    encodeMs: entry.encodeMs,
    dutyMs: entry.dutyMs,
  })),
  expectedStageIds.map((stageId, stageIndex) => ({
    stageIndex,
    stageId,
    encodeMs: 2,
    dutyMs: 3,
  })),
);
assert.equal(events.length, 44);
console.log('ok  driver encodes and submits every two-stream stage exactly once');

await assert.rejects(
  driveTwoStreamBoundary(makeCooperative(21), {
    encodeStage: ({ stageIndex }) => ({ stageIndex }),
    submitStage() {},
  }),
  /exhausted ranges before stage 21/,
);
console.log('ok  driver fails loud on missing declared work');

await assert.rejects(
  driveTwoStreamBoundary(makeCooperative(23), {
    encodeStage: ({ stageIndex }) => ({ stageIndex }),
    submitStage() {},
  }),
  /left ranges unconsumed/,
);
console.log('ok  driver fails loud on surplus declared work');

const backbone = new TwoStreamBackbone({});
const forwardState = backbone.createForwardState({}, 1297, {});
assert.throws(
  () => backbone.dispatchForwardStage({}, forwardState, 1),
  /out of order; expected 0/,
);
assert.throws(
  () => backbone.getForwardResult(forwardState),
  /incomplete at stage 0\/22/,
);
assert.throws(
  () => backbone.createForwardState({}, 0, {}),
  /N_img must be a positive safe integer/,
);
console.log('ok  executable graph rejects skipped stages and incomplete output');

console.log('\nALL COOPERATIVE TWO-STREAM CONTRACT CHECKS PASSED');
