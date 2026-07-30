/**
 * Contract test for exact postprocessor output-channel duties.
 *
 * The channel plan retains gather and PixelShuffle as single duties while
 * splitting every convolution into contiguous output-channel ranges. The
 * driver must reject mutated coverage before encoding any GPU work.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  POST_PROCESSOR_CHANNEL_BOUNDARY_ID,
  createPostProcessorChannelDutyPlan,
  definePostProcessorChannelManifest,
  drivePostProcessorChannelBoundary,
} from '../src/lib/cooperative_post_processor.js';
import {
  evaluatePostProcessorSmokeAcceptance,
} from './post_processor_smoke_acceptance.mjs';
import {
  readJsonReport,
  writeJsonReportAtomic,
} from './json_report_atomic.mjs';

const reportDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'sf3d-post-report-'));
try {
  const reportPath = path.join(reportDirectory, 'report.json');
  fs.writeFileSync(reportPath, '');
  writeJsonReportAtomic(reportPath, { schema: 'test', ok: true });
  assert.deepEqual(readJsonReport(reportPath), { schema: 'test', ok: true });
  assert.deepEqual(fs.readdirSync(reportDirectory), ['report.json']);
} finally {
  fs.rmSync(reportDirectory, { recursive: true, force: true });
}
console.log('ok  atomic report replacement cannot inherit a stale empty artifact');

const conv2dChannelRangeWGSL = fs.readFileSync(
  new URL('../src/shaders/conv2d_channel_range.wgsl', import.meta.url),
  'utf8',
);
assert.match(conv2dChannelRangeWGSL, /applyRelu:\s*u32/);
assert.match(conv2dChannelRangeWGSL, /if\s*\(params\.applyRelu\s*!=\s*0\)/);
assert.match(conv2dChannelRangeWGSL, /sum\s*=\s*max\(sum,\s*0\.0\)/);
console.log('ok  ranged Conv can fuse ReLU without changing the legacy shader');

const plan = createPostProcessorChannelDutyPlan(256);
assert.equal(plan.channelsPerDuty, 256);
assert.equal(plan.duties.length, 51);

for (let plane = 0; plane < 3; plane++) {
  const planeDuties = plan.duties.filter(duty => duty.plane === plane);
  assert.equal(planeDuties[0].kind, 'gather');
  assert.equal(planeDuties.at(-1).kind, 'pixel-shuffle-copy');

  for (const [stageId, totalChannels, expectedRanges] of [
    ['conv-0-relu', 1024, 4],
    ['conv-1-relu', 1024, 4],
    ['conv-2-relu', 1024, 4],
    ['conv-3', 640, 3],
  ]) {
    const ranges = planeDuties.filter(
      duty => duty.kind === 'conv-range' && duty.stageId === stageId,
    );
    assert.equal(ranges.length, expectedRanges);
    assert.equal(ranges[0].channelStart, 0);
    assert.equal(ranges.at(-1).channelEnd, totalChannels);
    assert.equal(
      ranges.reduce((sum, range) => sum + range.channelCount, 0),
      totalChannels,
    );
    for (let rangeIndex = 0; rangeIndex < ranges.length; rangeIndex++) {
      const range = ranges[rangeIndex];
      assert.equal(range.rangeIndex, rangeIndex);
      assert.equal(range.rangeCount, expectedRanges);
      if (rangeIndex > 0) {
        assert.equal(range.channelStart, ranges[rangeIndex - 1].channelEnd);
      }
    }
  }
}
console.log('ok  channel plan covers every postprocessor output channel exactly once');

assert.throws(
  () => createPostProcessorChannelDutyPlan(0),
  /positive safe integer/,
);
assert.throws(
  () => createPostProcessorChannelDutyPlan(1.5),
  /positive safe integer/,
);
console.log('ok  channel profile rejects invalid caller policy');

const manifest = definePostProcessorChannelManifest(256);
const boundary = manifest.phases[0].boundaries[0];
assert.equal(boundary.boundaryId, POST_PROCESSOR_CHANNEL_BOUNDARY_ID);
assert.equal(boundary.totalItems, 51);
assert.equal(boundary.progressWeight, 51);
assert.equal(boundary.metadata.channelsPerDuty, 256);
assert.equal(boundary.chunking.mode, 'fixed');
assert.equal(boundary.chunking.chunkItems, 1);
console.log('ok  manifest preserves the effective channel profile');

function cooperativeFor(totalDuties) {
  let next = 0;
  return {
    startBoundary(boundaryId) {
      assert.equal(boundaryId, POST_PROCESSOR_CHANNEL_BOUNDARY_ID);
      return {
        nextRange() {
          if (next >= totalDuties) return null;
          const itemStart = next++;
          return {
            rangeIndex: itemStart,
            itemStart,
            itemEnd: itemStart + 1,
            itemCount: 1,
          };
        },
        async runGpuDuty(range, duty) {
          duty.encode(); // kit 0.1.41: encode returns the buffer; facade owns submission
        },
      };
    },
  };
}

const events = [];
const result = await drivePostProcessorChannelBoundary(
  cooperativeFor(plan.duties.length),
  {
    plan,
    encodeDuty(duty) {
      events.push(`encode:${duty.dutyIndex}:${duty.kind}`);
      return duty;
    },
    submitDuty(commandBuffer) {
      events.push(`submit:${commandBuffer.dutyIndex}`);
    },
  },
);
assert.equal(result.completedDuties, 51);
assert.equal(result.totalDuties, 51);
assert.equal(result.telemetry.length, 51);
// kit >=0.1.41: encode returns the buffer and the kit submits it, so the fake
// facade observes one event (encode) per duty, not two (encode + submit).
assert.equal(events.length, 51);
console.log('ok  driver encodes the complete declared graph (kit owns submission)');

const malformedDuties = plan.duties.map(duty => ({ ...duty }));
const malformedIndex = malformedDuties.findIndex(
  duty => duty.kind === 'conv-range' && duty.rangeIndex === 1,
);
malformedDuties[malformedIndex].channelStart =
  malformedDuties[malformedIndex - 1].channelStart;
const malformedPlan = {
  ...plan,
  duties: malformedDuties,
};
let malformedEncodeCount = 0;
await assert.rejects(
  drivePostProcessorChannelBoundary(
    cooperativeFor(malformedPlan.duties.length),
    {
      plan: malformedPlan,
      encodeDuty() {
        malformedEncodeCount++;
        return {};
      },
      submitDuty() {},
    },
  ),
  /channel duty plan/,
);
assert.equal(malformedEncodeCount, 0);
console.log('ok  driver rejects duplicated or skipped channel coverage before GPU work');

assert.deepEqual(
  evaluatePostProcessorSmokeAcceptance({
    armSelection: 'channel',
    outputIdentical: null,
    cooperativeComplete: true,
    progressHonest: true,
    cadenceObserved: false,
  }),
  { paired: false, ok: false },
);
assert.deepEqual(
  evaluatePostProcessorSmokeAcceptance({
    armSelection: 'pair',
    outputIdentical: true,
    cooperativeComplete: true,
    progressHonest: true,
    cadenceObserved: false,
  }),
  { paired: true, ok: false },
);
assert.deepEqual(
  evaluatePostProcessorSmokeAcceptance({
    armSelection: 'pair',
    outputIdentical: true,
    cooperativeComplete: true,
    progressHonest: true,
    cadenceObserved: true,
  }),
  { paired: true, ok: true },
);
console.log('ok  one-arm and zero-frame profiling cannot masquerade as paired acceptance');

console.log('\nALL COOPERATIVE POSTPROCESSOR CHANNEL CONTRACT CHECKS PASSED');
