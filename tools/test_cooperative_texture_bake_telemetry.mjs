#!/usr/bin/env node

import assert from 'node:assert/strict';

import { makeCooperativeTextureBake } from '../src/lib/cooperative_texture_bake.js';

function makeScratch(label, size) {
  return {
    label,
    size,
    destroyed: false,
    destroy() {
      assert.equal(this.destroyed, false, `${label} destroyed more than once`);
      this.destroyed = true;
    },
  };
}

async function runArm(schedulingMode) {
  const submitted = [];
  const scratch = [];
  const device = {
    queue: {
      submit(commandBuffers) {
        submitted.push(...commandBuffers);
      },
      async onSubmittedWorkDone() {},
    },
  };
  const cooperativeBatch = makeCooperativeTextureBake(device, {
    batchTexels: 2,
    schedulingMode,
    invocationId: `test:texture-bake:${schedulingMode}`,
  });

  const report = await cooperativeBatch(4, async (start, end) => {
    if (schedulingMode === 'cooperative' && start > 0) {
      assert.equal(
        scratch[0].destroyed,
        true,
        'the prior range scratch must retire after its exact prefix fence',
      );
    }
    const buffer = makeScratch(`scratch-${start}-${end}`, 64);
    scratch.push(buffer);
    return {
      encode: () => ({ start, end }),
      submit: commandBuffer => device.queue.submit([commandBuffer]),
      hostEncodeMs: 1.25,
      scratchResources: [{ buffer, size: buffer.size, label: buffer.label }],
    };
  });

  assert.equal(submitted.length, 2);
  assert.equal(scratch.every(buffer => buffer.destroyed), true);
  assert.ok(report.textureBakeTelemetry, 'texture-bake telemetry must be present');
  assert.equal(report.textureBakeTelemetry.schema, 'sf3d.texture-bake-duty-telemetry.v1');
  assert.equal(report.textureBakeTelemetry.ranges.length, 2);
  assert.equal(report.textureBakeTelemetry.scratch.allocatedCount, 2);
  assert.equal(report.textureBakeTelemetry.scratch.allocatedBytes, 128);
  assert.equal(report.textureBakeTelemetry.scratch.retiredCount, 2);
  assert.equal(report.textureBakeTelemetry.scratch.retiredBytes, 128);
  assert.equal(report.textureBakeTelemetry.scratch.activeCount, 0);
  assert.equal(report.textureBakeTelemetry.scratch.activeBytes, 0);
  assert.equal(report.textureBakeTelemetry.queueFences.length, schedulingMode === 'cooperative' ? 2 : 1);
  assert.equal(
    report.textureBakeTelemetry.queueFences.every(fence => fence.queueWaitMs >= 0),
    true,
  );
  assert.equal(
    report.textureBakeTelemetry.ranges.every(range => (
      range.hostEncodeMs >= 0
      && range.dutyLifecycleMs >= 0
      && range.dutyWallMs >= 0
      && range.browserYieldMs >= 0
      && range.prepareEncodeInterval?.endMs >= range.prepareEncodeInterval?.startMs
      && range.submitInterval?.endMs >= range.submitInterval?.startMs
      && range.dutyLifecycleInterval?.endMs >= range.dutyLifecycleInterval?.startMs
      && range.browserYieldIntervals.every(interval => interval.endMs >= interval.startMs)
    )),
    true,
  );
  assert.equal(
    report.textureBakeTelemetry.queueFences.every(fence => (
      fence.queueInterval.endMs >= fence.queueInterval.startMs
      && fence.retirementInterval.endMs >= fence.retirementInterval.startMs
    )),
    true,
  );
}

await runArm('cooperative');
await runArm('disabled');

console.log('cooperative texture-bake telemetry contract passed');
