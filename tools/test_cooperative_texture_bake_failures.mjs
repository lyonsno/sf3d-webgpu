#!/usr/bin/env node

import assert from 'node:assert/strict';

import { makeCooperativeTextureBake } from '../src/lib/cooperative_texture_bake.js';

function makeScratch(label) {
  return {
    label,
    size: 64,
    destroyed: false,
    destroy() {
      assert.equal(this.destroyed, false, `${label} destroyed more than once`);
      this.destroyed = true;
    },
  };
}

async function expectCleanup({
  label,
  schedulingMode = 'cooperative',
  failEncode = false,
  failSubmit = false,
  failFirstFence = false,
  failEveryFence = false,
  abortAfterSubmit = false,
  failBrowserYield = false,
  failProgress = false,
}) {
  const controller = new AbortController();
  const scratch = makeScratch(label);
  let fenceCalls = 0;
  const device = {
    queue: {
      submit() {
        if (failSubmit) throw new Error(`${label}: submit failed`);
        if (abortAfterSubmit) controller.abort(`${label}: cancelled`);
      },
      async onSubmittedWorkDone() {
        fenceCalls += 1;
        if (failEveryFence) {
          throw new Error(`${label}: device unavailable`);
        }
        if (failFirstFence && fenceCalls === 1) {
          throw new Error(`${label}: prefix fence failed`);
        }
      },
    },
  };
  const cooperativeBatch = makeCooperativeTextureBake(device, {
    batchTexels: 1,
    schedulingMode,
    signal: controller.signal,
    invocationId: `test:texture-bake:failure:${label}`,
    onProgress(progress) {
      if (failProgress && progress.completedItems > 0) {
        throw new Error(`${label}: progress callback failed`);
      }
    },
    onBrowserYield() {
      if (failBrowserYield) throw new Error(`${label}: browser-yield callback failed`);
    },
  });

  await assert.rejects(
    cooperativeBatch(2, async () => ({
      encode() {
        if (failEncode) throw new Error(`${label}: encode failed`);
        return { label };
      },
      submit(commandBuffer) {
        device.queue.submit([commandBuffer]);
      },
      scratchResources: [{ buffer: scratch, size: scratch.size, label }],
    })),
  );

  assert.equal(scratch.destroyed, true, `${label}: scratch must retire after failure`);
  if (failFirstFence || failEveryFence) {
    assert.ok(fenceCalls >= 2, `${label}: submitted scratch requires a recovery fence`);
  }
  if (abortAfterSubmit) assert.ok(fenceCalls >= 1, `${label}: submitted scratch requires a fence`);
}

await expectCleanup({ label: 'encode', failEncode: true });
await expectCleanup({ label: 'submit', failSubmit: true });
await expectCleanup({ label: 'fence', failFirstFence: true });
await expectCleanup({ label: 'device-loss', failEveryFence: true });
await expectCleanup({ label: 'browser-yield', failBrowserYield: true });
await expectCleanup({ label: 'progress', failProgress: true });
await expectCleanup({
  label: 'disabled-abort',
  schedulingMode: 'disabled',
  abortAfterSubmit: true,
});

console.log('cooperative texture-bake failure cleanup contract passed');
