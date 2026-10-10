import assert from 'node:assert/strict';
import {createSf3dProducer} from '../src/lib/sf3d_producer.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {fakeWeightDevice, weightFixture, installWeightFetch} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage = {STORAGE:128, COPY_SRC:4, COPY_DST:8, UNIFORM:64, MAP_READ:1};
globalThis.GPUShaderStage = {COMPUTE:4};
const fixture = weightFixture();
const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
const restoreFetch = installWeightFetch(fixture);
let borrowed;
try {
  await assert.rejects(async () => {
    borrowed = await createSf3dProducer({device:fakeWeightDevice(), weightsUrl:'fixture.bin', workers:{}, memoryBudget:createLoaderMemoryBudget({cpuBytes:1, gpuBytes:1})});
  }, /owned.*device/i, 'an unobserved borrowed renderer/device baseline cannot silently gain a budget claim');
  for (const [cpuBytes,gpuBytes,allocated] of [[1,10000,0],[3*fixture.bytes.length,8,2]]) {
    const device = fakeWeightDevice();
    device.lost = new Promise(() => {});
    const adapter = {limits:device.limits, info:{vendor:'apple', architecture:'metal-3', description:'synthetic'}, async requestDevice(){return device;}};
    Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu:{async requestAdapter(){return adapter;}}}});
    const budget = createLoaderMemoryBudget({cpuBytes,gpuBytes});
    await assert.rejects(createSf3dProducer({memoryBudget:budget, expectedWeightBytes:fixture.bytes.length}), /before allocation/);
    assert.equal(device.buffers.length,allocated);
    assert.equal(device.destroyed,1,'failed budgeted producer retires its owned device, not just its successful weights');
    assert.equal(budget.snapshot().gpu.liveBytes,0);
    assert.equal(budget.snapshot().cpu.liveBytes,0);
  }
} finally {
  if(borrowed) await borrowed.dispose().completion;
  restoreFetch(); if(original)Object.defineProperty(globalThis,'navigator',original);else delete globalThis.navigator;
}
console.log('Actual producer wires the budget before loading; borrowed devices refuse; failed owned initialization retires allocations and device.');
