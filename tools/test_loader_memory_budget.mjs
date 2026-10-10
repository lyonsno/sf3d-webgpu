import assert from 'node:assert/strict';
import {weightFixture, fakeWeightDevice, installWeightFetch} from './fixtures/weight_resource_fixture.mjs';
import {loadWeights} from '../src/lib/weights.js';

globalThis.GPUBufferUsage = {STORAGE:128, COPY_SRC:4, COPY_DST:8};
const fixture = weightFixture();
const restoreInvalid = installWeightFetch(fixture);
try {
  await assert.rejects(loadWeights(fakeWeightDevice(), 'fixture.bin', null, {memoryBudget:{}, expectedWeightBytes:fixture.bytes.length}), /authenticated loader budget/,
    'the existing loader must not silently ignore a requested allocation guard');
} finally { restoreInvalid(); }
let api;
try { api = await import('../src/lib/loader_memory_budget.js'); }
catch (error) { if (error.code !== 'ERR_MODULE_NOT_FOUND') throw error; }
assert.equal(typeof api?.createLoaderMemoryBudget, 'function', 'the real loader needs reservations before CPU expansion and GPU allocation');
const budgetFor = (cpuBytes, gpuBytes = 1_000_000) => api.createLoaderMemoryBudget({cpuBytes, gpuBytes});
{
  const fp16 = weightFixture({tensorShapes:new Map([['image_tokenizer.image_mean',[300000]]]), fp16Names:new Set(['image_tokenizer.image_mean'])});
  const headerBytes = new DataView(fp16.bytes.buffer).getUint32(12, true);
  const budget = budgetFor(fp16.bytes.length + headerBytes), device = fakeWeightDevice(), restore = installWeightFetch(fp16);
  budget.bindOwnedDevice(device);
  try {
    await assert.rejects(loadWeights(device, 'fixture.bin', null, {memoryBudget:budget, expectedWeightBytes:fp16.bytes.length}),
      error => error.name === 'SF3DMemoryBudgetError' && /fp16-conversion/.test(error.memoryBudget.label));
    assert.equal(device.buffers.length, 0, 'CPU expansion refusal precedes the GPU allocator');
    assert.equal(budget.snapshot().cpu.liveBytes, 0);
  } finally { restore(); budget.restore(); }
}
{
  const budget = budgetFor(8, 8), device = fakeWeightDevice();
  budget.bindOwnedDevice(device);
  const lease = budget.reserveCpu(8, 'held-source');
  assert.throws(() => budget.reserveCpu(1, 'conversion'), error => error.name === 'SF3DMemoryBudgetError' && error.memoryBudget.label === 'conversion');
  lease.release(); lease.release();
  assert.equal(budget.snapshot().cpu.liveBytes, 0);
  const buffer = device.createBuffer({size:8});
  assert.throws(() => device.createBuffer({size:4}), /before allocation/);
  assert.equal(device.buffers.length, 1, 'refusal must not call the device allocator');
  buffer.destroy(); buffer.destroy();
  assert.equal(budget.snapshot().gpu.liveBytes, 0);
  budget.restore();
}
{
  let fetches = 0;
  const original = globalThis.fetch;
  globalThis.fetch = async () => { fetches++; return new Response(fixture.bytes); };
  try {
    const device = fakeWeightDevice(), budget = budgetFor(fixture.bytes.length - 1);
    budget.bindOwnedDevice(device);
    await assert.rejects(loadWeights(device, 'fixture.bin', null, {memoryBudget:budget, expectedWeightBytes:fixture.bytes.length}), /before allocation/);
    assert.equal(fetches, 0, 'source reservation refusal must precede fetch/body ingestion');
    assert.equal(device.buffers.length, 0);
    assert.equal(budget.snapshot().cpu.liveBytes, 0);
  } finally { globalThis.fetch = original; }
}
{
  const device = fakeWeightDevice(), budget = budgetFor(2 * fixture.bytes.length, 8), restore = installWeightFetch(fixture);
  budget.bindOwnedDevice(device);
  try {
    await assert.rejects(loadWeights(device, 'fixture.bin', null, {memoryBudget:budget, expectedWeightBytes:fixture.bytes.length}), /before allocation/);
    assert.equal(device.buffers.length, 2);
    assert.ok(device.buffers.every(buffer => buffer.destroyed === 1), 'partial loading must retire every prior successful allocation');
    assert.equal(budget.snapshot().gpu.liveBytes, 0);
    assert.equal(budget.snapshot().cpu.liveBytes, 0);
  } finally { restore(); budget.restore(); }
}
{
  const device = fakeWeightDevice(), budget = budgetFor(3 * fixture.bytes.length), restore = installWeightFetch(fixture);
  budget.bindOwnedDevice(device);
  try {
    const weights = await loadWeights(device, 'fixture.bin', null, {memoryBudget:budget, expectedWeightBytes:fixture.bytes.length});
    assert.ok(budget.snapshot().cpu.liveBytes > 0, 'lazy raw copies stay charged to the weight owner');
    assert.equal(budget.snapshot().cpu.physicalMemoryMeasured, false);
    assert.equal(budget.snapshot().gpu.physicalMemoryMeasured, false);
    weights._rawGet('image_estimator.model.visual.ln_pre.weight');
    weights._rawGetCPU('image_estimator.model.visual.ln_pre.weight');
    weights.dispose(); weights.dispose();
    assert.equal(budget.snapshot().cpu.liveBytes, 0);
    assert.equal(budget.snapshot().gpu.liveBytes, 0);
    budget.restore();
  } finally { restore(); }
}
{
  const device = fakeWeightDevice(), budget = budgetFor(3 * fixture.bytes.length), original = globalThis.fetch;
  budget.bindOwnedDevice(device);
  globalThis.fetch = async () => new Response(fixture.bytes.subarray(0, fixture.bytes.length - 1));
  try {
    await assert.rejects(loadWeights(device, 'fixture.bin', null, {memoryBudget:budget, expectedWeightBytes:fixture.bytes.length}), /weight.*(length|bytes)/i);
    assert.equal(device.buffers.length, 0);
    assert.equal(budget.snapshot().cpu.liveBytes, 0);
  } finally { globalThis.fetch = original; budget.restore(); }
  await assert.rejects(loadWeights(device, 'fixture.bin', null, {memoryBudget:budget}), /expectedWeightBytes/);
}
console.log('SF3D source/copy/conversion reservations and device refusal precede allocation; partial and lazy ownership clean up without physical-memory claims.');
for(const lazy of [false,true]){
  const device=fakeWeightDevice(),budget=budgetFor(3*fixture.bytes.length),restore=installWeightFetch(fixture);
  let failMap=!lazy;const create=device.createBuffer;
  device.createBuffer=function(desc){const buffer=create.call(device,desc);if(failMap)buffer.getMappedRange=()=>{throw Error('injected mapped upload failure');};return buffer;};
  budget.bindOwnedDevice(device);let weights;
  try{
    if(lazy)weights=await loadWeights(device,'fixture.bin',null,{memoryBudget:budget,expectedWeightBytes:fixture.bytes.length});
    const before=budget.snapshot().gpu.liveBytes;failMap=true;
    if(lazy)assert.throws(()=>weights._rawGet('image_estimator.model.visual.ln_pre.weight'),/mapped upload failure/);
    else await assert.rejects(loadWeights(device,'fixture.bin',null,{memoryBudget:budget,expectedWeightBytes:fixture.bytes.length}),/mapped upload failure/);
    assert.equal(device.buffers.at(-1).destroyed,1,'successful allocation must retire after failed mapped upload');
    assert.equal(budget.snapshot().gpu.liveBytes,before,'failed eager/lazy upload cannot strand a charge');
  }finally{restore();weights?.dispose();device.destroy();budget.restore();}
}
{
  const budget=budgetFor(8,8);let destroyed=0;
  const buffer={size:4};Object.defineProperty(buffer,'destroy',{value(){destroyed++;},writable:false});
  const device={createBuffer:()=>buffer,destroy(){}};budget.bindOwnedDevice(device);
  assert.throws(()=>device.createBuffer({size:4}), /read only|retirement hook/i);
  assert.equal(destroyed,1,'failed buffer instrumentation must retire the actual allocation');
  assert.equal(budget.snapshot().gpu.liveBytes,0,'failed instrumentation must not strand an unreachable charge');budget.restore();
}
{
  const device=fakeWeightDevice(),budget=budgetFor(2*fixture.bytes.length,8),restore=installWeightFetch(fixture);
  const create=device.createBuffer;device.createBuffer=function(desc){const buffer=create(desc);if(device.buffers.length===1)buffer.destroy=function(){throw Error('injected retirement failure');};return buffer;};
  budget.bindOwnedDevice(device);
  try{
    await assert.rejects(loadWeights(device,'fixture.bin',null,{memoryBudget:budget,expectedWeightBytes:fixture.bytes.length}),/cleanup|retirement/i);
    assert.equal(device.buffers[1].destroyed,1,'one failed retirement cannot skip another owned buffer');
    assert.equal(budget.snapshot().cpu.liveBytes,0,'GPU retirement failure cannot skip CPU release');
    assert.equal(budget.snapshot().gpu.liveBytes,4,'failed destruction remains charged, not falsely reclaimed');
  }finally{restore();/* failed fake destruction remains charged; no false restore */}
}
