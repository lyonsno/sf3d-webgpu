import assert from 'node:assert/strict';
import {createSf3dProducer} from '../src/lib/sf3d_producer.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {fakeWeightDevice, weightFixture, installWeightFetch} from './fixtures/weight_resource_fixture.mjs';

// Real producer/loader lifecycle, synthetic tiny weights and device. This does
// not execute the model or establish platform capacity/physical reclamation.
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64,MAP_READ:1};
globalThis.GPUShaderStage={COMPUTE:4};
const fixture=weightFixture(), restoreFetch=installWeightFetch(fixture);
const installedFetch=globalThis.fetch;
let sourceFetches=0;
globalThis.fetch=async (...args)=>{if(String(args[0])==='fixture.bin')sourceFetches++;return installedFetch(...args);};
async function hostWithBudget(cpuBytes=3*fixture.bytes.length,gpuBytes=10000){
  const budget=createLoaderMemoryBudget({cpuBytes,gpuBytes});
  const device=fakeWeightDevice(), adapter={limits:device.limits,info:{vendor:'apple',description:'synthetic'},async requestDevice(){return device;}};
  await budget.requestOwnedDevice(adapter);
  const hostBuffer=device.createBuffer({size:80,label:'host-renderer'});
  return{device,adapter,budget,hostBuffer};
}
function assertHostPreserved({device,budget,hostBuffer}){
  assert.equal(device.destroyed,0,'producer must not destroy its host device');
  assert.equal(hostBuffer.destroyed,0,'host renderer allocation survives producer failure/disposal');
  budget.assertDeviceAcquiredHere(device);
  assert.equal(budget.snapshot().gpu.liveBytes,80,'only the renderer baseline remains charged');
  const count=device.buffers.length;
  assert.throws(()=>device.createBuffer({size:10000,label:'too-large-after-producer'}),/before allocation/);
  assert.equal(device.buffers.length,count,'host guard remains installed after producer ends');
  const nextFrame=device.createBuffer({size:4,label:'next-host-frame'});nextFrame.destroy();
}
function retireHost({device,budget,hostBuffer}){hostBuffer.destroy();device.destroy();budget.restore();}
try{
  {
    const host=await hostWithBudget(1), before=sourceFetches;
    await assert.rejects(createSf3dProducer({device:host.device,adapter:host.adapter,memoryBudget:host.budget,workers:{},weightsUrl:'fixture.bin',expectedWeightBytes:fixture.bytes.length}),
      error=>error.name==='SF3DMemoryBudgetError'&&error.memoryBudget.label==='weight-source',
      'the actual constructor must consume an authenticated host allowance and refuse source allocation');
    assert.equal(sourceFetches,before,'source refusal precedes the weight request');
    assertHostPreserved(host);retireHost(host);
  }
  {
    const host=await hostWithBudget(), producer=await createSf3dProducer({device:host.device,adapter:host.adapter,memoryBudget:host.budget,workers:{},weightsUrl:'fixture.bin',expectedWeightBytes:fixture.bytes.length});
    assert.equal(producer.deviceInjected,true);assert.equal(producer.device,host.device);
    assert.ok(host.budget.snapshot().cpu.liveBytes>0,'producer-owned retained source copies are charged');
    const producerBuffers=host.device.buffers.filter(buffer=>buffer!==host.hostBuffer);
    const disposal=producer.dispose();assert.equal(producer.dispose().completion,disposal.completion);
    assert.deepEqual(await disposal.completion,{status:'released'});
    assert.ok(producerBuffers.length>0&&producerBuffers.every(buffer=>buffer.destroyed===1));
    assert.equal(host.budget.snapshot().cpu.liveBytes,0);
    assertHostPreserved(host);retireHost(host);
  }
  {
    const host=await hostWithBudget();
    host.device.createBindGroupLayout=()=>{throw Error('injected host producer initialization failure');};
    await assert.rejects(createSf3dProducer({device:host.device,adapter:host.adapter,memoryBudget:host.budget,workers:{},weightsUrl:'fixture.bin',expectedWeightBytes:fixture.bytes.length}),/injected host producer initialization failure/);
    assert.ok(host.device.buffers.filter(buffer=>buffer!==host.hostBuffer).every(buffer=>buffer.destroyed===1));
    assert.equal(host.budget.snapshot().cpu.liveBytes,0);
    assertHostPreserved(host);retireHost(host);
  }
  {
    const host=await hostWithBudget(), before=sourceFetches, count=host.device.buffers.length;
    await assert.rejects(createSf3dProducer({device:host.device,memoryBudget:host.budget,weights:{},workers:{}}),/injected.*weights|weights.*unaccounted/);
    assert.equal(sourceFetches,before);assert.equal(host.device.buffers.length,count);
    assertHostPreserved(host);retireHost(host);
  }
  {
    const budget=createLoaderMemoryBudget({cpuBytes:1000,gpuBytes:1000}),device=fakeWeightDevice();
    budget.bindOwnedDevice(device);
    const before=sourceFetches;
    await assert.rejects(createSf3dProducer({device,memoryBudget:budget,workers:{}}),/requested.*budget|acquisition.*unaccounted/);
    assert.equal(sourceFetches,before);assert.equal(device.destroyed,0);assert.equal(device.buffers.length,0);
    device.destroy();budget.restore();
  }
}finally{restoreFetch();}
console.log('Actual producer consumes a guarded-from-acquisition host allowance; source refusal, initialization failure and disposal preserve the host device, allocations and guard.');
