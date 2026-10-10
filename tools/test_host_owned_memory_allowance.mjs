import assert from 'node:assert/strict';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';

// Synthetic devices test the local ownership protocol, not WebGPU capacity,
// platform acquisition, or physical memory reclamation.
const createBudget = () => createLoaderMemoryBudget({cpuBytes:128, gpuBytes:128, totalBytes:128});
{
  const budget=createBudget(), device=fakeWeightDevice();
  let requests=0;
  const adapter={async requestDevice(descriptor){requests++; assert.equal(descriptor.label,'host'); return device;}};
  assert.equal(typeof budget.requestOwnedDevice,'function','host acquisition must install the allowance before returning the device');
  assert.equal(await budget.requestOwnedDevice(adapter,{label:'host'}),device);
  budget.assertDeviceAcquiredHere(device);
  assert.equal(requests,1);
  assert.equal(budget.snapshot().deviceAcquisition,'requested-through-budget');
  const host=device.createBuffer({size:80,label:'host-renderer'});
  assert.throws(()=>device.createBuffer({size:49,label:'model'}),/before allocation/);
  assert.equal(device.buffers.length,1,'combined host/model demand refuses before allocator');
  assert.equal(budget.snapshot().gpu.liveBytes,80);
  await assert.rejects(budget.requestOwnedDevice(adapter),/already|acquisition/);
  assert.equal(requests,1,'an installed allowance cannot acquire an extra device');
  host.destroy(); device.destroy(); budget.restore();
  assert.throws(()=>budget.assertDeviceAcquiredHere(device),/restored|installed/);
}
{
  const budget=createBudget(), device=fakeWeightDevice();
  device.createBuffer({size:8,label:'earlier-unaccounted-buffer'});
  budget.bindOwnedDevice(device);
  budget.assertDevice(device);
  assert.throws(()=>budget.assertDeviceAcquiredHere(device),/acquisition|requested/,
    'binding after acquisition cannot attest the earlier host baseline');
  assert.equal(budget.snapshot().deviceAcquisition,'bound-after-acquisition');
  device.destroy(); budget.restore();
}
{
  const budget=createBudget(), device=fakeWeightDevice();
  let deliver, requests=0;
  const adapter={requestDevice(){requests++; return new Promise(resolve=>{deliver=resolve;});}};
  const pending=budget.requestOwnedDevice(adapter);
  await assert.rejects(budget.requestOwnedDevice(adapter),/acquisition|already/);
  assert.equal(requests,1,'overlapping acquisitions refuse before contacting adapter');
  budget.restore(); deliver(device);
  await assert.rejects(pending,/restored/);
  assert.equal(device.destroyed,1,'a device returned after invalidation is retired, not exposed');
}
{
  const budget=createBudget(), device=fakeWeightDevice();
  Object.defineProperty(device,'createBuffer',{value:device.createBuffer,writable:false});
  const adapter={async requestDevice(){return device;}};
  await assert.rejects(budget.requestOwnedDevice(adapter),/read only|installation/);
  assert.equal(device.destroyed,1,'failed instrumentation retires the newly acquired device');
}
{
  const budget=createBudget(), device=fakeWeightDevice();
  Object.defineProperty(device,'createBuffer',{value:device.createBuffer,writable:false});
  device.destroy=()=>{throw Error('device retirement denied');};
  let requests=0;
  const adapter={async requestDevice(){requests++; return device;}};
  await assert.rejects(budget.requestOwnedDevice(adapter),error=>{
    assert.ok(error instanceof AggregateError);
    assert.equal(error.errors.at(-1).message,'device retirement denied');
    return true;
  });
  await assert.rejects(budget.requestOwnedDevice(adapter),/cleanup.*unresolved/);
  assert.throws(()=>budget.reserveCpu(1,'retry'),/cleanup.*unresolved/);
  assert.equal(requests,1,'unresolved acquisition cleanup cannot multiply allocations');
}
console.log('Host allowance guards acquisition, refuses unaccounted baseline and overlapping requests, and preserves unresolved cleanup.');
