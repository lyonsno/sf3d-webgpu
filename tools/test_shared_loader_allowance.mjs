import assert from 'node:assert/strict';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
const make=(cpuBytes=150,gpuBytes=150,parentBudget)=>createLoaderMemoryBudget({cpuBytes,gpuBytes,parentBudget});
{
  const parent=make(),a=make(150,150,parent),b=make(150,150,parent);
  const held=a.reserveCpu(100,'runtime-a-input');
  assert.equal(parent.snapshot().cpu.liveBytes,100,'child reservation must charge the one shared allowance, not leave it unused');
  assert.throws(()=>b.reserveCpu(60,'runtime-b-input'),/before allocation/);
  assert.equal(b.snapshot().cpu.liveBytes,0,'parent refusal cannot leave a child charge');
  assert.throws(()=>parent.restore(),/retire/,'live descendant allocations prevent restoring the parent');
  held.release();held.release();assert.equal(parent.snapshot().cpu.liveBytes,0);
  const next=b.reserveCpu(150,'runtime-b-after-retirement');next.release();a.restore();b.restore();parent.restore();
}
{
  const parent=make(),a=make(150,150,parent),b=make(150,150,parent),da=fakeWeightDevice(),db=fakeWeightDevice();
  a.bindOwnedDevice(da);b.bindOwnedDevice(db);
  const first=da.createBuffer({size:100,label:'runtime-a'});
  assert.equal(parent.snapshot().gpu.liveBytes,100);
  assert.throws(()=>db.createBuffer({size:60,label:'runtime-b'}),/before allocation/);
  assert.equal(db.buffers.length,0,'shared refusal precedes the second actual allocator');
  assert.equal(b.snapshot().gpu.liveBytes,0);
  first.destroy();first.destroy();const second=db.createBuffer({size:150});second.destroy();
  assert.equal(parent.snapshot().gpu.liveBytes,0);assert.equal(parent.snapshot().gpu.peakLiveBytes,150);
  a.restore();b.restore();parent.restore();
}
{
  const parent=make(400,400),middle=make(250,250,parent),a=make(200,200,middle),b=make(200,200,middle);
  const held=a.reserveCpu(200,'leaf-a');assert.equal(parent.snapshot().cpu.liveBytes,200);
  assert.throws(()=>b.reserveCpu(100,'leaf-b'),/before allocation/,'each ancestor quota, not only root quota, binds');
  assert.equal(parent.snapshot().cpu.liveBytes,200);assert.equal(middle.snapshot().cpu.liveBytes,200);assert.equal(b.snapshot().cpu.liveBytes,0);
  held.release();a.restore();b.restore();middle.restore();parent.restore();
}
{
  const parent=make(100,100),child=make(8,8,parent);let called=0;
  const device={createBuffer(){called++;assert.equal(parent.snapshot().gpu.liveBytes,8,'ancestor charged before allocation');assert.equal(child.snapshot().gpu.liveBytes,8,'child charged before allocation');throw Error('injected allocator failure');},destroy(){}};
  child.bindOwnedDevice(device);
  assert.throws(()=>child.reserveCpu(9,'local-quota'),/before allocation/);assert.equal(parent.snapshot().cpu.liveBytes,0);
  assert.throws(()=>device.createBuffer({size:8}),/injected allocator failure/);
  assert.equal(parent.snapshot().gpu.liveBytes,0);assert.equal(child.snapshot().gpu.liveBytes,0);
  child.restore();parent.restore();assert.equal(called,1);
}
{
  const parent=make(8,8),child=make(8,8,parent),device=fakeWeightDevice(),create=device.createBuffer;
  let failDevice=true;device.destroy=()=>{if(failDevice)throw Error('injected device destruction failure');};
  device.createBuffer=function(desc){const buffer=create.call(this,desc);buffer.destroy=()=>{throw Error('injected retirement failure');};return buffer;};
  child.bindOwnedDevice(device);const held=device.createBuffer({size:8});assert.throws(()=>held.destroy(),/retirement failure/);
  assert.equal(parent.snapshot().gpu.liveBytes,8,'failed destroy cannot return shared capacity');
  assert.throws(()=>parent.restore(),/retire/);assert.throws(()=>device.destroy(),/device destruction failure/);
  assert.equal(parent.snapshot().gpu.liveBytes,8,'failed device destruction also retains the shared charge');
  failDevice=false;device.destroy();assert.equal(parent.snapshot().gpu.liveBytes,0);
  child.restore();parent.restore();
}
{
  assert.throws(()=>make(1,1,{}),/authenticated parent/);
  const parent=make(8,8),child=make(8,8,parent);parent.restore();
  assert.throws(()=>child.reserveCpu(1,'after-parent-restore'),/restored/);
  assert.equal(child.snapshot().cpu.liveBytes,0);assert.throws(()=>make(8,8,parent),/restored/);child.restore();
}
console.log('Shared authenticated CPU/GPU allowance binds before allocation across runtimes and ancestors; failed allocation/retirement preserve honest ownership.');
