import assert from 'node:assert/strict';
import {captureGpuBufferAllocations} from '../src/lib/gpu.js';
import {dispatchConv2dChannelRange} from '../src/lib/shader_ops.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={UNIFORM:64,STORAGE:128,COPY_SRC:4,COPY_DST:8};
const device=fakeWeightDevice(),borrowed=Array.from({length:4},()=>device.createBuffer({size:4}));
device.createBindGroup=desc=>desc;
const resourceContext={device,pipelines:new Map()};
const pass={setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}};
const encoder={beginComputePass(){return pass}};
const owned=[];
const params={inC:1,inH:1,inW:1,outC:1,kH:1,kW:1,padH:0,padW:0,strideH:1,strideW:1,resourceContext};
function encode(){return dispatchConv2dChannelRange(device,encoder,...borrowed,params,{channelStart:0,channelCount:1})}
captureGpuBufferAllocations(encode,{ownedAllocations:owned});
assert.equal(owned.length,1,'resident shader uniform must enter caller capture inventory');
assert.equal(owned[0].size,64);
owned[0].buffer.destroy();
captureGpuBufferAllocations(encode,{ownedAllocations:owned});
assert.equal(owned.length,2,'a retired uniform cannot be resurrected from a global cache');
assert.notEqual(owned[0].buffer,owned[1].buffer);
assert.ok(borrowed.every(b=>!b.destroyed));
const other=fakeWeightDevice();
assert.throws(()=>dispatchConv2dChannelRange(other,encoder,...borrowed,params,{channelStart:0,channelCount:1}),/resource context.*device/);
const failing=fakeWeightDevice();
const original=failing.createBuffer;
failing.createBuffer=desc=>{const b=original(desc);b.getMappedRange=()=>{throw Error('mapping refused')};return b};
const failedOwned=[];
assert.throws(()=>captureGpuBufferAllocations(()=>dispatchConv2dChannelRange(failing,encoder,...borrowed,
  {...params,resourceContext:{device:failing,pipelines:new Map()}},{channelStart:0,channelCount:1}),
  {ownedAllocations:failedOwned}),/mapping refused/);
assert.equal(failedOwned.length,1,'allocation remains in exact inventory when initialization rejects');
assert.equal(failedOwned[0].buffer.destroyed,0,'caller drain owns failure cleanup');
console.log('PASS selected postprocessor shader uniform custody and nonresurrection');

