import assert from 'node:assert/strict';
import * as clip from '../src/lib/clip_estimator.js';
import {captureGpuBufferAllocations} from '../src/lib/gpu.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
assert.equal(typeof clip.encodeClipVisualPhase,'function','same CLIP transformer needs canonical complete computation phases');
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:16};
const device=fakeWeightDevice(),owner={allocations:[]};
device.createBindGroup=d=>d;
device.queue.writeBuffer=()=>{throw Error('uniform write failed before dispatch');};
const pipelines={layernorm:{getBindGroupLayout(){return {};}},linear:{getBindGroupLayout(){return {};}},residentOwner:owner};
const pair={weight:{size:3072},bias:{size:3072}},input={size:153600};
assert.throws(()=>captureGpuBufferAllocations(()=>clip.encodeClipVisualPhase({},device,input,pair,pipelines,'pre'),
  {ownedAllocations:owner.allocations}),/uniform write failed/);
assert.equal(owner.allocations.length,1,'actual uniform enters reachable inventory before failed write');
assert.equal(owner.allocations[0].buffer,device.buffers[0]);assert.equal(owner.allocations[0].size,16);
for(const allocation of owner.allocations)allocation.buffer.destroy();
device.queue.writeBuffer=(buffer,offset,data)=>new Uint8Array(buffer.data,offset,data.byteLength)
  .set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));
const params=[],encoder={beginComputePass(){return {setPipeline(){},setBindGroup(i,bg){
  const b=bg.entries[0].resource.buffer;if(b.data&&b.size<=32)params.push(new Uint32Array(b.data).slice());},
  dispatchWorkgroups(){},end(){}};}};
const block={ln1:pair,ln2:pair,qkv:pair,outProj:pair,fc:pair,proj:pair};
const owned=[],start=device.buffers.length;
const out=captureGpuBufferAllocations(()=>clip.encodeClipVisualPhase(encoder,device,input,block,
  {...pipelines,residentOwner:{allocations:owned}},'block'),{ownedAllocations:owned}).value;
assert.equal(out.size,153600);assert.equal(device.buffers.slice(start).reduce((n,b)=>n+b.size,0),2764960);
assert.equal(owned.length,device.buffers.length-start,'all actual block work/uniform allocations have owner');
const linear=params.filter(p=>p.length===8);
assert.deepEqual(linear.map(p=>[p[1],p[2],p[4]]),[[768,2304,0],[768,768,1],[768,3072,1],[3072,768,1]],
  'existing canonical QKV native orientation and transposed projections, no CPU substitution');
assert.throws(()=>clip.encodeClipVisualPhase(encoder,device,input,block,pipelines,'unknown'),/canonical visual phase/);
assert.equal(typeof clip.projectClipFeatures,'function');assert.equal(typeof clip.runClipMaterialHead,'function');
console.log('PASS actual CLIP phase graph, canonical weight orientations and uniform failure ownership; synthetic policy only');
