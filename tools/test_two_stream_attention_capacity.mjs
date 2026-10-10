import assert from 'node:assert/strict';
import {TwoStreamBackbone,createTwoStreamAttentionDutyPlan} from '../src/lib/two_stream.js';
import {defineTwoStreamManifest} from '../src/lib/cooperative_two_stream.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64};
const plan=createTwoStreamAttentionDutyPlan(1297,{residentFFN:true,linearRowsPerDuty:128,attentionRowsPerDuty:32});
assert.equal(plan.length,7958,'smaller complete-query tiles must declare all actual work, not ignore caller configuration');
assert.equal(defineTwoStreamManifest({dutyGranularity:'attention-tile',N_img:1297,residentFFN:true,attentionRowsPerDuty:32}).phases[0].boundaries[0].totalItems,7958);
for(const rows of [1,16,32,128,256]){
  const device=fakeWeightDevice(),backbone=new TwoStreamBackbone(device);
  backbone._dispatchLinearNoBias=()=>{};
  const s=backbone._createAttentionState({}, {}, {}, {},65,2,1024,rows);
  assert.equal(s.tileQCapacity,rows);assert.equal(s.scoreBuf.size,16*rows*2*4);
  assert.equal(s.tileCount,Math.ceil(65/rows));
  const copies=[],bindings=[];
  device.createBindGroup=({entries})=>{bindings.push(entries);return entries;};
  const enc={beginComputePass(){return{setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}};},
    copyBufferToBuffer(src,from,dst,to,size){copies.push({from,to,size});}};
  for(let i=0;i<s.tileCount;i++)backbone._dispatchAttentionTile(enc,s,i);
  assert.equal(s.nextTileIndex,s.tileCount);
  assert.equal(copies.reduce((n,c)=>n+c.size,0),65*1024*4);
  for(let i=0,offset=0;i<copies.length;i++){assert.equal(copies[i].to,offset);offset+=copies[i].size;}
  assert.equal(copies.at(-1).to+copies.at(-1).size,65*1024*4,'final partial query is retained at its original output position');
  for(const entries of bindings)for(const e of entries){const r=e.resource;if(r.buffer&&r.size!=null)
    assert.ok((r.offset??0)+r.size<=r.buffer.size,'query/key/value/scores stay within exact actual backing');}
}
for(const rows of [0,-1,1.5,NaN])assert.throws(()=>createTwoStreamAttentionDutyPlan(1297,{attentionRowsPerDuty:rows}),/attentionRowsPerDuty/);
assert.equal(createTwoStreamAttentionDutyPlan(1297).length,4218,'legacy default remains unchanged');
console.log('measured smaller query scratch retains complete keys, every query and partial output tail');

