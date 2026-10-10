import assert from 'node:assert/strict';
import * as resident from '../src/lib/resident_dino.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
assert.equal(typeof resident.selectDinoPhase,'function','actual resident DINO selection is required');
const template={blocks:Array.from({length:24},(_,i)=>({id:i})),patchEmbed:{weight:{}},clsToken:{},posEmbed:{},layernorm:{weight:{}}};
for(let i=0;i<24;i++){
  const selected=resident.selectDinoPhase(template,{blockStart:i,blockEnd:i+1,isFirst:i===0,isLast:i===23});
  assert.equal(selected.blocks.length,24);
  for(let j=0;j<24;j++)assert.equal(selected.blocks[j],j===i?template.blocks[i]:null);
  assert.equal('patchEmbed' in selected,i===0);assert.equal('layernorm' in selected,i===23);
}
assert.throws(()=>resident.selectDinoPhase(template,{blockStart:0,blockEnd:25,isFirst:true,isLast:true}),/DINO phase/);
assert.throws(()=>resident.selectDinoPhase(template,{blockStart:0,blockEnd:1,isFirst:false,isLast:false}),/DINO phase/);
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
const fixture=weightFixture(),header=new DataView(fixture.bytes.buffer).getUint32(12,true),etag='"consumer-refusal"',prior=globalThis.fetch;
let calls=0;globalThis.fetch=async(url,init)=>{
  const [,start,end]=init.headers.Range.match(/^bytes=(\d+)-(\d+)$/);calls++;
  return new Response(fixture.bytes.slice(Number(start),Number(end)+1),{status:206,headers:{ETag:etag,
    'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`}});
};
const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:2*header+4096,gpuBytes:4096});
await budget.requestOwnedDevice({requestDevice:async()=>device});
try{
  await assert.rejects(()=>resident.runResidentDino({device,memoryBudget:budget,weightsUrl:'fixture.bin',expectedWeightBytes:fixture.bytes.length,
    expectedSourceETag:etag,imageChw:new Float32Array(3*512*512),onBeforePhase:async()=>{throw Error('fresh host refuses');},withResult:async()=>{}}),/fresh host refuses/);
  assert.equal(calls,2,'fresh refusal allows schema metadata only, no weight payload');
  assert.equal(device.buffers.length,0,'native phase refusal occurs before model allocation');
}finally{globalThis.fetch=prior;device.destroy();budget.restore();}
console.log('resident DINO selection preserves all24 indices and exact setup/final weights');
