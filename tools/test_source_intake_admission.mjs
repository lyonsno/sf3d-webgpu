import assert from 'node:assert/strict';
import {createFlatTensorRangeSource} from '../src/lib/flat_tensor_ranges.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {weightFixture} from './fixtures/weight_resource_fixture.mjs';
const fixture=weightFixture(),header=new DataView(fixture.bytes.buffer).getUint32(12,true),etag='"synthetic-intake"';
const original=globalThis.fetch,reads=[],duties=[];
globalThis.fetch=async(url,init)=>{
  const [,a,b]=init.headers.Range.match(/^bytes=(\d+)-(\d+)$/),start=Number(a),end=Number(b);
  reads.push({start,end});return new Response(fixture.bytes.slice(start,end+1),{status:206,headers:{
    ETag:etag,'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`,'Content-Length':String(end-start+1)}});
};
const budget=createLoaderMemoryBudget({cpuBytes:2*header+64,gpuBytes:1});
const options={memoryBudget:budget,expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag};
try{
  await createFlatTensorRangeSource('synthetic.bin',{...options,onBeforeSourceIntake:async d=>{
    duties.push({...d,reads:reads.length,live:budget.snapshot().cpu.liveBytes});
  }});
  assert.deepEqual(duties.map(d=>[d.name,d.rangeCpuBytes,d.reads,d.live]),
    [['weight-header-prefix',32,0,0],['weight-header',2*header,1,0]],
    'each source intake has a fresh new-backing gate before destination/response fetch allocation');
  const before=reads.length,peak=budget.snapshot().cpu.peakLiveBytes,eventCount=budget.events.length;
  assert.ok(Number.isSafeInteger(peak),'refusal witness must read an actual authoritative peak counter');
  await assert.rejects(createFlatTensorRangeSource('synthetic.bin',{...options,
    onBeforeSourceIntake:async()=>{throw 0;}}),e=>e===0);
  assert.equal(reads.length,before);assert.equal(budget.snapshot().cpu.peakLiveBytes,peak);
  assert.deepEqual(budget.events.slice(eventCount),[],
    'refused source range cannot reserve then release backing beneath an already higher historical peak');
  await assert.rejects(createFlatTensorRangeSource('synthetic.bin',{...options,
    onBeforeSourceIntake:async d=>{if(d.name==='weight-header')throw Error('header refused');}}),/header refused/);
  assert.equal(reads.length,before+1);assert.equal(budget.snapshot().cpu.liveBytes,0);
}finally{globalThis.fetch=original;budget.restore();}
console.log('PASS prefix/full-header fresh exact new-backing admission before fetch, falsy refusal and no admission cleanup credit');
