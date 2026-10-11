import assert from 'node:assert/strict';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {runResidentDecoder} from '../src/lib/resident_decoder.js';
import {runResidentDino} from '../src/lib/resident_dino.js';
import {runResidentTwoStream} from '../src/lib/resident_two_stream.js';
import {runResidentPostProcessor} from '../src/lib/resident_post_processor.js';
const originalFetch=globalThis.fetch,fixture=weightFixture({fullDecoder:true}),reads=[],etag='"synthetic-intake"';
globalThis.fetch=async(url,init)=>{reads.push(url);const [,a,b]=init.headers.Range.match(/^bytes=(\d+)-(\d+)$/);
  const start=Number(a),end=Number(b);return new Response(fixture.bytes.slice(start,end+1),{status:206,
    headers:{ETag:etag,'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`,'Content-Length':String(end-start+1)}});};
const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:1024*1024,gpuBytes:1024*1024});
await budget.requestOwnedDevice({async requestDevice(){return device;}});
try{
  const cases=[
    [runResidentDecoder,{decoder:{device,_uniformCache:new Map()},heads:['density','vertex_offset']}],
    [runResidentDino,{imageChw:new Float32Array(3*512*512)}],
    [runResidentTwoStream,{backbone:{device,_uniformCache:new Map()},N_img:1297,imageTokensBuf:{size:1297*1024*4}}],
    [runResidentPostProcessor,{postProcessor:{device},triplanesBuf:{size:113246208}}]
  ];
  for(const [run,extra]of cases)await assert.rejects(()=>run({device,memoryBudget:budget,
    weightsUrl:'synthetic.bin',expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag,
    onBeforeSourceIntake:async()=>{throw Error('source refused before allocator');},
    onBeforePhase:async()=>{throw Error('late phase fallback');},onBeforeDuty:async()=>{},withResult:async()=>{},...extra}),/source refused before allocator/);
  assert.equal(reads.length,0);assert.equal(budget.snapshot().cpu.liveBytes,0);
}finally{budget.restore();globalThis.fetch=originalFetch;}
console.log('PASS optional actual DINO/two-stream/postprocessor/decoder source-intake callback precedes every range and failure cleanup; local fixture only');
