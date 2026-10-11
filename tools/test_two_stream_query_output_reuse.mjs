import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {TwoStreamBackbone} from '../src/lib/two_stream.js';
import {defineTwoStreamManifest,retireTwoStreamWork,runCooperativeTwoStream} from '../src/lib/cooperative_two_stream.js';
import {createLoaderMemoryBudget,isLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {captureGpuBufferAllocations,createEmptyBuffer} from '../src/lib/gpu.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64};

// Actual dispatch/allocation methods; synthetic GPU executor. This proves
// complete ordered row lifetime and local policy, not native shader conformance.
function fixture(reuse) {
  const device=fakeWeightDevice(),backbone=new TwoStreamBackbone(device);
  const N=65,D=1024,rows=32,trace=[];
  backbone._dispatchLinearNoBias=()=>{};
  backbone._cachedUniform=()=>({});
  backbone.pipelines={crossAttnScores:'scores',crossAttnSoftmax:'softmax',crossAttnApply:'apply'};
  device.createBindGroup=({entries})=>entries;
  const state=backbone._createAttentionState({}, {}, {}, {},N,2,D,rows,reuse);
  const original=new Float32Array(N*D);
  for(let i=0;i<original.length;i++)original[i]=Math.fround((i%71-35)/13);
  new Float32Array(state.qBuf.data).set(original);
  const encoder={
    beginComputePass() {
      let pipeline,bindings;
      return {setPipeline(p){pipeline=p;},setBindGroup(_,b){bindings=b;},dispatchWorkgroups(){},
        end(){
          const q=bindings[1].resource,out=bindings[5].resource;
          assert.notEqual(q.buffer,out.buffer,'compute read/write bindings never alias');
          const offset=q.offset??0,count=q.size/4,start=offset/4;
          if(pipeline==='scores')assert.deepEqual(
            new Float32Array(q.buffer.data,offset,count),original.subarray(start,start+count),
            'each next tile must read its original complete query rows');
          if(pipeline==='apply'){
            const dst=new Float32Array(out.buffer.data,0,count);
            for(let i=0;i<count;i++)dst[i]=Math.fround(original[start+i]*2+1);
          }
          trace.push({kind:pipeline,offset,bytes:q.size});
        }};
    },
    copyBufferToBuffer(src,from,dst,to,size){
      assert.notEqual(src,dst,'tile staging and full result backing remain distinct');
      trace.push({kind:'copy',offset:to,bytes:size});
      new Uint8Array(dst.data,to,size).set(new Uint8Array(src.data,from,size));
    },
  };
  return {device,backbone,state,encoder,trace,original,N,D};
}
const f=fixture(true),old=fixture(false);
assert.ok(f.state.attnOutBuf===f.state.qBuf,
  'completed query rows must become the exact complete attention output without a second full allocation');
assert.notEqual(old.state.attnOutBuf,old.state.qBuf,'legacy default keeps distinct query/output stores');
assert.equal(old.device.buffers.length-f.device.buffers.length,1);
assert.equal(f.state.reuseAttentionQueryStorage,true);
for(const x of [f,old]){
  assert.throws(()=>x.backbone._finishAttention(x.encoder,x.state),/incomplete/);
  for(let i=0;i<x.state.tileCount;i++)x.backbone._dispatchAttentionTile(x.encoder,x.state,i);
  assert.equal(x.state.nextTileIndex,3);
  assert.deepEqual(x.trace.map(t=>t.kind),['scores','softmax','apply','copy','scores','softmax','apply','copy','scores','softmax','apply','copy']);
  assert.deepEqual(x.trace.filter(t=>t.kind==='copy').map(t=>[t.offset,t.bytes]),[[0,131072],[131072,131072],[262144,4096]]);
  assert.throws(()=>x.backbone._dispatchAttentionTile(x.encoder,x.state,2),/order/);
  assert.throws(()=>x.backbone._dispatchAttentionTile(x.encoder,x.state,3),/invalid/);
  assert.equal(x.trace.filter(t=>t.kind==='copy').reduce((n,t)=>n+t.bytes,0),65*1024*4);
}
assert.deepEqual(new Uint8Array(f.state.attnOutBuf.data),new Uint8Array(old.state.attnOutBuf.data));
// Aliased slots are one actual owned buffer, including exact native failure.
f.backbone._residentWorkOwner={buffers:new Set(Object.values(f.state).filter(x=>x?.device===f.device)),allocations:[]};
let blocked=true,destroys=0;const native=f.state.qBuf.destroy;
f.state.qBuf.destroy=function(){destroys++;if(blocked)throw Error('query retirement unresolved');return native.call(this);};
await assert.rejects(retireTwoStreamWork(f.backbone),/retirement/);
assert.ok(f.backbone._residentWorkOwner.buffers.has(f.state.qBuf));
blocked=false;await retireTwoStreamWork(f.backbone);
assert.equal(f.state.qBuf.destroyed,1);assert.equal(destroys,2);assert.equal(f.backbone._residentWorkOwner,null);
for(const value of [null,1,'yes',{},NaN]){
  const x=fixture(false),count=x.device.buffers.length;
  assert.throws(()=>x.backbone._createAttentionState({}, {}, {}, {},65,2,1024,32,value),/query.*reuse|boolean/i);
  assert.equal(x.device.buffers.length,count,'invalid mode refuses before allocation');
}
const defaults=new TwoStreamBackbone(fakeWeightDevice()).createAttentionForwardState({},1,{});
assert.equal(defaults.reuseAttentionQueryStorage,false);
const selected=new TwoStreamBackbone(fakeWeightDevice()).createAttentionForwardState({},1,{}, {residentFFN:true,reuseAttentionQueryStorage:true});
assert.equal(selected.reuseAttentionQueryStorage,true);
assert.equal(defineTwoStreamManifest({dutyGranularity:'attention-tile',N_img:1297,residentFFN:true,
  attentionRowsPerDuty:32,reuseAttentionQueryStorage:true}).metadata.reuseAttentionQueryStorage,true);
for(const value of [true,1,'yes',null])assert.throws(()=>new TwoStreamBackbone(fakeWeightDevice())
  .createAttentionForwardState({},1,{}, {reuseAttentionQueryStorage:value}),/query.*reuse|resident/i);
for(const options of [{reuseAttentionQueryStorage:true},{reuseAttentionQueryStorage:null},
  {reuseAttentionQueryStorage:true,residentFFN:true},
  {reuseAttentionQueryStorage:true,residentFFN:true,retireIntermediateBuffers:true}])
  await assert.rejects(runCooperativeTwoStream({device:{},backbone:{},N_img:1,...options}),/query.*reuse|resident/i);

// Actual resident consumer selects true before cooperative execution. The seam
// uses tiny synthetic source/embedding buffers, not complete-model evidence.
const code=fs.readFileSync(new URL('../src/lib/resident_two_stream.js',import.meta.url),'utf8')
  .replace(/^import .*;\n/gm,'').replace(/^export /gm,'');
const factory=vm.compileFunction(code+'\nreturn runResidentTwoStream;',
  ['createWeightPhaseSource','isLoaderMemoryBudget','captureGpuBufferAllocations',
    'dispatchTokenizerEmbedding','runCooperativeTwoStream','retireTwoStreamWork']);
const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:256*1024*1024,gpuBytes:1024*1024*1024});
await budget.requestOwnedDevice({requestDevice:async()=>device});
device.createCommandEncoder=()=>({finish(){return {};}});
const backbone={device,_uniformCache:new Map()},sentinel=Error('exact resident query-reuse selection');
let calls=0,sourceDisposed=0;
const run=factory(async()=>({template:{tokenizer:{}},describe:()=>[],withWeights:async(_,work)=>work({embeddings:{}}),
  dispose:async()=>{sourceDisposed++;}}),isLoaderMemoryBudget,captureGpuBufferAllocations,
  ()=>createEmptyBuffer(device,16),async options=>{
    calls++;assert.equal(options.reuseAttentionQueryStorage,true);
    assert.equal(options.residentFFN,true);assert.equal(options.retireIntermediateBuffers,true);
    assert.equal(options.dutyGranularity,'attention-tile');assert.equal(options.attentionRowsPerDuty,32);
    throw sentinel;
  },retireTwoStreamWork);
await assert.rejects(run({device,backbone,memoryBudget:budget,weightsUrl:'full-canonical-source',
  imageTokensBuf:{size:1297*1024*4},N_img:1297,onBeforePhase:async()=>{},onBeforeDuty:async()=>{},
  withResult:async()=>{},attentionRowsPerDuty:32}),error=>error===sentinel);
assert.equal(calls,1);assert.equal(sourceDisposed,1);assert.equal(backbone._residentAdapterOwner,null);
assert.ok(device.buffers.every(x=>x.destroyed===1));assert.equal(budget.snapshot().gpu.liveBytes,0);
device.destroy();budget.restore();
console.log('PASS complete query-row output reuse: every original query read before ordered overwrite, full result/tail/default/invalid mode; synthetic executor only');
console.log('PASS exact aliased-owner native-failure recovery and actual resident opt-in selection; unchanged full duty configuration, no native-math claim');
