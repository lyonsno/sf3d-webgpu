import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {captureGpuBufferAllocations} from '../src/lib/gpu.js';
import {dispatchTokenizerEmbedding} from '../src/lib/tokenizer_embedding.js';

// Execute the actual adapter and unchanged embedding allocator with a tiny
// synthetic command/source host. No model arithmetic, native prefix or
// physical reclamation claim is made by this test.
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64};
const source=fs.readFileSync(new URL('../src/lib/resident_two_stream.js',import.meta.url),'utf8')
  .replace(/^import .*;\n/gm,'').replace(/^export /gm,'');
const factory=vm.compileFunction(source+'\nreturn {runResidentTwoStream,disposeResidentTwoStream};',
  ['createWeightPhaseSource','isLoaderMemoryBudget','captureGpuBufferAllocations',
    'dispatchTokenizerEmbedding','runCooperativeTwoStream','retireTwoStreamWork'],
  {parsingContext:vm.createContext({GPUBufferUsage:globalThis.GPUBufferUsage})});

function fixture({holdSetup=false,failSetup=false,failFence=false,failDestroy=false}={}){
  const events=[],buffers=[];let setupComplete=false,blocked=false,releaseSetup,enteredSetup;
  const entered=new Promise(resolve=>enteredSetup=resolve);
  const completeBytes=3*1024*96*96*4;
  const makeBuffer=(size,label='')=>{
    const buffer={size,label,destroyed:0,getMappedRange(){return new ArrayBuffer(size);},unmap(){},
      destroy(){
        if(failDestroy&&label==='embedding-output')throw Error('embedding destroy unresolved');
        buffer.destroyed++;events.push('destroy:'+label);
      }};
    buffers.push(buffer);return buffer;
  };
  const device={queue:{submit(){events.push('submit');},async onSubmittedWorkDone(){
    events.push('prefix');if(blocked)throw Error('setup prefix unresolved');}},
    createBuffer({size,label}){return makeBuffer(size,size===completeBytes?'embedding-output':label);},
    createShaderModule(){return {};},createComputePipeline(){return {getBindGroupLayout(){return {};}};},
    createBindGroup(){return {};},createCommandEncoder(){return {finish(){return {};},
      beginComputePass(){return {setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}};}};},
    destroy(){throw Error('borrowed device destruction');}};
  const borrowed=makeBuffer(1297*1024*4,'borrowed-dino'),weights=makeBuffer(completeBytes,'phase-weight');
  const backbone={device,pipelines:{},_uniformCache:new Map()};
  const template={tokenizer:{embeddings:'embedding'},backbone:Object.fromEntries(
    ['latentInit','normTriplane','projTriplane','normImage','projImage','normLatent','projLatent','projOut']
      .map(key=>[key,{key}]))};
  template.backbone.mainBlocks=Array.from({length:4},()=>({transformerBlocks:[{},{},{}],fuseBlockIn:{},fuseBlockOut:{}}));
  let sourceDisposed=0,completeResult;
  const weightSource={template,loadingReport:{source:'synthetic'},phases:[],describe(){return [];},
    async withWeights(selection,work){return work(selection===template.tokenizer?{embeddings:weights}:selection);},
    async dispose(){sourceDisposed++;events.push('source-dispose');}};
  let embedding;
  const api=factory(async()=>weightSource,()=>true,captureGpuBufferAllocations,dispatchTokenizerEmbedding,
    async options=>{
      for(const stageId of ['setup','block-0-basic-0','final']){
        await options.withGroupWeights({stageId},async selected=>{
          if(stageId==='setup'){
            embedding=selected.tokenizer_embeddings_buf;
            assert.equal(embedding.size,completeBytes);assert.equal(embedding.destroyed,0);
            enteredSetup();if(holdSetup)await new Promise(resolve=>releaseSetup=resolve);
            if(failSetup){blocked=failFence;throw Error('setup computation failed');}
            await device.queue.onSubmittedWorkDone();setupComplete=true;events.push('setup-complete');
            if(failFence)blocked=true;
          }else{
            assert.equal(embedding.destroyed,1,'consumed embedding must retire before next group work');
            assert.equal(selected.tokenizer_embeddings_buf,undefined,'later weights must not retain a destroyed embedding');
          }
        });
      }
      completeResult={buffer:makeBuffer(completeBytes,'complete-result'),C:1024,N:27648,planeSize:96};
      return {result:completeResult,report:{source:'synthetic'}};
    },async()=>{events.push('retire-work');});
  const options={device,backbone,memoryBudget:{assertDeviceAcquiredHere(value){assert.equal(value,device);}},
    weightsUrl:'synthetic',imageTokensBuf:borrowed,N_img:1297,
    async onBeforePhase(phase){
      events.push('phase:'+phase.name);
      if(phase.name==='two-stream-block-0-basic-0'){
        assert.equal(setupComplete,true);
        assert.equal(embedding.destroyed,1,'consumed embedding must retire before next fresh weight observation');
        assert.equal(buffers.find(b=>b.label==='tokenizer-rearrange-uniform').destroyed,1);
      }
    },async onBeforeDuty(){},async withResult(result){
      assert.equal(result,completeResult);assert.equal(result.buffer.destroyed,0);
      assert.equal(borrowed.destroyed,0);events.push('consumer');return 'complete';}};
  return {...api,options,events,buffers,backbone,device,borrowed,entered,
    release(){releaseSetup();},unblock(){blocked=false;failDestroy=false;},
    get embedding(){return embedding;},get sourceDisposed(){return sourceDisposed;}};
}

const success=fixture({holdSetup:true});
const running=success.runResidentTwoStream(success.options);
await success.entered;
assert.equal(success.embedding.destroyed,0,'submitted setup still owns its embedding');
assert.equal(success.buffers.find(b=>b.label==='tokenizer-rearrange-uniform').destroyed,0);
success.release();
const result=await running;
assert.equal(result.value,'complete');
assert.equal(result.tokenizerEmbeddingRetirement.status,'retired');
assert.equal(result.tokenizerEmbeddingRetirement.afterGroup,'setup');
assert.equal(result.tokenizerEmbeddingRetirement.logicalBytes,113246228);
assert.equal(result.tokenizerEmbeddingRetirement.bufferCount,2);
assert.match(result.tokenizerEmbeddingRetirement.authority,/logical.*not physical/);
assert.equal(success.embedding.destroyed,1,'terminal cleanup must not destroy an already retired embedding twice');
assert.equal(success.borrowed.destroyed,0);assert.equal(success.sourceDisposed,1);
assert.equal(success.backbone._residentAdapterOwner,null);
assert(success.events.indexOf('setup-complete')<success.events.indexOf('destroy:embedding-output'));
assert(success.events.indexOf('destroy:embedding-output')<success.events.indexOf('phase:two-stream-block-0-basic-0'));
console.log('ok actual resident adapter retires only captured embedding/uniform after complete setup and before fresh intake');

for(const failure of [{failFence:true},{failSetup:true,failFence:true},{failDestroy:true}]){
  const x=fixture(failure);
  await assert.rejects(x.runResidentTwoStream(x.options),error=>
    /setup prefix unresolved|retirement failed/.test(JSON.stringify(error,(key,value)=>
      value instanceof Error||value?.message?{message:value.message,errors:value.errors}:value)));
  assert(!x.events.includes('phase:two-stream-block-0-basic-0'),'failed retirement must prevent next weight intake');
  assert.equal(x.embedding.destroyed,0,'unresolved backing remains in actual owner inventory');
  const owner=x.backbone._residentAdapterOwner;assert(owner);
  assert(owner.buffers.has(x.embedding)||owner.allocations.some(a=>a.buffer===x.embedding));
  assert.equal(x.borrowed.destroyed,0);assert.equal(x.sourceDisposed,failure.failDestroy?1:0);
  await assert.rejects(x.runResidentTwoStream(x.options),/nonquarantined/);
  x.unblock();await x.disposeResidentTwoStream(x.backbone);
  assert.equal(x.embedding.destroyed,1);assert.equal(x.backbone._residentAdapterOwner,null);
  assert.equal(x.sourceDisposed,1);assert.equal(x.borrowed.destroyed,0);
}
console.log('ok failed setup/fence/destruction preserves quarantine and exact owned recovery without borrowed disposal');
