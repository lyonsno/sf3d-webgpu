import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {captureGpuBufferAllocations} from '../src/lib/gpu.js';
import {dispatchTokenizerEmbedding} from '../src/lib/tokenizer_embedding.js';
import {TwoStreamBackbone} from '../src/lib/two_stream.js';

// Actual adapter, unchanged rearrangement allocator, and actual final residual
// method. Synthetic weights/command host test custody, not GPU arithmetic,
// immutable backend conformance or physical reclamation.
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64};
const source=fs.readFileSync(new URL('../src/lib/resident_two_stream.js',import.meta.url),'utf8')
  .replace(/^import .*;\n/gm,'').replace(/^export /gm,'');
const factory=vm.compileFunction(source+'\nreturn {runResidentTwoStream,disposeResidentTwoStream};',
  ['createWeightPhaseSource','isLoaderMemoryBudget','captureGpuBufferAllocations',
    'dispatchTokenizerEmbedding','runCooperativeTwoStream','retireTwoStreamWork'],
  {parsingContext:vm.createContext({GPUBufferUsage:globalThis.GPUBufferUsage})});

function fixture({rematerialize=false,holdFinal=false,failFence=false,failDestroy=false,failFinalFence=false,failFinalDestroy=false,refuseReload=false}={}){
  const events=[],buffers=[];let blocked=false,releaseFinal,enteredFinal,round=0;
  const entered=new Promise(resolve=>enteredFinal=resolve),completeBytes=3*1024*96*96*4;
  const makeBuffer=(size,label='')=>{
    const buffer={size,label,destroyed:0,getMappedRange(){return new ArrayBuffer(size);},unmap(){},
      destroy(){if((failDestroy&&label.startsWith('embedding-output-'))||(failFinalDestroy&&label==='embedding-output-2'))throw Error('embedding destroy unresolved');
        buffer.destroyed++;events.push('destroy:'+label);}};
    buffers.push(buffer);return buffer;
  };
  const device={queue:{submit(){events.push('submit');},async onSubmittedWorkDone(){
    events.push('prefix');if(blocked)throw Error('consumer prefix unresolved');}},
    createBuffer({size,label}){return makeBuffer(size,size===completeBytes?'embedding-output-'+round:label);},
    createShaderModule(){return {};},createComputePipeline(){return {getBindGroupLayout(){return {};}};},
    createBindGroup(){return {};},createCommandEncoder(){return {finish(){return {};},
      beginComputePass(){return {setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}};}};},
    destroy(){throw Error('borrowed device destruction');}};
  const borrowed=makeBuffer(1297*1024*4,'borrowed-dino'),weights=makeBuffer(completeBytes,'phase-weight');
  const backbone={device,pipelines:{},_uniformCache:new Map(),_diagnosticBuffers:{}};
  const template={tokenizer:{embeddings:'embedding'},backbone:Object.fromEntries(
    ['latentInit','normTriplane','projTriplane','normImage','projImage','normLatent','projLatent','projOut']
      .map(key=>[key,{key}]))};
  template.backbone.mainBlocks=Array.from({length:4},()=>({transformerBlocks:[{},{},{}],fuseBlockIn:{},fuseBlockOut:{}}));
  let sourceDisposed=0,initial,finalOperand,completeResult;
  const weightSource={template,loadingReport:{source:'synthetic'},phases:[],describe(){return [];},
    async withWeights(selection,work){if(selection===template.tokenizer)round++;
      return work(selection===template.tokenizer?{embeddings:weights}:selection);},
    async dispose(){sourceDisposed++;events.push('source-dispose');}};
  const api=factory(async()=>weightSource,()=>true,captureGpuBufferAllocations,dispatchTokenizerEmbedding,
    async options=>{
      for(const stageId of ['setup','block-0-basic-0','final']){
        await options.withGroupWeights({stageId},async selected=>{
          if(stageId==='setup'){
            initial=selected.tokenizer_embeddings_buf;assert.equal(initial.size,completeBytes);assert.equal(initial.destroyed,0);
            await device.queue.onSubmittedWorkDone();events.push('setup-complete');if(failFence)blocked=true;
          }
          if(stageId==='final'){
            // This is the real final model method, not a fixture that declares
            // the residual absent. Its exact operand must be live and complete.
            backbone._dispatchLinear=()=>{};backbone._dispatchTranspose=()=>{};
            backbone._dispatchAdd=(encoder,destination,operand,count)=>{
              assert(operand,'complete final residual operand is mandatory');
              assert.equal(operand.size,completeBytes);assert.equal(operand.destroyed,0);
              assert.equal(count,completeBytes/4);finalOperand=operand;events.push('actual-final-residual');
            };
            const state={weights:selected,currentTriplane:makeBuffer(completeBytes,'borrowed-current-triplane')};
            TwoStreamBackbone.prototype._dispatchForwardFinal.call(backbone,{},state);
            enteredFinal();if(holdFinal)await new Promise(resolve=>releaseFinal=resolve);
            if(failFinalFence)blocked=true;
            await device.queue.onSubmittedWorkDone();events.push('final-complete');completeResult=state.result;
          }
        });
      }
      return {result:completeResult,report:{source:'synthetic'}};
    },async()=>{events.push('retire-work');});
  const options={device,backbone,memoryBudget:{assertDeviceAcquiredHere(value){assert.equal(value,device);}},
    weightsUrl:'synthetic',imageTokensBuf:borrowed,N_img:1297,rematerializeTokenizerEmbedding:rematerialize,
    async onBeforePhase(phase){events.push('phase:'+phase.name);
      if(refuseReload&&phase.name==='two-stream-embedding-rematerialize-weights')throw Error('reload refused');
      if(rematerialize&&phase.name==='two-stream-block-0-basic-0')assert.equal(initial.destroyed,1);
    },async onBeforeDuty(){},async withResult(result){assert.equal(result,completeResult);
      assert.equal(result.buffer.destroyed,0);assert.equal(borrowed.destroyed,0);events.push('consumer');return 'complete';}};
  return {...api,options,events,buffers,backbone,borrowed,entered,
    release(){releaseFinal();},unblock(){blocked=false;failDestroy=false;failFinalDestroy=false;},
    get initial(){return initial;},get finalOperand(){return finalOperand;},
    get round(){return round;},get sourceDisposed(){return sourceDisposed;}};
}

for(const rematerialize of [false,true]){
  const x=fixture({rematerialize,holdFinal:true}),running=x.runResidentTwoStream(x.options);
  // A broken final may reject before it can signal the held consumer boundary.
  const outcome=await Promise.race([x.entered.then(()=>({entered:true})),running.then(()=>({completed:true}),error=>({error}))]);
  if(outcome.error)throw outcome.error;assert.equal(outcome.entered,true);
  assert.equal(x.finalOperand.destroyed,0,'final prefix still borrows exact embedding');
  if(rematerialize){assert.notEqual(x.initial,x.finalOperand);assert.equal(x.initial.destroyed,1);}
  else assert.equal(x.initial,x.finalOperand);
  x.release();const result=await running;
  assert.equal(result.value,'complete');assert.equal(result.rematerializeTokenizerEmbedding,rematerialize);
  assert.equal(result.tokenizerEmbeddingRetirement.status,'retired');
  assert.equal(result.tokenizerEmbeddingRetirement.afterGroup,'final');
  assert.equal(result.tokenizerEmbeddingRetirement.logicalBytes,113246228);
  assert.equal(result.tokenizerEmbeddingRetirement.bufferCount,2);
  assert.match(result.tokenizerEmbeddingRetirement.authority,/logical.*not physical/);
  assert.equal(x.round,rematerialize?2:1);
  assert.equal(x.initial.destroyed,1);assert.equal(x.finalOperand.destroyed,1);
  assert.equal(x.borrowed.destroyed,0);assert.equal(x.sourceDisposed,1);
  assert.equal(x.backbone._residentAdapterOwner,null);
  assert(x.events.indexOf('actual-final-residual')<x.events.indexOf('final-complete'));
  assert(x.events.indexOf('final-complete')<x.events.indexOf('destroy:'+x.finalOperand.label));
  if(rematerialize){
    assert(x.events.indexOf('destroy:'+x.initial.label)<x.events.indexOf('phase:two-stream-embedding-rematerialize-weights'));
    assert(x.events.indexOf('phase:two-stream-embedding-rematerialize-rearrange')<x.events.indexOf('phase:two-stream-final'));
  }
}
console.log('ok actual final model method consumes complete live retained or rebuilt embedding until final prefix');

for(const failure of [{failFence:true},{failDestroy:true},{refuseReload:true},{failFinalFence:true},{failFinalDestroy:true}]){
  const x=fixture({rematerialize:true,...failure});let caught;
  try{await x.runResidentTwoStream(x.options);}catch(error){caught=error;}
  assert(caught);assert.equal(x.events.includes('actual-final-residual'),!!(failure.failFinalFence||failure.failFinalDestroy));
  assert.equal(x.borrowed.destroyed,0);
  if(failure.refuseReload){assert.equal(x.backbone._residentAdapterOwner,null);assert.equal(x.initial.destroyed,1);}
  else{
    const owner=x.backbone._residentAdapterOwner;assert(owner);
    const unresolved=failure.failFinalFence||failure.failFinalDestroy?x.finalOperand:x.initial;
    assert.equal(unresolved.destroyed,0);
    assert(owner.buffers.has(unresolved)||owner.allocations.some(a=>a.buffer===unresolved));
    await assert.rejects(x.runResidentTwoStream(x.options),/nonquarantined/);
    x.unblock();await x.disposeResidentTwoStream(x.backbone);
    assert.equal(unresolved.destroyed,1);assert.equal(x.backbone._residentAdapterOwner,null);
  }
  assert.equal(x.sourceDisposed,1);
}
console.log('ok unresolved setup retirement or refused immutable reload blocks final and retains exact recovery/borrowed custody');
