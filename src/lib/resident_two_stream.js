import {createWeightPhaseSource} from './weights.js';
import {isLoaderMemoryBudget} from './loader_memory_budget.js';
import {captureGpuBufferAllocations} from './gpu.js';
import {dispatchTokenizerEmbedding} from './tokenizer_embedding.js';
import {runCooperativeTwoStream,retireTwoStreamWork} from './cooperative_two_stream.js';

export function selectTwoStreamPhase(template,stageId){
  const t=template.backbone;
  if(stageId==='setup')return Object.fromEntries(['latentInit','normTriplane','projTriplane','normImage',
    'projImage','normLatent','projLatent'].map(key=>[key,t[key]]));
  if(stageId==='final')return {projOut:t.projOut};
  const match=/^block-([0-3])-(?:fuse-(in|out)|basic-([0-2]))$/.exec(stageId);
  if(!match)throw Error('exact two-stream stage selection required');
  const block=Number(match[1]),basic=match[3]===undefined?null:Number(match[3]);
  const selected=basic===null?{[match[2]==='in'?'fuseBlockIn':'fuseBlockOut']:
    t.mainBlocks[block][match[2]==='in'?'fuseBlockIn':'fuseBlockOut']}:
    {transformerBlocks:t.mainBlocks[block].transformerBlocks.map((value,i)=>i===basic?value:null)};
  return {mainBlocks:t.mainBlocks.map((value,i)=>i===block?selected:null)};
}

/** Complete same-backbone consumer; diagnostic admission remains caller-owned. */
export async function runResidentTwoStream({device,backbone,memoryBudget,weightsUrl,expectedWeightBytes,
  expectedSourceETag,imageTokensBuf,N_img,onBeforePhase,onBeforeDuty,onAfterDuty,onProgress,withResult,
  foregroundOpportunities=null,linearRowsPerDuty=128}){
  if(!isLoaderMemoryBudget(memoryBudget))throw TypeError('authenticated loader budget required');
  memoryBudget.assertDeviceAcquiredHere(device);
  if(backbone?.device!==device||backbone._residentAdapterOwner)throw Error('owned nonquarantined backbone required');
  if(N_img!==1297||imageTokensBuf?.size!==1297*1024*4)throw Error('complete SF3D DINO tokens required');
  if(typeof onBeforePhase!=='function'||typeof onBeforeDuty!=='function'||typeof withResult!=='function')
    throw TypeError('fresh weight/duty observers and complete result consumer required');
  const owner={buffers:new Set(),allocations:[],source:null};
  backbone._residentAdapterOwner=owner;
  let failure,failed=false;
  try{
    owner.source=await createWeightPhaseSource(device,weightsUrl,{memoryBudget,expectedWeightBytes,expectedSourceETag});
    const source=owner.source,template=source.template;
    await onBeforePhase({name:'two-stream-embedding-weights',tensors:source.describe(template.tokenizer),workGpuBytes:0});
    let embedding;
    await source.withWeights(template.tokenizer,async selected=>{
      // Conversion custody has ended, not necessarily physical backing.
      // A fresh post-upload observation includes anything still resident.
      await onBeforePhase({name:'two-stream-embedding-rearrange',tensors:[],workGpuBytes:3*1024*96*96*4+20});
      const encoder=device.createCommandEncoder();
      captureGpuBufferAllocations(()=>{
        embedding=dispatchTokenizerEmbedding(encoder,device,backbone.pipelines,selected.embeddings);
      },{ownedAllocations:owner.allocations});
      device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
    });
    const executed=await runCooperativeTwoStream({device,backbone,imageTokensBuf,N_img,weights:{},
      dutyGranularity:'attention-tile',linearRowsPerDuty,residentFFN:true,retireIntermediateBuffers:true,
      foregroundOpportunities,onProgress,onBeforeDuty,onAfterDuty,
      async withGroupWeights(group,work){
        const selection=selectTwoStreamPhase(template,group.stageId);
        await onBeforePhase({name:'two-stream-'+group.stageId,tensors:source.describe(selection)});
        return source.withWeights(selection,weights=>work({...weights,tokenizer_embeddings_buf:embedding}));
      },
    });
    owner.buffers.add(executed.result.buffer);
    if(executed.result.C!==1024||executed.result.N!==27648||executed.result.planeSize!==96||
      executed.result.buffer.size!==3*1024*96*96*4)throw Error('complete triplane feature shape changed');
    await onBeforeDuty({kind:'output',dutyId:'two-stream-output'},null);
    const value=await withResult(executed.result);
    return {value,cooperative:executed.report,loadingReport:source.loadingReport,weightPhases:source.phases,
      shape:[3,1024,96,96],blocks:4,basicBlocks:12,residentFFN:true,linearRowsPerDuty};
  }catch(error){failed=true;failure=error;throw error;}
  finally{
    try{await disposeResidentTwoStream(backbone);}
    catch(error){throw failed?new AggregateError([failure,error],'resident two-stream failed and cleanup is unresolved',{cause:failure}):error;}
  }
}

/** Recovery retains exact source/work inventories on the caller's backbone. */
export async function disposeResidentTwoStream(backbone){
  const owner=backbone._residentAdapterOwner;if(!owner)return;
  for(const allocation of owner.allocations)owner.buffers.add(allocation.buffer);
  owner.allocations.length=0;
  await backbone.device.queue.onSubmittedWorkDone();
  const errors=[];
  try{await retireTwoStreamWork(backbone);}catch(error){errors.push(error);}
  for(const buffer of [...owner.buffers,...backbone._uniformCache.values()]){
    try{buffer.destroy();owner.buffers.delete(buffer);
      for(const [key,value] of backbone._uniformCache)if(value===buffer)backbone._uniformCache.delete(key);
    }catch(error){errors.push(error);}
  }
  if(owner.source)try{await owner.source.dispose();owner.source=null;}catch(error){errors.push(error);}
  if(errors.length)throw new AggregateError(errors,'resident two-stream retirement failed');
  backbone._residentAdapterOwner=null;
}
