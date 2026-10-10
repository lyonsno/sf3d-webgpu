import {createWeightPhaseSource} from './weights.js';
import {isLoaderMemoryBudget} from './loader_memory_budget.js';
import {createStorageBuffer,createEmptyBuffer} from './gpu.js';
import {computeCameraInput} from './inference.js';
import {SF3DImageTokenizer} from './sf3d_backbone.js';
import {runCooperativeDino} from './cooperative_dino.js';

/** Select real schema leaves; keep block indices rather than renumbering work. */
export function selectDinoPhase(template,{blockStart,blockEnd,isFirst,isLast}){
  if(template?.blocks?.length!==24||!Number.isSafeInteger(blockStart)||!Number.isSafeInteger(blockEnd)||
    blockStart<0||blockEnd<=blockStart||blockEnd>24||isFirst!==(blockStart===0)||isLast!==(blockEnd===24))
    throw TypeError('exact DINO phase indices and setup/final boundaries required');
  return {blocks:template.blocks.map((block,i)=>i>=blockStart&&i<blockEnd?block:null),
    ...(isFirst?{patchEmbed:template.patchEmbed,clsToken:template.clsToken,posEmbed:template.posEmbed}:{}),
    ...(isLast?{layernorm:template.layernorm}:{}),};
}

/**
 * Complete DINO encoder with immutable range weights and one real block/duty.
 * The invoker supplies fresh phase observation and owns physical admission;
 * this adapter's budget/destruction is only explicit resource accounting.
 * withResult consumes the complete GPU output before this adapter retires it.
 */
export async function runResidentDino({device,memoryBudget,weightsUrl,expectedWeightBytes,expectedSourceETag,
  imageChw,onBeforePhase,onProgress,withResult,foregroundOpportunities=null}){
  if(!isLoaderMemoryBudget(memoryBudget))throw TypeError('authenticated loader budget required');
  memoryBudget.assertDeviceAcquiredHere(device);
  if(!(imageChw instanceof Float32Array)||imageChw.length!==3*512*512||imageChw.some(v=>!Number.isFinite(v)))
    throw TypeError('complete finite SF3D CHW input required');
  if(typeof onBeforePhase!=='function'||typeof withResult!=='function')throw TypeError('fresh phase observer and complete result consumer required');
  let source,tokenizer,imageBuf,cameraInputBuf,cameraEmbedBuf,result,cameraLease,failure,failed=false;
  const phaseDescriptions=[];
  try{
    source=await createWeightPhaseSource(device,weightsUrl,{memoryBudget,expectedWeightBytes,expectedSourceETag});
    const before=async(name,selection,details={})=>{
      const descriptor={name,tensors:source.describe(selection),...details};
      phaseDescriptions.push(descriptor);await onBeforePhase(descriptor);
    };
    await before('camera',source.template.cameraEmbedder);
    tokenizer=new SF3DImageTokenizer(device);tokenizer.init();
    const camera=computeCameraInput();cameraLease=memoryBudget.reserveCpu(camera.byteLength,'resident-camera-input');
    imageBuf=createStorageBuffer(device,imageChw,0,'resident-image');
    cameraInputBuf=createStorageBuffer(device,camera,0,'resident-camera-input');
    cameraEmbedBuf=createEmptyBuffer(device,768*4,0,'resident-camera-embedding');
    await source.withWeights(source.template.cameraEmbedder,async weights=>{
      const encoder=device.createCommandEncoder();
      tokenizer._dispatchLinear(encoder,cameraInputBuf,cameraEmbedBuf,weights.weight,weights.bias,1,25,768);
      device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
    });
    const encoded=await runCooperativeDino({device,tokenizer,imageBuf,cameraEmbedBuf,weights:source.template.imageTokenizer,
      numBlocks:24,chunkBlocks:1,schedulingMode:'cooperative',foregroundOpportunities,onProgress,retireIntermediateBuffers:true,
      async withChunkWeights(boundary,work){
        const selection=selectDinoPhase(source.template.imageTokenizer,boundary);
        await before('dino-block-'+boundary.blockStart,selection,boundary);
        return source.withWeights(selection,work);
      },
    });
    result=encoded.result;
    if(result.N!==1297||result.tokensBuf?.size!==1297*1024*4)throw Error('complete DINO output shape changed');
    await onBeforePhase({name:'dino-output',tensors:[],outputBytes:result.tokensBuf.size});
    const value=await withResult(result);
    return {value,cooperative:encoded.report,loadingReport:source.loadingReport,weightPhases:source.phases,
      phaseDescriptions,shape:[1297,1024],blocks:24};
  }catch(error){failed=true;failure=error;throw error;}
  finally{
    // A failed prefix fence cannot authorize retirement of live backing.
    try{await device.queue.onSubmittedWorkDone();}
    catch(error){throw failed?new AggregateError([failure,error],'resident DINO failed and queue drain is unresolved',{cause:failure}):error;}
    const errors=[],attempt=fn=>{try{fn();}catch(error){errors.push(error);}};
    if(tokenizer)try{await tokenizer.retireCapturedWorkBuffers();}catch(error){errors.push(error);}
    for(const buffer of [result?.tokensBuf,imageBuf,cameraInputBuf,cameraEmbedBuf,...(tokenizer?tokenizer._uniformCache.values():[])])
      if(buffer)attempt(()=>buffer.destroy());
    if(cameraLease)attempt(()=>cameraLease.release());
    if(source)try{await source.dispose();}catch(error){errors.push(error);}
    if(errors.length)throw new AggregateError(failed?[failure,...errors]:errors,'resident DINO retirement failed',{cause:failure});
  }
}
