import {createWeightPhaseSource} from './weights.js';
import {isLoaderMemoryBudget} from './loader_memory_budget.js';
import {runCooperativePostProcessor,retirePostProcessorWork} from './cooperative_post_processor.js';

export function selectPostProcessorPhase(template, layerIndex) {
  const layers=template?.postProcessor?.convLayers;
  if(!Array.isArray(layers)||layers.length!==4||!Number.isSafeInteger(layerIndex)
      ||layerIndex<0||layerIndex>=4)throw TypeError('exact four-layer postprocessor selection required');
  return {convLayers:layers.map((layer,index)=>index===layerIndex?layer:null)};
}

/** The complete existing upsampler consumes borrowed backbone output; no eager weights. */
export async function runResidentPostProcessor({device,postProcessor,memoryBudget,weightsUrl,
  expectedWeightBytes,expectedSourceETag,triplanesBuf,onBeforePhase,onBeforeDuty,onAfterDuty,
  onProgress,withResult,foregroundOpportunities=null,channelsPerDuty=16}) {
  if(!isLoaderMemoryBudget(memoryBudget))throw TypeError('authenticated loader budget required');
  memoryBudget.assertDeviceAcquiredHere(device);
  if(postProcessor?.device!==device||postProcessor._residentAdapterOwner||postProcessor._postProcessorWorkOwner)
    throw Error('owned nonquarantined postprocessor handle required');
  if(triplanesBuf?.size!==113246208)throw Error('complete [3,1024,96,96] borrowed backbone input required');
  if(typeof onBeforePhase!=='function'||typeof onBeforeDuty!=='function'||typeof withResult!=='function')
    throw TypeError('fresh phase/duty observers and complete result consumer required');
  const owner={source:null,buffers:new Set()};
  postProcessor._residentAdapterOwner=owner;
  let failed=false,failure;
  try {
    owner.source=await createWeightPhaseSource(device,weightsUrl,
      {memoryBudget,expectedWeightBytes,expectedSourceETag});
    const source=owner.source;
    const executed=await runCooperativePostProcessor({
      device,triplanesBuf,weights:{convLayers:[null,null,null,null]},
      dutyGranularity:'channel-range',channelsPerDuty,schedulingMode:'cooperative',
      completionPolicy:'strict-prefix',residentWork:postProcessor,
      onBeforeDuty,onAfterDuty,onProgress,foregroundOpportunities,
      async withGroupWeights(group,work) {
        if(group.kind!=='conv-range')return work({convLayers:[null,null,null,null]});
        const selection=selectPostProcessorPhase(source.template,group.layerIndex);
        await onBeforePhase({name:'post-processor-plane-'+group.plane+'-'+group.stageId,
          tensors:source.describe(selection),workGpuBytes:0});
        return source.withWeights(selection,work);
      },
    });
    owner.buffers.add(executed.result.buffer);
    const result=executed.result;
    if(result.C!==40||result.H!==384||result.W!==384||result.numPlanes!==3
        ||result.buffer.size!==70778880)throw Error('complete postprocessor output shape changed');
    const value=await withResult(result);
    return {value,shape:[3,40,384,384],planes:3,convolutionsPerPlane:4,
      channelsPerDuty,cooperative:executed.report,loadingReport:source.loadingReport,
      weightPhases:source.phases};
  } catch(error) {failed=true;failure=error;throw error;}
  finally {
    try {await disposeResidentPostProcessor(postProcessor);}
    catch(error) {throw failed?new AggregateError([failure,error],
      'resident postprocessor failed and cleanup is unresolved',{cause:failure}):error;}
  }
}

/** Exact queue/destruction recovery remains reachable on the caller's handle. */
export async function disposeResidentPostProcessor(postProcessor) {
  const owner=postProcessor._residentAdapterOwner;
  if(!owner&&!postProcessor._postProcessorWorkOwner)return;
  await postProcessor.device.queue.onSubmittedWorkDone();
  const errors=[];
  try {
    await retirePostProcessorWork(postProcessor);
    postProcessor._postProcessorWorkOwner=null;
  } catch(error) {errors.push(error);}
  for(const buffer of owner?.buffers??[]) {
    try {buffer.destroy();owner.buffers.delete(buffer);}
    catch(error) {errors.push(error);}
  }
  if(owner?.source)try {await owner.source.dispose();owner.source=null;}
  catch(error) {errors.push(error);}
  if(errors.length)throw new AggregateError(errors,'resident postprocessor retirement failed');
  postProcessor._residentAdapterOwner=null;
}
