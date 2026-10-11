import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {createPostProcessorChannelDutyPlan,describePostProcessorDutyDemand} from '../src/lib/cooperative_post_processor.js';
import {acceptResidentTwoStream,residentTwoStreamExpectedPhases} from './resident_two_stream_acceptance.mjs';
import {phaseHostHeadroomIsAdmitted} from './memory_admission.mjs';
export const POST_PROCESSOR_OUTPUT_BYTES=70778880;

/** Preserve the entire fresh observation and its authority-bearing metadata.
 * Lifetime history stays in the raw journal and terminal monitor summary. */
export function projectFreshProcessEvidence(observation){
  const {processes,unavailableProcessObservations,...fresh}=observation;
  return fresh;
}

export function validateCompleteReadbackChunk({offset,bytes,receivedBytes,totalBytes}){
  if(!Number.isSafeInteger(offset)||offset!==receivedBytes||offset%4||
    !Number.isSafeInteger(bytes)||bytes<=0||bytes%4||bytes>128*1024*4||offset+bytes>totalBytes)
    throw Error('partial, skipped, duplicate or oversized readback chunk');
  return offset+bytes;
}

export function residentBackbonePrefix(report){
  const count=residentTwoStreamExpectedPhases(report.requested?.attentionRowsPerDuty,report.requested?.rematerializeTokenizerEmbedding??false).length;
  return {...report,expectedPhaseOrder:report.expectedPhaseOrder?.slice(0,count),
    phaseObservations:report.phaseObservations?.slice(0,count)};
}

export function postProcessorExpectedPhases(channelsPerDuty){
  const phases=['post-processor-output-allocation'];
  for(const duty of createPostProcessorChannelDutyPlan(channelsPerDuty).duties){
    if(duty.kind==='conv-range'&&duty.rangeIndex===0)
      phases.push('post-processor-plane-'+duty.plane+'-'+duty.stageId);
    phases.push('post-processor-duty-'+duty.dutyIndex);
  }
  phases.push('post-processor-output');
  return phases;
}

export function postProcessorPhaseDemand(phase,channelsPerDuty){
  const plan=createPostProcessorChannelDutyPlan(channelsPerDuty);
  let weightGpuBytes=0,rangeCpuBytes=0,workGpuBytes=0;
  if(phase.name==='post-processor-output-allocation'){
    if(phase.duty?.kind!=='output-allocation'||phase.duty.workGpuBytes!==POST_PROCESSOR_OUTPUT_BYTES)
      throw Error('effective complete output allocation mismatch');
    workGpuBytes=POST_PROCESSOR_OUTPUT_BYTES;
  }else if(phase.name==='post-processor-output'){
    workGpuBytes=128*1024*4;rangeCpuBytes=2*workGpuBytes;
  }else if(phase.name.startsWith('post-processor-duty-')){
    const index=Number(phase.name.slice('post-processor-duty-'.length)),expected=plan.duties[index];
    if(!expected||Object.keys(expected).some(k=>phase.duty?.[k]!==expected[k]))
      throw Error('effective complete postprocessor duty identity mismatch');
    workGpuBytes=describePostProcessorDutyDemand(expected).workGpuBytes;
    if(phase.duty.workGpuBytes!==workGpuBytes)throw Error('effective postprocessor new backing mismatch');
  }else{
    const duty=plan.duties.find(d=>'post-processor-plane-'+d.plane+'-'+d.stageId===phase.name&&d.kind==='conv-range');
    if(!duty)throw Error('unknown complete postprocessor phase');
    const expected=['weight','bias'].map(s=>'post_processor.upsample.'+(2*duty.layerIndex)+'.'+s);
    if(phase.tensors?.map(t=>t.name).join(',')!==expected.join(','))throw Error('effective selected convolution weights mismatch');
    for(const tensor of phase.tensors){
      if(!Number.isSafeInteger(tensor.size)||tensor.size<=0||![0,1].includes(tensor.dtype))
        throw Error('invalid postprocessor tensor component');
      weightGpuBytes+=tensor.size*(tensor.dtype===1?2:1);
      rangeCpuBytes=Math.max(rangeCpuBytes,2*tensor.size+(tensor.dtype===1?2*tensor.size:0));
    }
  }
  return {components:[{name:'selected-weight-uploads',bytes:weightGpuBytes,scope:'gpu'},
    {name:'one-range-response-destination-conversion',bytes:rangeCpuBytes,scope:'cpu'},
    {name:'identified-postprocessor-new-backing',bytes:workGpuBytes,scope:'gpu'}],
    weightGpuBytes,rangeCpuBytes,workGpuBytes,requiredBytes:weightGpuBytes+rangeCpuBytes+workGpuBytes,
    authority:'identified explicit new backing; no opaque driver or physical reclamation guarantee'};
}

export function inspectPostProcessorOutput(filename){
  const fd=fs.openSync(filename,'r'),hash=createHash('sha256'),chunk=Buffer.alloc(65536);
  let bytes=0,finite=0,nonzero=0,min=Infinity,max=-Infinity;
  try{
    if(fs.fstatSync(fd).size!==POST_PROCESSOR_OUTPUT_BYTES)throw Error('missing/partial complete postprocessor output');
    for(let count;(count=fs.readSync(fd,chunk,0,chunk.length,null))>0;){
      if(count%4)throw Error('misaligned postprocessor output');
      hash.update(chunk.subarray(0,count));bytes+=count;
      for(let i=0;i<count;i+=4){const value=chunk.readFloatLE(i);if(Number.isFinite(value))finite++;
        if(value!==0)nonzero++;min=Math.min(min,value);max=Math.max(max,value);}
    }
  }finally{fs.closeSync(fd);}
  if(finite!==POST_PROCESSOR_OUTPUT_BYTES/4||!nonzero)throw Error('nonfinite/blank complete postprocessor output');
  return {bytes,finite,nonzero,min,max,sha256:hash.digest('hex')};
}

/** Local semantic completion judge; raw artifact and backbone joins are separate below. */
export function acceptPostProcessorCompletion(report){
  const errors=[],require=(ok,message)=>{if(!ok)errors.push(message)};
  require(report.requested?.throughPostProcessor===true&&report.requested.throughBackbone===true,
    'effective requested complete postprocessor route missing');
  let plan,expected;
  try{plan=createPostProcessorChannelDutyPlan(report.requested?.postChannelsPerDuty);
    expected=postProcessorExpectedPhases(plan.channelsPerDuty);}
  catch(error){return {ok:false,errors:[...errors,error.message]};}
  const post=report.postProcessor,cooperative=post?.cooperative,telemetry=cooperative?.adapterTelemetry,boundary=cooperative?.boundaries?.[0];
  require(post?.shape?.join(',')==='3,40,384,384'&&post.planes===3&&post.convolutionsPerPlane===4,
    'complete postprocessor shape/planes/convolutions missing');
  require(post?.channelsPerDuty===plan.channelsPerDuty&&telemetry?.channelsPerDuty===plan.channelsPerDuty&&
    telemetry.dutyGranularity==='channel-range'&&telemetry.residentWork===true,'effective resident configuration mismatch');
  require(cooperative?.status==='succeeded'&&cooperative.schedulingMode==='cooperative'&&
    cooperative.routeId==='sf3d.image-to-mesh.webgpu-local.v0'&&
    cooperative.manifestId==='sf3d.post-processor-channel-cooperative-boundaries.v0'&&
    boundary?.boundaryId==='post-processor-triplane-channel-ranges'&&
    cooperative.queueCompletionAuthority==='per-gpu-duty-prefix-fence'&&boundary?.completedItems===plan.duties.length&&
    boundary.totalItems===plan.duties.length&&boundary.actualRangeCount===plan.duties.length,'complete native postprocessor prefix execution missing');
  require(telemetry?.stageDuties?.length===plan.duties.length&&plan.duties.every((d,i)=>
    Object.keys(d).every(k=>telemetry?.stageDuties?.[i]?.[k]===d[k])),'effective complete channel coverage mismatch');
  const weights=post?.weightPhases;
  require(weights?.length===12&&weights.every((phase,index)=>phase.status==='completed-retired'&&
    phase.tensorNames?.join(',')===['weight','bias'].map(s=>'post_processor.upsample.'+(2*(index%4))+'.'+s).join(',')),
    'complete exact twelve selected convolution phases missing');
  require(post?.loadingReport?.sourceETag===report.canonicalSource?.etag&&!!report.canonicalSource?.etag&&
    post.loadingReport.expectedWeightBytes===report.canonicalSource.byteLength,'effective postprocessor range source mismatch');
  const observed=report.phaseObservations?.filter(p=>p.phase.startsWith('post-processor-'));
  require(observed?.map(p=>p.phase).join(',')===expected.join(','),'complete actual postprocessor observation order missing');
  require(observed?.every(p=>p.verdict==='admitted'&&phaseHostHeadroomIsAdmitted(report,p)&&
    p.host?.hostname===report.source?.hostname&&p.host.model==='Mac14,9'&&p.host.processor==='Apple M2 Pro'&&
    p.process?.coverage==='sampled-owned-process-tree'&&p.process.lastObservation?.status==='observed'&&
    p.process.lastObservation.runId===report.runId&&p.process.lastObservation.rootPid===report.rootPid&&
    p.process.freshness?.route==='new-probe-after-request'&&p.process.freshness.requestedAtUnixMs>=p.phaseRequestedAtUnixMs&&
    p.process.lastObservation.atUnixMs>=p.process.freshness.requestedAtUnixMs&&
    p.process.freshness.probeAtUnixMs===p.process.lastObservation.atUnixMs&&
    Number.isSafeInteger(p.process.sampleCount)&&p.process.sampleCount>0&&
    p.process.freshness.observationIndex===p.process.sampleCount&&
    p.process.lastObservation.sampledAggregatePhysicalFootprintBytes+p.demand.requiredBytes<=report.requested.processBudgetBytes),
    'fresh current guard-bound postprocessor observations missing');
  for(const phase of observed??[]){
    try{
      const expected=postProcessorPhaseDemand(phase.descriptor,plan.channelsPerDuty);
      require(['weightGpuBytes','rangeCpuBytes','workGpuBytes','requiredBytes'].every(k=>phase.demand?.[k]===expected[k])&&
        expected.components.every(c=>phase.demand?.components?.some(actual=>
          actual.name===c.name&&actual.bytes===c.bytes&&actual.scope===c.scope)),
        'identified new backing differs from effective source duty');
    }catch(error){errors.push(error.message);}
  }
  require(report.postProcessorReadbackBytes===POST_PROCESSOR_OUTPUT_BYTES&&
    report.postProcessorOutput?.bytes===POST_PROCESSOR_OUTPUT_BYTES&&
    report.postProcessorOutput.finite===POST_PROCESSOR_OUTPUT_BYTES/4&&report.postProcessorOutput.nonzero>0,
    'complete finite nonblank postprocessor readback missing');
  return {ok:!errors.length,errors};
}
export function acceptNativeResidentPostProcessor(report){
  const stage=acceptPostProcessorCompletion(report),errors=[...stage.errors];
  try{errors.push(...acceptResidentTwoStream(residentBackbonePrefix(report)).errors);}
  catch(error){errors.push(error.message);}
  try{
    const expected=[...residentTwoStreamExpectedPhases(report.requested.attentionRowsPerDuty,report.requested.rematerializeTokenizerEmbedding??false),
      ...postProcessorExpectedPhases(report.requested.postChannelsPerDuty)];
    if(report.phaseObservations?.map(p=>p.phase).join(',')!==expected.join(','))errors.push('whole requested execution phase order mismatch');
  }catch(error){errors.push(error.message);}
  try{
    const output=inspectPostProcessorOutput(report.evidencePaths?.postProcessorOutput);
    if(output.sha256!==report.postProcessorOutput?.sha256)errors.push('retained postprocessor output hash mismatch');
  }catch(error){errors.push('raw postprocessor output: '+error.message);}
  return {ok:!errors.length,errors,authority:'complete native DINO/backbone/postprocessor only; no mesh/material/GLB/foreground/physical-fit claim'};
}
