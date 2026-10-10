import fs from 'node:fs';
import {createHash} from 'node:crypto';
// Component accounting for the source-fixed encoder, not opaque driver backing.
export function dinoPhaseDemand(phase){
  if(!['preprocess','camera','dino-output'].includes(phase.name)&&!/^dino-block-(?:[0-9]|1[0-9]|2[0-3])$/.test(phase.name))throw Error('unknown encoder phase');
  const tensors=phase.tensors??[];
  let weightGpuBytes=0,rangeCpuBytes=0;
  for(const t of tensors){
    if(!Number.isSafeInteger(t.size)||t.size<=0||![0,1].includes(t.dtype))throw Error('invalid tensor component');
    weightGpuBytes+=t.size*(t.dtype===1?2:1);
    // Entire response + destination + FP16 expansion may coexist for one unit.
    rangeCpuBytes=Math.max(rangeCpuBytes,2*t.size+(t.dtype===1?2*t.size:0));
  }
  const tokenBytes=1297*1024*4;
  const components=[];
  const add=(name,bytes,scope)=>components.push({name,bytes,scope});
  add('selected-weight-uploads',weightGpuBytes,'gpu');add('one-range-response-destination-conversion',rangeCpuBytes,'cpu');
  if(phase.name==='preprocess'){
    // Caller input fixed503x503; condition framing/crop/resize allocations are
    // included conservatively, not substituted with prior Lanczos patch input.
    if(phase.width!==503||phase.height!==503)throw Error('source-fixed chair dimensions required for preprocessing demand');
    add('decoded-pixels-and-framing-intermediates',80*1024*1024,'cpu');
  }
  if(phase.name==='camera')add('image-camera-input-output-uniform',3*512*512*4+100+768*4+32,'gpu');
  if(phase.name==='dino-block-0'){
    add('nine-token-stores-and-four-token-hidden',13*tokenBytes,'gpu');
    add('full-attention-scores',16*1297*1297*4,'gpu');
    // Account all five snapshots up front, rather than pretending the later
    // block0/1/2/23 diagnostic allocations are absent.
    add('five-diagnostic-token-snapshots',5*tokenBytes,'gpu');
    add('modulation-silu-and-uniforms',2*1024*4+768*4+4096,'gpu');
  }
  if(phase.name==='dino-output'){add('complete-readback',tokenBytes,'gpu');add('persisted-readback-response',2*tokenBytes,'cpu');}
  return {components,weightGpuBytes,rangeCpuBytes,workGpuBytes:components.filter(c=>c.scope==='gpu'&&c.name!=='selected-weight-uploads').reduce((n,c)=>n+c.bytes,0),
    requiredBytes:components.reduce((n,c)=>n+c.bytes,0),authority:'identified explicit new backing; no opaque driver or physical reclamation guarantee'};
}

export function inspectDinoOutput(filename){
  const bytes=fs.readFileSync(filename);if(bytes.length!==1297*1024*4)throw Error('missing or partial complete DINO output');
  let finite=0,nonzero=0,min=Infinity,max=-Infinity;
  for(let i=0;i<bytes.length;i+=4){const value=bytes.readFloatLE(i);if(Number.isFinite(value))finite++;if(value!==0)nonzero++;min=Math.min(min,value);max=Math.max(max,value);}
  if(finite!==1297*1024||!nonzero)throw Error('nonfinite or blank DINO output');
  return {sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length,finite,nonzero,min,max};
}

export function inspectDinoInputBytes(bytes){
  if(bytes.length!==3*512*512*4)throw Error('missing or partial complete DINO input');
  for(let i=0;i<bytes.length;i+=4)if(!Number.isFinite(bytes.readFloatLE(i)))throw Error('nonfinite DINO input');
  return {sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length,finite:3*512*512};
}

export function acceptResidentDino(report){
  const errors=[],require=(condition,message)=>{if(!condition)errors.push(message);};
  require(report.status==='passed'&&report.phase==='complete','run not complete');
  require(report.source?.clean&&report.source.revision===report.requested?.revision,'source not exact clean requested head');
  require(report.backend?.isFallbackAdapter===false&&/apple/i.test(report.backend?.vendor??''),'effective native Apple/nonfallback missing');
  require(report.artifact?.servedSha256===report.artifact?.sha256&&!!report.artifact?.sha256&&report.artifact?.kitVersion==='0.1.53'&&report.artifact.sourceRevision===report.source?.revision,'effective built/served kit artifact mismatch');
  require(report.canonicalSource?.sha256===report.requested?.weightsSha256&&!!report.canonicalSource?.sha256,'immutable canonical source missing');
  const phases=report.dino?.weightPhases;
  require(Array.isArray(phases)&&phases.length===25&&phases.every(p=>p.status==='completed-retired'),'camera or24 complete weight phases missing');
  require(report.dino?.loadingReport?.sourceETag===report.canonicalSource?.etag&&!!report.canonicalSource?.etag&&
    report.dino.loadingReport.expectedWeightBytes===report.canonicalSource.byteLength,'effective range source identity mismatch');
  const cooperative=report.dino?.cooperative,boundary=cooperative?.boundaries?.[0];
  require(cooperative?.status==='succeeded'&&cooperative.schedulingMode==='cooperative'&&
    cooperative.queueCompletionAuthority==='per-gpu-duty-prefix-fence'&&boundary?.completedItems===24&&boundary.totalItems===24&&boundary.actualRangeCount===24,
    'complete effective kit encoder execution missing');
  const observations=report.phaseObservations;
  require(Array.isArray(observations)&&observations.length===27&&observations.every(o=>o.verdict==='admitted'&&o.host?.source==='live-macos'&&
    o.host.hostname===report.source?.hostname&&o.host.model==='Mac14,9'&&o.host.processor==='Apple M2 Pro'&&!o.host.observerErrors?.length&&
    o.host.hostFreeBytes>=o.demand?.requiredBytes&&o.process?.coverage==='sampled-owned-process-tree'&&o.process?.lastObservation?.runId===report.runId&&
    o.process.lastObservation.rootPid===report.rootPid&&o.process.lastObservation.status==='observed'&&
    o.process.lastObservation.sampledAggregatePhysicalFootprintBytes+o.demand.requiredBytes<=report.requested?.processBudgetBytes),
    'fresh effective-host phase observations incomplete');
  require(observations?.map(o=>o.phase).join(',')===['preprocess','camera',...Array.from({length:24},(_,i)=>'dino-block-'+i),'dino-output'].join(','),'phase order or all24 blocks missing');
  require(report.validationError===null&&!report.memorySafety,'native validation or original process safety failure');
  require(report.budget?.cpu.liveBytes===0&&report.budget?.gpu.liveBytes===0,'unretired owned resources');
  require(report.processObservation?.status==='observed'&&report.processObservation.coverage==='sampled-owned-process-tree'&&
    report.processObservation.sampledPeakAggregatePhysicalFootprintBytes<=report.requested?.processBudgetBytes,'terminal process guard missing');
  require(report.cleanup?.browser?.exitObserved&&report.cleanup?.server==='closed','exact owned cleanup missing');
  try{
    const output=inspectDinoOutput(report.evidencePaths?.output);
    require(output.sha256===report.output?.sha256,'raw output hash mismatch');
  }catch(error){errors.push('raw output: '+error.message);}
  try{
    const input=inspectDinoInputBytes(fs.readFileSync(report.evidencePaths?.input));
    require(input.sha256===report.inputTensor?.sha256&&input.sha256===report.inputConsumed?.sha256,'raw input differs from the complete CHW tensor consumed');
  }
  catch(error){errors.push('raw input: '+error.message);}
  return {ok:!errors.length,errors,authority:'complete native DINO encoder only; not full SF3D, reference parity or production fit'};
}
