import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {acceptResidentDino} from './resident_dino_acceptance.mjs';
const D=1024,N=27648,L=3089,I=1297,tri=N*D*4,latent=L*D*4;
const scratch=rows=>rows*(2*D+3*4096)*4;
const attention=(q,kv)=>(2*q+2*kv)*D*4+16*128*kv*4+128*D*4;
/** Approved complete1297-token/128-row graph; independently checked against
 * the source planner by test_two_stream_review_contracts. Reporter input is
 * never the authority for a shorter sequence. */
export function residentTwoStreamExpectedPhases(){
  const names=['preprocess','camera',...Array.from({length:24},(_,i)=>'dino-block-'+i),'dino-output',
    'two-stream-embedding-weights','two-stream-embedding-rearrange'];
  const add=(stage,indices)=>{names.push('two-stream-'+stage);
    for(const index of indices)names.push('two-stream-duty-'+index);};
  add('setup',[0]);
  for(let b=0;b<4;b++){
    const start=1+b*837;
    add('block-'+b+'-fuse-in',[start,start+26]);
    for(let basic=0;basic<3;basic++){
      const first=start+27+basic*53;
      add('block-'+b+'-basic-'+basic,[first,first+26,first+52]);
    }
    const out=start+186;
    add('block-'+b+'-fuse-out',[out,out+217,out+433,out+434,out+650]);
  }
  add('final',[3349]);names.push('two-stream-output');return names;
}
/** Source-fixed actual new backing, not driver backing or physical fit. */
export function twoStreamPhaseDemand(phase){
  let weights=0,cpu=0,work=0;
  for(const t of phase.tensors??[]){
    if(!Number.isSafeInteger(t.size)||t.size<=0||![0,1].includes(t.dtype))throw Error('invalid tensor component');
    weights+=t.size*(t.dtype===1?2:1);cpu=Math.max(cpu,2*t.size+(t.dtype===1?2*t.size:0));
  }
  if(phase.name==='two-stream-embedding-weights')work=0;
  else if(phase.name==='two-stream-embedding-rearrange'){
    if(phase.tensors?.length)throw Error('rearrangement cannot reacquire embedding weights');
    work=tri+20;
  }
  else if(phase.name==='two-stream-output')return {components:[
    {name:'bounded-complete-output-readback',bytes:128*D*4,scope:'gpu'},
    {name:'complete-output-stream-response',bytes:2*128*D*4,scope:'cpu'}],
    weightGpuBytes:0,rangeCpuBytes:2*128*D*4,workGpuBytes:128*D*4,requiredBytes:3*128*D*4};
  else if(phase.name==='two-stream-duty'){
    const d=phase.duty;
    switch(d?.kind){
      case 'setup':work=3*tri+2*I*D*4+2*1792*D*4+latent+256+128*1024;break;
      case 'fuse-prepare':{
        if(!['in','out'].includes(d.direction))throw Error('exact resident fuse direction required');
        const q=d.direction==='in'?L:N,kv=d.direction==='in'?N:L;
        work=(q+kv)*D*4+attention(q,kv)+D*4+4096;break;
      }
      case 'basic-self-prepare':work=latent+attention(L,L)+4096;break;
      case 'basic-cross-prepare':work=3*latent+attention(L,I)+4096;break;
      case 'fuse-finish':work=5*latent+scratch(128)+4096;break;
      case 'basic-finish':work=5*latent+scratch(128)+4096;break;
      case 'fuse-attention-linear-range':work=d.rangeIndex===0?tri+4096:0;break;
      case 'fuse-residual-norm':work=2*tri+4096;break;
      case 'fuse-resident-ffn-range':work=d.rangeIndex===0?tri+scratch(128)+4096:0;break;
      case 'fuse-final-residual':work=tri+4096;break;
      case 'final':work=2*tri+4096;break;
      case 'attention-tile':break;
      default:throw Error('unknown or nonresident backbone duty');
    }
  }else if(!/^two-stream-(setup|final|block-[0-3]-(fuse-(in|out)|basic-[0-2]))$/.test(phase.name))
    throw Error('unknown resident two-stream phase');
  const components=[{name:'selected-weight-uploads',bytes:weights,scope:'gpu'},
    {name:'one-range-response-destination-conversion',bytes:cpu,scope:'cpu'},
    {name:'actual-new-work-and-uniform-backing',bytes:work,scope:'gpu'}];
  return {components,weightGpuBytes:weights,rangeCpuBytes:cpu,workGpuBytes:work,
    requiredBytes:weights+cpu+work,authority:'identified new backing; no opaque driver or physical reclamation guarantee'};
}

export function inspectTwoStreamOutputBytes(bytes){
  if(bytes.byteLength!==tri)throw Error('complete 3x1024x96x96 output required');
  let finite=0,nonzero=0;
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  for(let i=0;i<bytes.byteLength;i+=4){const v=view.getFloat32(i,true);if(Number.isFinite(v))finite++;if(v!==0)nonzero++;}
  if(finite!==tri/4||!nonzero)throw Error('nonfinite or blank complete backbone output');
  return {bytes:tri,finite,nonzero,sha256:createHash('sha256').update(bytes).digest('hex')};
}

export function inspectTwoStreamOutput(path){
  const fd=fs.openSync(path,'r'),hash=createHash('sha256'),chunk=Buffer.alloc(65536);
  let bytes=0,finite=0,nonzero=0,min=Infinity,max=-Infinity;
  try{
    if(fs.fstatSync(fd).size!==tri)throw Error('missing/partial complete backbone output');
    for(let count;(count=fs.readSync(fd,chunk,0,chunk.length,null))>0;){
      if(count%4)throw Error('misaligned backbone output');
      hash.update(chunk.subarray(0,count));bytes+=count;
      for(let i=0;i<count;i+=4){const v=chunk.readFloatLE(i);if(Number.isFinite(v))finite++;if(v!==0)nonzero++;min=Math.min(min,v);max=Math.max(max,v);}
    }
  }finally{fs.closeSync(fd);}
  if(finite!==tri/4||!nonzero)throw Error('nonfinite or blank complete backbone output');
  return {bytes,finite,nonzero,min,max,sha256:hash.digest('hex')};
}

export function acceptResidentTwoStream(report){
  const base=acceptResidentDino({...report,phaseObservations:report.phaseObservations?.slice(0,27)});
  const errors=[...base.errors],require=(test,message)=>{if(!test)errors.push(message);};
  const b=report.twoStream,c=b?.cooperative,boundary=c?.boundaries?.[0];
  require(b?.residentFFN===true&&b.linearRowsPerDuty===128&&b.blocks===4&&b.basicBlocks===12&&
    b.shape?.join(',')==='3,1024,96,96','complete effective resident backbone configuration missing');
  require(b?.weightPhases?.length===23&&b.weightPhases.every(p=>p.status==='completed-retired'),'complete embedding and22 weight groups missing');
  require(c?.status==='succeeded'&&c.routeId==='sf3d.image-to-mesh.webgpu-local.v0'&&
    c.manifestId==='sf3d.two-stream-attention-cooperative-boundaries.v0'&&c.schedulingMode==='cooperative'&&
    c.queueCompletionAuthority==='per-gpu-duty-prefix-fence'&&boundary?.boundaryId==='two-stream-attention-duties'&&
    boundary?.completedItems===3350&&boundary.totalItems===3350&&boundary.actualRangeCount===3350&&
    c.adapterTelemetry?.residentFFN===true&&c.adapterTelemetry.linearRowsPerDuty===128&&
    c.adapterTelemetry.dutyGranularity==='attention-tile'&&c.adapterTelemetry.declaredDutyCount===3350,'complete actual3350-duty cooperative backbone missing');
  require(!!b?.loadingReport?.sourceETag&&b.loadingReport.sourceETag===report.canonicalSource?.etag&&
    b.loadingReport.expectedWeightBytes===report.canonicalSource?.byteLength,'backbone effective immutable source drift');
  const expected=residentTwoStreamExpectedPhases().join(',');
  require(report.expectedPhaseOrder?.join(',')===expected&&
    report.phaseObservations?.map(o=>o.phase).join(',')===expected,'actual approved backbone cliff sequence incomplete');
  require(report.phaseObservations?.every(o=>o.verdict==='admitted'&&o.host?.source==='live-macos'&&
    o.host.hostFreeBytes>=o.demand.requiredBytes&&o.host.hostname===report.source?.hostname&&
    o.host.model==='Mac14,9'&&o.host.processor==='Apple M2 Pro'&&!o.host.observerErrors?.length&&
    o.process?.coverage==='sampled-owned-process-tree'&&o.process.lastObservation?.runId===report.runId&&
    o.process.lastObservation.rootPid===report.rootPid&&o.process.lastObservation.status==='observed'&&
    o.process.lastObservation.sampledAggregatePhysicalFootprintBytes+o.demand.requiredBytes<=report.requested?.processBudgetBytes),
    'fresh actual backbone cliffs not admitted');
  require(report.twoStreamReadbackBytes===tri,'complete streamed readback not persisted');
  try{require(inspectTwoStreamOutput(report.evidencePaths.twoStreamOutput).sha256===report.twoStreamOutput?.sha256,'raw backbone output identity mismatch');}
  catch(error){errors.push('backbone output: '+error.message);}
  return {ok:!errors.length,errors,authority:'complete native DINO plus two-stream backbone; no postprocessor/mesh/material/GLB or foreground/physical-fit claim'};
}
