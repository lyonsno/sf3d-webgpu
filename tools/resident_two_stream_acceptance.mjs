import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {acceptResidentDino} from './resident_dino_acceptance.mjs';
import {phaseHostHeadroomIsAdmitted} from './memory_admission.mjs';
const D=1024,N=27648,L=3089,I=1297,tri=N*D*4,latent=L*D*4;
const scratch=rows=>rows*(2*D+3*4096)*4;
const attention=(q,kv,rows)=>(2*q+2*kv)*D*4+16*rows*kv*4+rows*D*4;
/** Approved complete1297-token/128-row graph; independently checked against
 * the source planner by test_two_stream_review_contracts. Reporter input is
 * never the authority for a shorter sequence. */
export function residentTwoStreamExpectedPhases(attentionRowsPerDuty=128,rematerializeTokenizerEmbedding=false){
  if(typeof rematerializeTokenizerEmbedding!=='boolean')throw Error('exact boolean tokenizer rematerialization required');
  if(!Number.isSafeInteger(attentionRowsPerDuty)||attentionRowsPerDuty<=0)throw Error('exact positive attention row configuration required');
  const latentTiles=Math.ceil(L/attentionRowsPerDuty),triTiles=Math.ceil(N/attentionRowsPerDuty);
  const blockDuties=446+7*latentTiles+triTiles;
  const names=['preprocess','camera',...Array.from({length:24},(_,i)=>'dino-block-'+i),'dino-output',
    'two-stream-embedding-weights','two-stream-embedding-rearrange'];
  const add=(stage,indices)=>{names.push('two-stream-'+stage);
    for(const index of indices)names.push('two-stream-duty-'+index);};
  add('setup',[0]);
  for(let b=0;b<4;b++){
    const start=1+b*blockDuties;
    add('block-'+b+'-fuse-in',[start,start+1+latentTiles]);
    for(let basic=0;basic<3;basic++){
      const first=start+2+latentTiles+basic*(3+2*latentTiles);
      add('block-'+b+'-basic-'+basic,[first,first+1+latentTiles,first+2+2*latentTiles]);
    }
    const out=start+11+7*latentTiles;
    add('block-'+b+'-fuse-out',[out,out+1+triTiles,out+217+triTiles,out+218+triTiles,out+434+triTiles]);
  }
  if(rematerializeTokenizerEmbedding)names.push('two-stream-embedding-rematerialize-weights','two-stream-embedding-rematerialize-rearrange');
  add('final',[1+4*blockDuties]);names.push('two-stream-output');return names;
}
/** Source-fixed actual new backing, not driver backing or physical fit. */
export function twoStreamPhaseDemand(phase){
  let weights=0,cpu=0,work=0;
  for(const t of phase.tensors??[]){
    if(!Number.isSafeInteger(t.size)||t.size<=0||![0,1].includes(t.dtype))throw Error('invalid tensor component');
    weights+=t.size*(t.dtype===1?2:1);cpu=Math.max(cpu,2*t.size+(t.dtype===1?2*t.size:0));
  }
  if(phase.name==='two-stream-embedding-rematerialize-weights'){
    const t=phase.tensors?.[0];
    if(phase.tensors?.length!==1||t.name!=='tokenizer.embeddings'||t.dtype!==1||t.size!==56623104||
      t.shape?.join(',')!=='3,1024,96,96')throw Error('complete canonical tokenizer rematerialization weights required');
  }else if(phase.name==='two-stream-embedding-weights')work=0;
  else if(['two-stream-embedding-rearrange','two-stream-embedding-rematerialize-rearrange'].includes(phase.name)){
    if(phase.tensors?.length)throw Error('rearrangement cannot reacquire embedding weights');
    work=tri+20;
  }
  else if(phase.name==='two-stream-output')return {components:[
    {name:'bounded-complete-output-readback',bytes:128*D*4,scope:'gpu'},
    {name:'complete-output-stream-response',bytes:2*128*D*4,scope:'cpu'}],
    weightGpuBytes:0,rangeCpuBytes:2*128*D*4,workGpuBytes:128*D*4,requiredBytes:3*128*D*4};
  else if(phase.name==='two-stream-duty'){
    const d=phase.duty;
    const rows=phase.attentionRowsPerDuty??128,normX=phase.normX??true;
    const reuse=phase.reuseDeadTriplaneStorage??false;
    const residualReuse=phase.reuseAttentionResidualStorage??false;
    if(!Number.isSafeInteger(rows)||rows<=0||typeof normX!=='boolean'||typeof reuse!=='boolean'||
      typeof residualReuse!=='boolean'||(residualReuse&&!reuse))throw Error('exact attention row, optional normalization and storage-reuse configuration required');
    switch(d?.kind){
      case 'setup':work=3*tri+2*I*D*4+2*1792*D*4+latent+256+128*1024;break;
      case 'fuse-prepare':{
        if(!['in','out'].includes(d.direction))throw Error('exact resident fuse direction required');
        const q=d.direction==='in'?L:N,kv=d.direction==='in'?N:L;
        work=(q+(normX?kv:0))*D*4+attention(q,kv,rows)+D*4+4096;break;
      }
      case 'basic-self-prepare':work=latent+attention(L,L,rows)+4096;break;
      case 'basic-cross-prepare':work=3*latent+attention(L,I,rows)+4096;break;
      case 'fuse-finish':work=5*latent+scratch(128)+4096;break;
      case 'basic-finish':work=5*latent+scratch(128)+4096;break;
      case 'fuse-attention-linear-range':work=d.rangeIndex===0?tri+4096:0;break;
      case 'fuse-residual-norm':work=(residualReuse?1:2)*tri+4096;break;
      case 'fuse-resident-ffn-range':work=d.rangeIndex===0?(reuse?0:tri)+scratch(128)+4096:0;break;
      case 'fuse-final-residual':work=(reuse?0:tri)+4096;break;
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
  const rows=report.requested?.attentionRowsPerDuty;
  const rematerialize=report.requested?.rematerializeTokenizerEmbedding??false;
  require(typeof rematerialize==='boolean'&&(b?.rematerializeTokenizerEmbedding??false)===rematerialize,
    'effective tokenizer rematerialization differs from requested configuration');
  if(rematerialize){
    for(const [key,group]of [['setupTokenizerEmbeddingRetirement','setup'],['tokenizerEmbeddingRetirement','final']]){
      const r=b?.[key];require(r?.status==='retired'&&r.afterGroup===group&&r.bufferCount===2&&r.logicalBytes===tri+20,
        'complete logical tokenizer consumer retirement missing: '+group);
    }
    require([0,22].every(index=>b?.weightPhases?.[index]?.tensorNames?.join(',')==='tokenizer.embeddings'),
      'both actual complete tokenizer weight selections missing');
    for(const name of ['two-stream-embedding-rematerialize-weights','two-stream-embedding-rematerialize-rearrange']){
      const o=report.phaseObservations?.find(o=>o.phase===name);
      try{const demand=twoStreamPhaseDemand(o?.descriptor??{});
        require(o?.descriptor?.name===name&&['weightGpuBytes','rangeCpuBytes','workGpuBytes','requiredBytes'].every(k=>o.demand?.[k]===demand[k]),
          'effective complete tokenizer rematerialization demand missing or underdeclared');
      }catch(error){errors.push(error.message);}
    }
  }
  const reuse=report.requested?.reuseDeadTriplaneStorage??false;
  const residualReuse=report.requested?.reuseAttentionResidualStorage??false;
  require(typeof residualReuse==='boolean'&&(!residualReuse||reuse)&&
    (b?.reuseAttentionResidualStorage??false)===residualReuse&&
    (c?.adapterTelemetry?.reuseAttentionResidualStorage??false)===residualReuse,
    'effective attention residual storage reuse differs from requested configuration');
  if(residualReuse){
    const records=c?.adapterTelemetry?.attentionResidualReuse;
    const blockDuties=446+7*Math.ceil(L/rows)+Math.ceil(N/rows);
    require(records?.length===4&&records.every((r,block)=>r.block===block&&r.bytes===tri&&
      r.afterDutyIndex===1+block*blockDuties+11+7*Math.ceil(L/rows)+216+Math.ceil(N/rows)&&
      r.source==='owned-projected-attention-after-gpu-prefix'),'all four actual projected-attention residual transfers missing');
    const cliffs=Array.from({length:4},(_,block)=>'two-stream-duty-'+
      (1+block*blockDuties+11+7*Math.ceil(L/rows)+217+Math.ceil(N/rows)));
    const observed=report.phaseObservations?.filter(o=>cliffs.includes(o.phase));
    const expected=twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-residual-norm'},
      reuseDeadTriplaneStorage:true,reuseAttentionResidualStorage:true});
    require(observed?.length===4&&observed.every(o=>o.descriptor?.duty?.kind==='fuse-residual-norm'&&
      ['weightGpuBytes','rangeCpuBytes','workGpuBytes','requiredBytes'].every(k=>o.demand?.[k]===expected[k])),
      'actual reduced residual allocation demand missing or underdeclared');
    require(report.phaseObservations?.filter(o=>o.phase.startsWith('two-stream-duty-')).every(o=>
      o.descriptor?.reuseAttentionResidualStorage===true),'fresh duty demand lacks effective attention residual reuse identity');
  }
  require(typeof reuse==='boolean'&&(b?.reuseDeadTriplaneStorage??false)===reuse&&
    (c?.adapterTelemetry?.reuseDeadTriplaneStorage??false)===reuse,'effective owned-storage reuse differs from requested configuration');
  if(reuse){
    const records=c?.adapterTelemetry?.storageReuse;
    const blockDuties=446+7*Math.ceil(L/rows)+Math.ceil(N/rows);
    require(records?.length===4&&records.every((r,block)=>r.block===block&&r.bytes===tri&&
      r.afterDutyIndex===1+block*blockDuties+11+7*Math.ceil(L/rows)+217+Math.ceil(N/rows)&&
      r.source==='owned-work-inventory-after-gpu-prefix'),'all four actual owned-buffer prefix transfers missing');
    require(report.phaseObservations?.filter(o=>o.phase.startsWith('two-stream-duty-')).every(o=>
      o.descriptor?.reuseDeadTriplaneStorage===true),'fresh duty demand lacks effective storage-reuse identity');
  }
  const configured=Number.isSafeInteger(rows)&&rows>0;
  const duties=configured?2+4*(446+7*Math.ceil(L/rows)+Math.ceil(N/rows)):null;
  require(b?.residentFFN===true&&b.linearRowsPerDuty===128&&b.blocks===4&&b.basicBlocks===12&&
    b.shape?.join(',')==='3,1024,96,96'&&configured&&b.attentionRowsPerDuty===rows,'complete effective resident backbone configuration missing');
  require(b?.weightPhases?.length===(rematerialize?24:23)&&b.weightPhases.every(p=>p.status==='completed-retired'),'complete embedding and22 weight groups missing');
  require(c?.status==='succeeded'&&c.routeId==='sf3d.image-to-mesh.webgpu-local.v0'&&
    c.manifestId==='sf3d.two-stream-attention-cooperative-boundaries.v0'&&c.schedulingMode==='cooperative'&&
    c.queueCompletionAuthority==='per-gpu-duty-prefix-fence'&&boundary?.boundaryId==='two-stream-attention-duties'&&
    configured&&boundary?.completedItems===duties&&boundary.totalItems===duties&&boundary.actualRangeCount===duties&&
    c.adapterTelemetry?.residentFFN===true&&c.adapterTelemetry.linearRowsPerDuty===128&&
    c.adapterTelemetry.attentionRowsPerDuty===rows&&c.adapterTelemetry.dutyGranularity==='attention-tile'&&
    c.adapterTelemetry.declaredDutyCount===duties,'complete actual configured cooperative backbone missing');
  require(!!b?.loadingReport?.sourceETag&&b.loadingReport.sourceETag===report.canonicalSource?.etag&&
    b.loadingReport.expectedWeightBytes===report.canonicalSource?.byteLength,'backbone effective immutable source drift');
  const expected=configured&&typeof rematerialize==='boolean'?residentTwoStreamExpectedPhases(rows,rematerialize).join(','):null;
  require(configured&&report.expectedPhaseOrder?.join(',')===expected&&
    report.phaseObservations?.map(o=>o.phase).join(',')===expected,'actual approved backbone cliff sequence incomplete');
  require(report.phaseObservations?.every(o=>o.verdict==='admitted'&&o.host?.source==='live-macos'&&
    phaseHostHeadroomIsAdmitted(report,o)&&o.host.hostname===report.source?.hostname&&
    o.host.model==='Mac14,9'&&o.host.processor==='Apple M2 Pro'&&!o.host.observerErrors?.length&&
    o.process?.coverage==='sampled-owned-process-tree'&&o.process.lastObservation?.runId===report.runId&&
    o.process.lastObservation.rootPid===report.rootPid&&o.process.lastObservation.status==='observed'&&
    Number.isFinite(o.phaseRequestedAtUnixMs)&&o.process.freshness?.route==='new-probe-after-request'&&
    o.process.freshness.requestedAtUnixMs>=o.phaseRequestedAtUnixMs&&
    Number.isFinite(o.process.lastObservation.atUnixMs)&&o.process.lastObservation.atUnixMs>=o.process.freshness.requestedAtUnixMs&&
    o.process.freshness.probeAtUnixMs===o.process.lastObservation.atUnixMs&&
    Number.isSafeInteger(o.process.freshness.observationIndex)&&o.process.freshness.observationIndex>0&&
    o.process.freshness.observationIndex===o.process.sampleCount&&
    o.process.lastObservation.sampledAggregatePhysicalFootprintBytes+o.demand.requiredBytes<=report.requested?.processBudgetBytes),
    'fresh actual backbone cliffs not admitted');
  require(report.twoStreamReadbackBytes===tri,'complete streamed readback not persisted');
  try{require(inspectTwoStreamOutput(report.evidencePaths.twoStreamOutput).sha256===report.twoStreamOutput?.sha256,'raw backbone output identity mismatch');}
  catch(error){errors.push('backbone output: '+error.message);}
  return {ok:!errors.length,errors,authority:'complete native DINO plus two-stream backbone; no postprocessor/mesh/material/GLB or foreground/physical-fit claim'};
}
