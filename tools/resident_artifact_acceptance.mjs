import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {RESIDENT_ARTIFACT_CONFIG as C} from '../src/lib/resident_artifact.js';
import {describeResidentDecoderDemand} from '../src/lib/resident_decoder.js';
import {CLIP_PREP_WEIGHT_NAMES} from '../src/lib/clip_prep_core.js';
import {validateGlbPayload} from './full_route_benchmark_contract.mjs';
import {acceptNativeResidentPostProcessor,postProcessorExpectedPhases} from './resident_post_processor_acceptance.mjs';
import {residentTwoStreamExpectedPhases} from './resident_two_stream_acceptance.mjs';
import {phaseHostHeadroomIsAdmitted} from './memory_admission.mjs';

const exact=(ok,msg)=>{if(!ok)throw Error(msg);};
const integer=n=>Number.isSafeInteger(n)&&n>=0;
export function sourceIntakePhaseDemand(p,headerBytes) {
  const size=p.name.endsWith('weight-header-prefix')?16:headerBytes;
  exact(p.name.endsWith('weight-header-prefix')||p.name.endsWith('weight-header'),'unknown source intake');
  exact(p.sourceOffset===0&&p.sourceUnitBytes===size&&p.storageKind==='cpu-source'&&
    p.rangeCpuBytes===2*size&&p.requiredBytes===2*size&&p.workGpuBytes===0,'effective source intake backing mismatch');
  return {weightGpuBytes:0,workGpuBytes:0,rangeCpuBytes:2*size,requiredBytes:2*size,
    components:[{name:'header-destination-and-response',scope:'cpu',bytes:2*size}]};
}
const pair=name=>[name+'.weight',name+'.bias'];
const headNames=heads=>heads.flatMap(h=>Array.from({length:h==='density'||h==='vertex_offset'?3:4},(_,i)=>pair('decoder.heads.'+h+'.'+2*i)).flat());
function weightDemand(p,names,table,storageKind='gpu-buffer',workCpu=0) {
  exact(p.tensors?.map(t=>t.name).join(',')===names.join(','),'complete canonical selected weights/order missing: '+p.name);
  exact((p.storageKind??'gpu-buffer')===storageKind,'effective weight storage route changed');
  let expanded=0,range=0;
  for(const t of p.tensors){const observed=table.get(t.name);
    exact(observed&&['size','offset','dtype'].every(k=>observed[k]===t[k])&&
      observed.shape.join(',')===t.shape?.join(','),'effective canonical tensor metadata mismatch');
    expanded+=t.size*(t.dtype===1?2:1);
    range=Math.max(range,2*t.size+(storageKind==='gpu-buffer'&&t.dtype===1?2*t.size:0));
  }
  exact((p.workGpuBytes??0)===0&&(p.rangeCpuBytes??0)===workCpu,'effective selected weight work changed');
  const cpu=range+workCpu+(storageKind==='cpu-fp32'?expanded:0),gpu=storageKind==='gpu-buffer'?expanded:0;
  return {weightGpuBytes:gpu,workGpuBytes:0,rangeCpuBytes:cpu,requiredBytes:gpu+cpu,
    components:[{name:'selected-weight-backing',scope:storageKind==='cpu-fp32'?'cpu':'gpu',bytes:expanded},
      {name:'one-source-unit-and-conversion',scope:'cpu',bytes:range},{name:'actual-declared-cpu-work',scope:'cpu',bytes:workCpu}]};
}
const workDemand=(p,cpu,gpu=0)=>{
  exact(integer(cpu)&&integer(gpu)&&p.rangeCpuBytes===cpu&&p.workGpuBytes===gpu&&
    (p.requiredBytes===undefined||p.requiredBytes===cpu+gpu),'identified actual work backing mismatch: '+p.name);
  return {weightGpuBytes:0,rangeCpuBytes:cpu,workGpuBytes:gpu,requiredBytes:cpu+gpu,
    components:[{name:'explicit-cpu-work',scope:'cpu',bytes:cpu},{name:'explicit-gpu-work',scope:'gpu',bytes:gpu}]};
};

/** Strict complete route, with data-derived counts rather than hidden caps.
 * It validates the next expected phase before appending it; replayable raw
 * descriptors and canonical table permit a separate terminal re-run. */
export function createArtifactPhaseContract({table,headerBytes}) {
  const context={},steps=[],received=[];
  const add=(name,check)=>steps.push({name,check});
  const work=(name,cpu,gpu=0,check=()=>{})=>add(name,p=>{check(p);return workDemand(p,typeof cpu==='function'?cpu(p):cpu,typeof gpu==='function'?gpu(p):gpu);});
  const intake=scope=>{for(const name of ['weight-header-prefix','weight-header'])add(scope+'-'+name,p=>sourceIntakePhaseDemand(p,headerBytes));};
  const weights=(name,names,storage='gpu-buffer',cpu=0)=>add(name,p=>weightDemand(p,names,table,storage,cpu));
  const queries=(scope,heads,count)=>{
    weights(scope+'-decoder-selected-complete-heads',headNames(heads));
    work(scope+'-decoder-complete-output-allocation',()=>count()*heads.reduce((n,h)=>n+(h==='density'?1:3),0)*4,0,
      p=>exact(p.numPoints===count()&&p.heads?.join(',')===heads.join(','),'complete selected output shape changed'));
    // A lazy step appends all ranges only once the complete point count exists.
    add(scope+'-decoder-point-range',p=>{
      const n=count(),dummy={heads:Object.fromEntries(heads.map(h=>[h,Array(h==='density'||h==='vertex_offset'?3:4).fill({})]))};
      const demand=describeResidentDecoderDemand({numPoints:n,batchPoints:C.batchPoints,heads,weights:dummy});
      const range=(start,q)=>{const end=Math.min(n,start+demand.capacity);
        exact(q.kind==='decoder-point-range'&&q.start===start&&q.end===end&&q.numPoints===n&&q.heads?.join(',')===heads.join(','),'complete sequential decoder range changed');
        return workDemand(q,demand.rangeCpuBytes,start===0?demand.workGpuBytes:demand.components.uniformBytes);};
      const result=range(0,p),next=[];
      for(let start=demand.capacity;start<n;start+=demand.capacity)next.push({name:scope+'-decoder-point-range',check:q=>range(start,q)});
      steps.splice(1,0,...next);return result;
    });
  };
  for(const kind of ['grid','tets'])work('artifact-'+kind+'-source',2*(kind==='grid'?C.gridBytes:C.tetBytes),0,p=>{
    exact(p.sourceBytes===(kind==='grid'?C.gridBytes:C.tetBytes)&&p.sourceSha256===(kind==='grid'?C.gridSha256:C.tetSha256),'canonical tet asset identity changed');});
  work('artifact-grid-scale',C.gridBytes,0,p=>exact(p.gridPoints===C.gridPoints,'complete grid changed'));
  work('artifact-decoder-pipelines',0);
  intake('geometry-source');queries('geometry',['density','vertex_offset'],()=>C.gridPoints);
  work('artifact-threshold-sdf',4*C.gridPoints,0,p=>exact(p.gridPoints===C.gridPoints&&p.threshold===C.threshold,'threshold/grid changed'));
  work('marching-counted-scratch',p=>{
    exact(p.gridPoints===C.gridPoints&&integer(p.validTets)&&p.validTets<=C.tetrahedra&&
      integer(p.edgeCapacity)&&p.edgeCapacity>=3*p.validTets&&p.edgeCapacity<=4*p.validTets&&
      integer(p.numTriangles)&&p.numTriangles>=p.validTets&&p.numTriangles<=2*p.validTets,'complete counted crossing shape changed');
    let slots=1;while(slots<2*p.edgeCapacity)slots*=2;
    exact(p.hashSlots===slots,'actual complete crossing hash demand changed');context.scratch=p;
    return C.gridBytes+C.gridPoints+28*p.validTets+8*p.edgeCapacity+4*slots;
  });
  work('marching-complete-mesh',p=>12*p.numVertices+12*p.numFaces,0,p=>{
    exact(p.gridPoints===C.gridPoints&&integer(p.numVertices)&&p.numVertices>0&&p.numVertices<=context.scratch.edgeCapacity&&p.numFaces===context.scratch.numTriangles&&p.numFaces>0,'complete counted mesh missing');
    context.numVertices=p.numVertices;context.numFaces=p.numFaces;
  });
  work('artifact-geometry-persist',()=>24*(context.numVertices+context.numFaces));
  intake('materials');
  weights('materials-clip-preparation-weights',Object.values(CLIP_PREP_WEIGHT_NAMES),'cpu-fp32',512*512*4*4+3*224*224*4+50*768*4);
  work('materials-clip-pipeline-acquisition',0);
  const prefix='image_estimator.model.visual.';
  weights('materials-clip-pre-weights',pair(prefix+'ln_pre'));work('materials-clip-pre',0,2*50*768*4+16);
  for(let b=0;b<12;b++){
    const p=prefix+'transformer.resblocks.'+b;
    weights('materials-clip-block-'+b+'-weights',[...pair(p+'.ln_1'),...pair(p+'.ln_2'),p+'.attn.in_proj_weight',p+'.attn.in_proj_bias',...pair(p+'.attn.out_proj'),...pair(p+'.mlp.c_fc'),...pair(p+'.mlp.c_proj')]);
    work('materials-clip-block-'+b,0,18*50*768*4+160);
  }
  weights('materials-clip-post-weights',pair(prefix+'ln_post'));work('materials-clip-post',0,50*768*4+16);
  work('materials-clip-complete-token-readback',2*50*768*4,50*768*4);
  weights('materials-clip-visual-projection',[prefix+'proj'],'cpu-fp32',512*4);
  for(const h of ['roughness','metallic'])weights('materials-clip-'+h+'-head',
    ['0.0','0.2','0.4','1.0','1.2','2.0','2.2'].flatMap(l=>pair('image_estimator.heads.'+h+'.'+l)),'cpu-fp32',(512+3*512+2*(512+1))*4);
  work('artifact-uv-complete-typed-backing',()=>84*context.numVertices+141*context.numFaces+36,0,
    p=>exact(p.numVertices===context.numVertices&&p.numFaces===context.numFaces,'complete UV input changed'));
  work('artifact-full-raster',49*C.textureResolution*C.textureResolution,0,p=>exact(p.textureResolution===C.textureResolution,'texture resolution changed'));
  work('artifact-all-occupied-queries',p=>16*p.numOccupied,0,p=>{
    exact(p.textureResolution===C.textureResolution&&integer(p.numOccupied)&&p.numOccupied>0&&p.numOccupied<=C.textureResolution*C.textureResolution,'complete occupied texel count missing');context.numOccupied=p.numOccupied;});
  intake('bake-source');queries('bake',['features','perturb_normal'],()=>context.numOccupied);
  work('artifact-complete-textures-and-dilation',16*C.textureResolution*C.textureResolution,0,
    p=>exact(p.textureResolution===C.textureResolution&&p.numOccupied===context.numOccupied,'complete materialized texture shape changed'));
  work('glb-transforms',()=>84*context.numFaces); // UV duplicates3 vertices per face
  work('glb-albedo-image-data',4*C.textureResolution*C.textureResolution);
  work('glb-albedo-encoded',p=>{exact(integer(p.rangeCpuBytes)&&p.rangeCpuBytes>0,'actual encoded albedo bytes missing');context.albedoBytes=p.rangeCpuBytes;return p.rangeCpuBytes;});
  work('glb-normal-image-data',4*C.textureResolution*C.textureResolution);
  work('glb-normal-encoded',p=>{exact(integer(p.rangeCpuBytes)&&p.rangeCpuBytes>0,'actual encoded normal bytes missing');context.normalBytes=p.rangeCpuBytes;return p.rangeCpuBytes;});
  work('glb-json',p=>{exact(integer(p.rangeCpuBytes)&&p.rangeCpuBytes>0,'actual GLB JSON bytes missing');context.jsonBytes=p.rangeCpuBytes;return p.rangeCpuBytes;});
  work('glb-complete-buffer',p=>{const pad=n=>Math.ceil(n/4)*4;
    const bytes=28+pad(context.jsonBytes)+108*context.numFaces+pad(context.albedoBytes)+pad(context.normalBytes);
    context.glbBytes=bytes;return bytes;});
  work('artifact-glb-image-conformance',()=>2*Math.max(context.albedoBytes,context.normalBytes));
  // HTTP response/concatenation plus raw pre-UV geometry loaded for the
  // complete exporter join, and both streaming geometry scan buffers.
  work('artifact-glb-persist',()=>2*context.glbBytes+12*(context.numVertices+context.numFaces)+2*65536);
  return {context,received,get complete(){return steps.length===0;},accept(p){const step=steps[0];
    exact(step&&p.name===step.name,'effective complete artifact phase order mismatch: '+p.name+' expected '+step?.name);
    const demand=step.check(p);received.push({name:step.name,descriptor:p,demand});steps.shift();return demand;}};
}

/** Actual exporter contract, atop existing full GLB wire validator. */
export function inspectCompleteGlb(input,rawGeometry) {
  const bytes=typeof input==='string'?fs.readFileSync(input):input;
  const wire=validateGlbPayload(bytes);exact(wire.chunkCount===2,'complete GLB BIN missing');
  const jsonBytes=bytes.readUInt32LE(12),json=JSON.parse(bytes.subarray(20,20+jsonBytes).toString()),binStart=28+jsonBytes,binBytes=bytes.readUInt32LE(20+jsonBytes);
  exact(json.meshes?.length===1&&json.images?.length===2&&json.materials?.length===1&&json.buffers?.[0]?.byteLength===binBytes,'complete mesh/two learned textures/material missing');
  const primitive=json.meshes[0].primitives[0],material=json.materials[0];
  exact(json.scene===0&&json.scenes?.[0]?.nodes?.includes(0)&&json.nodes?.[0]?.mesh===0,'complete scene/node mesh connection missing');
  exact(primitive.material===0&&material.normalTexture?.index===1&&material.pbrMetallicRoughness?.baseColorTexture?.index===0,'complete actual textured material mapping missing');
  exact(json.textures?.[0]?.source===0&&json.textures?.[1]?.source===1,'actual material texture/image mapping missing or swapped');
  for(const key of ['roughnessFactor','metallicFactor'])exact(Number.isFinite(material.pbrMetallicRoughness[key])&&material.pbrMetallicRoughness[key]>=0&&material.pbrMetallicRoughness[key]<=1,'invalid actual material scalar');
  const get=(index,kind,width,componentType)=>{
    const a=json.accessors[index],v=json.bufferViews[a?.bufferView];
    exact(a&&a.type===kind&&a.componentType===componentType&&integer(a.count)&&a.count>0&&v?.buffer===0&&
      (v.byteOffset??0)+v.byteLength<=binBytes&&v.byteLength===a.count*width*4,'complete accessor shape/range mismatch');
    const payload=bytes.subarray(binStart+(v.byteOffset??0),binStart+(v.byteOffset??0)+v.byteLength);
    if(componentType===5126)for(let i=0;i<payload.length;i+=4)exact(Number.isFinite(payload.readFloatLE(i)),'nonfinite complete geometry/UV');
    return {count:a.count,payload};
  };
  const positions=get(primitive.attributes.POSITION,'VEC3',3,5126),normals=get(primitive.attributes.NORMAL,'VEC3',3,5126),uv=get(primitive.attributes.TEXCOORD_0,'VEC2',2,5126),indices=get(primitive.indices,'SCALAR',1,5125);
  exact(normals.count===positions.count&&uv.count===positions.count&&indices.count%3===0,'complete UV topology inconsistent');
  for(let i=0;i<indices.payload.length;i+=4)exact(indices.payload.readUInt32LE(i)<positions.count,'GLB face index outside actual complete mesh');
  const point=index=>[0,1,2].map(axis=>positions.payload.readFloatLE((3*index+axis)*4));
  let visibleTriangle=false;
  for(let i=0;i<indices.count;i+=3){
    const a=point(indices.payload.readUInt32LE(i*4)),b=point(indices.payload.readUInt32LE((i+1)*4)),c=point(indices.payload.readUInt32LE((i+2)*4));
    const u=b.map((x,j)=>x-a[j]),v=c.map((x,j)=>x-a[j]);
    if(u[1]*v[2]-u[2]*v[1]!==0||u[2]*v[0]-u[0]*v[2]!==0||u[0]*v[1]-u[1]*v[0]!==0)visibleTriangle=true;
  }
  exact(visibleTriangle,'blank/collapsed complete exported geometry');
  let geometryJoin;
  if(rawGeometry){
    const {meshInput,facesInput,numVertices,numFaces}=rawGeometry;
    inspectCompleteGeometry(meshInput,facesInput,numVertices,numFaces);
    const rawMesh=typeof meshInput==='string'?fs.readFileSync(meshInput):meshInput;
    const rawFaces=typeof facesInput==='string'?fs.readFileSync(facesInput):facesInput;
    exact(indices.count===3*numFaces&&positions.count===3*numFaces,'complete UV face duplication topology changed');
    for(let f=0;f<numFaces;f++)for(let corner=0;corner<3;corner++){
      const rawIndex=rawFaces.readUInt32LE((3*f+[0,2,1][corner])*4),p=point(indices.payload.readUInt32LE((3*f+corner)*4));
      const expected=[-rawMesh.readFloatLE((3*rawIndex+1)*4),rawMesh.readFloatLE((3*rawIndex+2)*4),-rawMesh.readFloatLE(3*rawIndex*4)];
      exact(p.every((x,axis)=>x===expected[axis]),'exported geometry detached from retained complete mesh transform/winding');
    }
    geometryJoin={source:'whole-retained-mesh/actual-export-transform-and-UV-duplication',facesMatched:numFaces,pointsMatched:3*numFaces};
  }
  const images=json.images.map(image=>{const v=json.bufferViews[image.bufferView];
    exact(image.mimeType==='image/jpeg'&&v?.buffer===0&&integer(v.byteLength)&&v.byteLength>0&&integer(v.byteOffset)&&v.byteOffset+v.byteLength<=binBytes,'complete embedded JPEG range missing');
    const payload=bytes.subarray(binStart+v.byteOffset,binStart+v.byteOffset+v.byteLength);
    exact(payload[0]===255&&payload[1]===216&&payload.at(-2)===255&&payload.at(-1)===217,'encoded texture is not a complete observed JPEG');
    return {bytes:payload.length,sha256:createHash('sha256').update(payload).digest('hex')};});
  return {bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex'),uvNumVertices:positions.count,uvNumFaces:indices.count/3,
    jsonBytes:Buffer.byteLength(JSON.stringify(json)),roughness:material.pbrMetallicRoughness.roughnessFactor,metallic:material.pbrMetallicRoughness.metallicFactor,images,geometryJoin};
}

/** Retain the actual attempted primary payload before judgment. Failed bytes
 * stay replayable at the same caller-owned path and never gain passed status. */
export async function persistArtifactDelivery({filename,bytes,report,key,persist,inspect}){
  fs.writeFileSync(filename,bytes,{flag:'wx'});
  const delivery={bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex'),validation:'unverified'};
  report[key]=delivery;await persist();
  try{Object.assign(delivery,await inspect(bytes),{validation:'passed'});}
  catch(error){delivery.validation='failed';delivery.error=String(error?.message??error);await persist();throw error;}
  await persist();return delivery;
}

/** Whole retained geometry, streaming file inputs rather than trusting report
 * decoration. This is the raw pre-UV mesh, not reference-math equivalence. */
export function inspectCompleteGeometry(meshInput,facesInput,numVertices,numFaces){
  exact(integer(numVertices)&&numVertices>0&&integer(numFaces)&&numFaces>0,'complete geometry counts missing');
  const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
  const scan=(input,bytes,inspect)=>{
    const hash=createHash('sha256');let offset=0;
    const accept=chunk=>{hash.update(chunk);for(let i=0;i<chunk.length;i+=4)inspect(chunk,i,(offset+i)/4);offset+=chunk.length;};
    if(typeof input==='string'){
      const fd=fs.openSync(input,'r'),chunk=Buffer.alloc(65536);
      try{exact(fs.fstatSync(fd).size===bytes,'partial complete geometry output');
        for(let n;(n=fs.readSync(fd,chunk,0,chunk.length,null))>0;)accept(chunk.subarray(0,n));}
      finally{fs.closeSync(fd);}
    }else{exact(input?.length===bytes,'partial complete geometry output');accept(input);}
    exact(offset===bytes,'partial complete geometry read');return {bytes,sha256:hash.digest('hex')};
  };
  const mesh=scan(meshInput,12*numVertices,(b,i,n)=>{const x=b.readFloatLE(i),axis=n%3;
    exact(Number.isFinite(x),'nonfinite complete geometry output');min[axis]=Math.min(min[axis],x);max[axis]=Math.max(max[axis],x);});
  exact(max.some((x,i)=>x>min[i]),'blank/default complete geometry output');
  const faces=scan(facesInput,12*numFaces,(b,i)=>exact(b.readUInt32LE(i)<numVertices,'complete geometry face index outside mesh'));
  return {mesh,faces,bounds:{min,max}};
}

export function acceptNativeArtifact(report,table) {
  const errors=[],require=(ok,msg)=>{if(!ok)errors.push(msg);};
  try {
    require(report.requested?.throughFullModel===true&&report.requested.throughPostProcessor===true&&report.requested.throughBackbone===true,'complete effective full-model request missing');
    require(report.requested.processBudgetBytes===2147483648&&report.requested.cpuBytes===256*1024*1024&&report.requested.gpuBytes===1024*1024*1024&&report.requested.totalBytes===1280*1024*1024,'effective guards changed');
    const prefix=[...residentTwoStreamExpectedPhases(report.requested.attentionRowsPerDuty,report.requested.rematerializeTokenizerEmbedding??false),...postProcessorExpectedPhases(report.requested.postChannelsPerDuty)];
    prefix.unshift('device-acquisition');
    for(const [before,scope]of [['camera','dino'],['two-stream-embedding-weights','two-stream'],['post-processor-output-allocation','post-processor']])
      prefix.splice(prefix.indexOf(before),0,scope+'-source-weight-header-prefix',scope+'-source-weight-header');
    require(report.artifactPrefixPhaseCount===prefix.length&&report.phaseObservations?.slice(0,prefix.length).map(o=>o.phase).join(',')===prefix.join(','),'effective fresh full prefix/source intake order missing');
    const base=report.phaseObservations?.filter(o=>o.phase!=='device-acquisition'&&!o.phase.startsWith('artifact-')&&!o.phase.startsWith('geometry-')&&!o.phase.startsWith('materials-')&&!o.phase.startsWith('bake-')&&!o.phase.startsWith('marching-')&&!o.phase.startsWith('glb-')&&!/^(dino|two-stream|post-processor)-source-/.test(o.phase));
    require(base?.length>0,'actual complete native prefix missing');
    if(base)errors.push(...acceptNativeResidentPostProcessor({...report,phaseObservations:base,expectedPhaseOrder:base.map(o=>o.phase)}).errors);
    const metadata=report.fullArtifact;
    require(metadata&&Object.entries(C).every(([k,v])=>metadata.config?.[k]===v),'effective canonical full-model config missing/changed');
    for(const [name,heads,points]of [['geometry',['density','vertex_offset'],C.gridPoints],['bake',['features','perturb_normal'],metadata?.numOccupied]]){
      const r=metadata?.[name];
      require(r?.numPoints===points&&r.heads?.join(',')===heads.join(',')&&r.batchPoints===C.batchPoints,'complete effective '+name+' query config missing');
      let cursor=0;for(const range of r?.ranges??[]){require(range.start===cursor&&range.end===Math.min(points,cursor+C.batchPoints)&&range.numPoints===points&&range.heads?.join(',')===heads.join(',')&&range.queueCompletionAuthority==='actual-prefix-before-range-reuse','complete '+name+' range/prefix drift');cursor=range.end;}
      require(cursor===points,'partial '+name+' output');
      require(r?.weightPhases?.length===1&&r.weightPhases[0].status==='completed-retired'&&r.weightPhases[0].tensorNames.join(',')===headNames(heads).join(','),'actual selected complete '+name+' weight phase missing');
      require(r?.loadingReport?.sourceETag===report.canonicalSource?.etag&&r?.loadingReport?.expectedWeightBytes===report.canonicalSource?.byteLength,'effective '+name+' source identity missing');
    }
    const m=metadata?.materials;
    require(m?.visualBackend==='webgpu-complete-12-blocks'&&m.tokens===50&&m.duties?.map(d=>d.name).join(',')===['clip-pre',...Array.from({length:12},(_,i)=>'clip-block-'+i),'clip-post'].join(',')&&m.duties.every(d=>d.queueCompletionAuthority==='actual-prefix-before-retirement'),'complete actual CLIP/prefix computation missing');
    require(m?.weightPhases?.length===18&&m.weightPhases.every(p=>p.status==='completed-retired')&&m.loadingReport?.sourceETag===report.canonicalSource?.etag&&m.loadingReport.expectedWeightBytes===report.canonicalSource?.byteLength,'actual material source/computation missing');
    const contract=createArtifactPhaseContract({table,headerBytes:report.canonicalSource?.headerBytes});
    for(const observation of report.phaseObservations??[]){
      const p=observation.descriptor,isArtifact=/^(artifact|geometry|materials|bake|marching|glb)-/.test(observation.phase);
      let demand;
      if(isArtifact)demand=contract.accept(p);
      else if(/^(dino|two-stream|post-processor)-source-/.test(observation.phase))demand=sourceIntakePhaseDemand(p,report.canonicalSource?.headerBytes);
      else if(observation.phase==='device-acquisition')demand=workDemand(p,0);
      else continue;
      require(['weightGpuBytes','rangeCpuBytes','workGpuBytes','requiredBytes'].every(k=>observation.demand?.[k]===demand[k]),'effective identified allocation demand changed');
      const process=observation.process;
      require(observation.verdict==='admitted'&&phaseHostHeadroomIsAdmitted(report,observation)&&observation.host?.source==='live-macos'&&
        observation.host.hostname===report.source?.hostname&&observation.host.model==='Mac14,9'&&observation.host.processor==='Apple M2 Pro'&&
        !observation.host.observerErrors?.length&&process?.coverage==='sampled-owned-process-tree'&&process.lastObservation?.runId===report.runId&&
        process.lastObservation.rootPid===report.rootPid&&process.lastObservation.status==='observed'&&process.freshness?.route==='new-probe-after-request'&&
        process.freshness.requestedAtUnixMs>=observation.phaseRequestedAtUnixMs&&process.lastObservation.atUnixMs>=process.freshness.requestedAtUnixMs&&
        process.freshness.probeAtUnixMs===process.lastObservation.atUnixMs&&process.freshness.observationIndex===process.sampleCount&&
        process.lastObservation.sampledAggregatePhysicalFootprintBytes+demand.requiredBytes<=2147483648,'fresh effective actual full-consumer guard join missing');
    }
    require(contract.complete,'actual artifact phase graph incomplete');
    require(report.expectedPhaseOrder?.join(',')===report.phaseObservations?.map(o=>o.phase).join(','),'terminal complete effective phase identity missing');
    const actual=inspectCompleteGlb(report.evidencePaths?.glb,{meshInput:report.evidencePaths?.mesh,facesInput:report.evidencePaths?.faces,numVertices:metadata?.numVertices,numFaces:metadata?.numFaces});
    const geometry=inspectCompleteGeometry(report.evidencePaths?.mesh,report.evidencePaths?.faces,metadata?.numVertices,metadata?.numFaces);
    require(['mesh','faces'].every((key,i)=>{const observed=i?report.faceOutput:report.meshOutput;
      return observed?.bytes===geometry[key].bytes&&observed.sha256===geometry[key].sha256;}),'raw complete geometry identity missing/mismatch');
    require(report.tetAssets?.length===2&&['grid','tets'].every(kind=>{
      const asset=report.tetAssets.find(a=>a.url===(kind==='grid'?'/tets/_grid_vertices.bin':'/tets/indices.bin'));
      return asset?.bytes===(kind==='grid'?C.gridBytes:C.tetBytes)&&asset.sha256===(kind==='grid'?C.gridSha256:C.tetSha256)&&asset.sourceRevision===report.source?.revision;
    }),'observed complete canonical tet/source revision identity missing');
    require(actual.bytes===metadata?.glbBytes&&actual.uvNumFaces===metadata?.numFaces&&actual.uvNumFaces===metadata?.uvNumFaces&&actual.uvNumVertices===metadata?.uvNumVertices&&
      actual.roughness===m?.roughness&&actual.metallic===m?.metallic,'actual complete GLB differs from computed mesh/material');
    require(actual.sha256===report.glbOutput?.sha256&&actual.bytes===contract.context.glbBytes&&actual.jsonBytes===contract.context.jsonBytes&&
      actual.images[0].bytes===contract.context.albedoBytes&&actual.images[1].bytes===contract.context.normalBytes,'raw GLB/export allocation identity mismatch');
    require([report.glbOutput,report.meshOutput,report.faceOutput].every(output=>output?.validation==='passed'),'delivered artifact validation not completed');
    require(report.decodedGlbImages?.length===2&&report.decodedGlbImages.every((image,i)=>image.source==='live-browser-decoded-embedded-JPEG'&&image.width===C.textureResolution&&image.height===C.textureResolution&&image.sha256===actual.images[i].sha256),'actual browser JPEG dimension/content conformance missing');
  } catch(error){errors.push(error.message);}
  return {ok:!errors.length,errors,authority:'complete source-bound native canonical textured GLB under identified guards; no reference parity, opaque physical fit or living-foreground claim'};
}
