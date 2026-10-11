import assert from 'node:assert/strict';
import {weightFixture} from './fixtures/weight_resource_fixture.mjs';
import {parseFlatWeightHeader} from '../src/lib/flat_tensor_ranges.js';
import {RESIDENT_ARTIFACT_CONFIG as C} from '../src/lib/resident_artifact.js';
import {exportGLB} from '../src/lib/texture_baker.js';
const subject=await import('./resident_artifact_acceptance.mjs').catch(e=>{if(e.code==='ERR_MODULE_NOT_FOUND'&&e.url===new URL('./resident_artifact_acceptance.mjs',import.meta.url).href)return {};throw e;});
assert.equal(typeof subject.createArtifactPhaseContract,'function','complete output needs source-bound phase/range witness, not a report status');
assert.equal(typeof subject.inspectCompleteGlb,'function','primary output must be parsed and checked, not just nonblank');
assert.equal(typeof subject.acceptNativeArtifact,'function','route/freshness/output/cleanup joins cannot close on defaults');
assert.equal(subject.acceptNativeArtifact({status:'passed',requested:{throughFullModel:true}}).ok,false);
for(const bytes of [Buffer.alloc(0),Buffer.alloc(48),Buffer.from('cached partial output')])assert.throws(()=>subject.inspectCompleteGlb(bytes));
const table=new Map(),contract=subject.createArtifactPhaseContract({table,headerBytes:202416});
assert.throws(()=>contract.accept({name:'artifact-full-raster',rangeCpuBytes:49*1024*1024}),/phase order/);
assert.equal(contract.complete,false);
assert.throws(()=>subject.sourceIntakePhaseDemand({sourceUnitBytes:16,sourceOffset:0,storageKind:'cpu-source',rangeCpuBytes:31,workGpuBytes:0,requiredBytes:31,name:'dino-source-weight-header-prefix'},202416),/source intake/);
assert.equal(subject.sourceIntakePhaseDemand({sourceUnitBytes:16,sourceOffset:0,storageKind:'cpu-source',rangeCpuBytes:32,workGpuBytes:0,requiredBytes:32,name:'dino-source-weight-header-prefix'},202416).requiredBytes,32);
// Complete constructive local graph, synthetic tensor metadata only: not a
// native/external contract witness. Point count is complete, not first-N.
const fixture=weightFixture({fullDecoder:true,fullMaterialNames:true}),headerBytes=new DataView(fixture.bytes.buffer).getUint32(12,true);
const tensorTable=parseFlatWeightHeader(fixture.bytes.buffer.slice(0,headerBytes),fixture.bytes.length).tensors;
const phases=[],pair=name=>[name+'.weight',name+'.bias'];
const work=(name,cpu,gpu=0,details={})=>phases.push({name,tensors:[],rangeCpuBytes:cpu,workGpuBytes:gpu,requiredBytes:cpu+gpu,...details});
const intake=scope=>{for(const [suffix,n]of [['weight-header-prefix',16],['weight-header',headerBytes]])
  work(scope+'-'+suffix,2*n,0,{sourceUnitBytes:n,sourceOffset:0,storageKind:'cpu-source'});};
const weights=(name,names,storageKind='gpu-buffer',rangeCpuBytes=0)=>phases.push({name,tensors:names.map(n=>({...tensorTable.get(n),name:n})),storageKind,rangeCpuBytes,workGpuBytes:0});
const query=(scope,heads,n)=>{
  weights(scope+'-decoder-selected-complete-heads',heads.flatMap(h=>Array.from({length:h==='density'||h==='vertex_offset'?3:4},(_,i)=>pair('decoder.heads.'+h+'.'+2*i)).flat()));
  work(scope+'-decoder-complete-output-allocation',n*(scope==='geometry'?16:24),0,{numPoints:n,heads});
  const capacity=Math.min(n,4096),uniform=(scope==='geometry'?356:428)*(n%capacity?2:1);
  for(let start=0;start<n;start+=capacity)work(scope+'-decoder-point-range',24*capacity,
    start===0?(scope==='geometry'?3076:4128)*capacity+(scope==='geometry'?16:24)+uniform+12*capacity:uniform,
    {kind:'decoder-point-range',start,end:Math.min(n,start+capacity),numPoints:n,heads});
};
work('artifact-grid-source',2*C.gridBytes,0,{sourceBytes:C.gridBytes,sourceSha256:C.gridSha256});
work('artifact-tets-source',2*C.tetBytes,0,{sourceBytes:C.tetBytes,sourceSha256:C.tetSha256});
work('artifact-grid-scale',C.gridBytes,0,{gridPoints:C.gridPoints});work('artifact-decoder-pipelines',0);
intake('geometry-source');query('geometry',['density','vertex_offset'],C.gridPoints);
work('artifact-threshold-sdf',4*C.gridPoints,0,{gridPoints:C.gridPoints,threshold:10});
work('marching-counted-scratch',C.gridBytes+C.gridPoints+28+24+32,0,{gridPoints:C.gridPoints,validTets:1,edgeCapacity:3,numTriangles:1,hashSlots:8});
work('marching-complete-mesh',48,0,{gridPoints:C.gridPoints,numVertices:3,numFaces:1});work('artifact-geometry-persist',96);
intake('materials');const prefix='image_estimator.model.visual.';
weights('materials-clip-preparation-weights',[prefix+'conv1.weight',prefix+'class_embedding',prefix+'positional_embedding'],'cpu-fp32',512*512*16+3*224*224*4+50*768*4);
work('materials-clip-pipeline-acquisition',0);weights('materials-clip-pre-weights',pair(prefix+'ln_pre'));work('materials-clip-pre',0,2*50*768*4+16);
for(let b=0;b<12;b++){const p=prefix+'transformer.resblocks.'+b;
  weights('materials-clip-block-'+b+'-weights',[...pair(p+'.ln_1'),...pair(p+'.ln_2'),p+'.attn.in_proj_weight',p+'.attn.in_proj_bias',...pair(p+'.attn.out_proj'),...pair(p+'.mlp.c_fc'),...pair(p+'.mlp.c_proj')]);
  work('materials-clip-block-'+b,0,18*50*768*4+160);}
weights('materials-clip-post-weights',pair(prefix+'ln_post'));work('materials-clip-post',0,50*768*4+16);work('materials-clip-complete-token-readback',2*50*768*4,50*768*4);
weights('materials-clip-visual-projection',[prefix+'proj'],'cpu-fp32',2048);
for(const h of ['roughness','metallic'])weights('materials-clip-'+h+'-head',['0.0','0.2','0.4','1.0','1.2','2.0','2.2'].flatMap(l=>pair('image_estimator.heads.'+h+'.'+l)),'cpu-fp32',12296);
work('artifact-uv-complete-typed-backing',84*3+141+36,0,{numVertices:3,numFaces:1});
work('artifact-full-raster',49*1024*1024,0,{textureResolution:1024});
work('artifact-all-occupied-queries',16*8193,0,{textureResolution:1024,numOccupied:8193});
intake('bake-source');query('bake',['features','perturb_normal'],8193);
work('artifact-complete-textures-and-dilation',16*1024*1024,0,{textureResolution:1024,numOccupied:8193});
work('glb-transforms',84);work('glb-albedo-image-data',4*1024*1024);work('glb-albedo-encoded',4);
work('glb-normal-image-data',4*1024*1024);work('glb-normal-encoded',4);work('glb-json',100);
work('glb-complete-buffer',28+100+108+8);work('artifact-glb-image-conformance',8);work('artifact-glb-persist',2*(28+100+108+8));
const replay=list=>{const c=subject.createArtifactPhaseContract({table:tensorTable,headerBytes});for(const p of list)c.accept(p);return c;};
const complete=replay(phases);assert.equal(complete.complete,true);
assert.equal(complete.received.filter(p=>p.name==='geometry-decoder-point-range').length,131);
assert.equal(complete.received.filter(p=>p.name==='bake-decoder-point-range').length,3);
assert.equal(replay(phases.slice(0,-1)).complete,false,'missing terminal primary-output allocation cannot close');
for(const name of ['geometry-decoder-point-range','materials-clip-block-11','artifact-full-raster','bake-decoder-point-range','glb-normal-encoded']){
  const index=phases.findIndex(p=>p.name===name),changed=structuredClone(phases);changed[index].rangeCpuBytes--;
  assert.throws(()=>replay(changed),/backing mismatch/,'underdeclared '+name+' cannot close');
  assert.throws(()=>replay(phases.filter((_,i)=>i!==index)),/range changed|phase order/,'skipped '+name+' cannot close');
}
for(const change of [p=>p.tensors.pop(),p=>p.tensors[0].offset++,p=>p.storageKind='cpu-fp32']){
  const changed=structuredClone(phases);change(changed.find(p=>p.name==='geometry-decoder-selected-complete-heads'));assert.throws(()=>replay(changed));}
const tail=structuredClone(phases);tail.findLast(p=>p.name==='geometry-decoder-point-range').end--;assert.throws(()=>replay(tail),/range changed/);
// Exercise real exporter structure with synthetic JPEG bytes, not browser
// encoding authority; native run must additionally decode both embedded JPEGs.
const oldDocument=globalThis.document;
globalThis.document={createElement(){return {getContext(){return {createImageData(w,h){return {data:new Uint8ClampedArray(w*h*4)};},putImageData(){}};},toBlob(done){done(new Blob([new Uint8Array([255,216,255,217])],{type:'image/jpeg'}));}};}};
let glb;
try{glb=Buffer.from(await exportGLB(new Float32Array([0,0,0,1,0,0,0,1,0]),new Float32Array([0,0,1,0,0,1,0,0,1]),new Uint32Array([0,1,2]),new Float32Array([0,0,1,0,0,1]),new Uint8Array(16),new Uint8Array(16),3,1,2,.7,.2,{requireNormalTexture:true}));}
finally{globalThis.document=oldDocument;}
const inspected=subject.inspectCompleteGlb(glb);assert.equal(inspected.uvNumFaces,1);assert.equal(inspected.images.length,2);assert.equal(inspected.roughness,.7);
const jsonLength=glb.readUInt32LE(12),binStart=28+jsonLength,json=JSON.parse(glb.subarray(20,20+jsonLength).toString());
for(const [offset,value]of [[json.bufferViews[0].byteOffset,NaN],[json.bufferViews[json.accessors[json.meshes[0].primitives[0].indices].bufferView].byteOffset,99]]){
  const bad=Buffer.from(glb);if(Number.isNaN(value))bad.writeFloatLE(value,binStart+offset);else bad.writeUInt32LE(value,binStart+offset);assert.throws(()=>subject.inspectCompleteGlb(bad));}
const blank=Buffer.from(glb);blank[binStart+json.bufferViews[json.images[0].bufferView].byteOffset]=0;assert.throws(()=>subject.inspectCompleteGlb(blank),/JPEG/);
assert.equal(typeof subject.inspectCompleteGeometry,'function','terminal raw canonical mesh/faces must be checked independently, not optional report decoration');
const geometry=subject.inspectCompleteGeometry(Buffer.from(new Float32Array([0,0,0,1,0,0,0,1,0]).buffer),Buffer.from(new Uint32Array([0,1,2]).buffer),3,1);
assert.equal(geometry.mesh.bytes,36);assert.equal(geometry.faces.bytes,12);
assert.throws(()=>subject.inspectCompleteGeometry(Buffer.alloc(0),Buffer.alloc(12),3,1),/partial/);
assert.throws(()=>subject.inspectCompleteGeometry(Buffer.from(new Float32Array([0,0,0,1,0,0,0,1,0]).buffer),Buffer.from(new Uint32Array([0,1,3]).buffer),3,1),/index/);
assert.throws(()=>subject.inspectCompleteGeometry(Buffer.alloc(36),Buffer.alloc(12),3,1),/blank/);
console.log('PASS complete local phase graph/all geometry ranges/tails, selected source, strict work demands, genuine GLB structure and raw geometry; synthetic policy/JPEG only');
