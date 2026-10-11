import {isLoaderMemoryBudget} from './loader_memory_budget.js';
import {runResidentDecoder} from './resident_decoder.js';
import {TriplaneDecoder} from './triplane_decoder.js';
import {runResidentMaterials} from './resident_clip.js';
import {runResidentMarchingTetrahedra,disposeResidentMarchingTetrahedra,scaleTensor} from './marching_tet.js';
import {unwrapResidentUV,rasterizeUV,exportGLB} from './texture_baker.js';
import {materializeTextures,createDilationScratch} from './materialize_core.js';
import {withForegroundScope} from './foreground_scope.js';

export const RESIDENT_ARTIFACT_CONFIG=Object.freeze({gridPoints:535882,tetrahedra:2971452,
  gridResolution:160,radius:0.87,threshold:10,textureResolution:1024,batchPoints:4096,
  gridBytes:6430584,tetBytes:47543232,
  gridSha256:'16f4ee01a050d1757c19b13a7e1dbd4d0918d08208d961c105ff74cbfc345dac',
  tetSha256:'606cc8b47f8744a64ff6f1f3c088d9d9113ff80539bd62cb1651f8dc629d1f1e'});
const hex=bytes=>Array.from(new Uint8Array(bytes),b=>b.toString(16).padStart(2,'0')).join('');

/** Complete canonical consumer of BORROWED live postprocessor planes.
 * No eager source, sampled grid, material defaults or escaping uncharged arrays.
 * The final consumer is awaited while every output lease is still owned.
 * Explicit backing accounting excludes DOM/canvas/JS metadata and driver memory;
 * a caller's fresh process/host guard remains mandatory, not a capacity claim. */
export async function runResidentArtifact({device,memoryBudget,triplanesBuf,conditionRgba,
  weightsUrl,expectedWeightBytes,expectedSourceETag,onBeforePhase,onBeforeDuty,
  onAfterDuty,onGeometry,withResult,withForeground,onProgress}) {
  if(!isLoaderMemoryBudget(memoryBudget))throw TypeError('authenticated loader budget required');
  memoryBudget.assertDeviceAcquiredHere(device);
  if(triplanesBuf?.size!==70778880||!(conditionRgba instanceof Uint8Array)||conditionRgba.length!==512*512*4)
    throw TypeError('complete borrowed postprocessor and actual condition RGBA required');
  if(typeof onBeforePhase!=='function'||typeof onBeforeDuty!=='function'||typeof withResult!=='function')
    throw TypeError('fresh allocation/range observers and awaited actual artifact consumer required');
  const c=RESIDENT_ARTIFACT_CONFIG,leases=[],meshHandle={},spans=[],guardPhases=[];
  let gridRaw,tets,positions,sdf,mesh,uv,raster,queryPositions,occupied,texture,dilationScratch;
  const reserve=(bytes,label)=>{const lease=memoryBudget.reserveCpu(bytes,label);leases.push(lease);return lease;};
  const guard=async(name,bytes,details={})=>{
    const phase={name,tensors:[],workGpuBytes:0,rangeCpuBytes:bytes,requiredBytes:bytes,...details};
    await onBeforePhase(phase);guardPhases.push(phase);return reserve(bytes,name);
  };
  const timed=async(name,work)=>{onProgress?.(name);const start=performance.now();
    const result=await work();spans.push({name,start,end:performance.now()});return result;};
  const cpu=(name,work)=>timed(name,()=>withForegroundScope({withForeground},name,work));
  const scoped=scope=>async phase=>{const descriptor={...phase,name:scope+'-'+phase.name};
    await onBeforePhase(descriptor);guardPhases.push(descriptor);};
  const dutyScoped=scope=>async duty=>{const descriptor={...duty,name:scope+'-'+(duty.name??duty.kind),scope};
    await onBeforeDuty(descriptor);guardPhases.push(descriptor);};
  const load=async(name,url,bytes,sha)=>{
    await onBeforePhase({name,tensors:[],workGpuBytes:0,rangeCpuBytes:2*bytes,
      requiredBytes:2*bytes,sourceBytes:bytes,sourceSha256:sha});
    const retained=reserve(bytes,name+'-retained'),transport=reserve(bytes,name+'-transport');
    try{const response=await fetch(url,{cache:'no-store'});
      if(!response.ok||Number(response.headers.get('Content-Length'))!==bytes)throw Error('complete canonical asset response required: '+name);
      const buffer=await response.arrayBuffer();
      if(buffer.byteLength!==bytes||hex(await crypto.subtle.digest('SHA-256',buffer))!==sha)
        throw Error('canonical asset content identity mismatch: '+name);
      return {buffer,lease:retained};
    }finally{transport.release();}
  };
  let geometry,materials,bake;
  try {
    const grid=await load('artifact-grid-source','/tets/_grid_vertices.bin',c.gridBytes,c.gridSha256);
    gridRaw=new Float32Array(grid.buffer);
    const indices=await load('artifact-tets-source','/tets/indices.bin',c.tetBytes,c.tetSha256);
    tets=new Int32Array(indices.buffer);
    const scaledLease=await guard('artifact-grid-scale',c.gridBytes,{gridPoints:c.gridPoints});
    positions=scaleTensor(gridRaw,[0,1],[-c.radius,c.radius]);gridRaw=null;
    // The local grid object retains its borrowed buffer until this scope ends;
    // keep its source lease, rather than claiming GC by retiring it early.
    await onBeforePhase({name:'artifact-decoder-pipelines',tensors:[],workGpuBytes:0,rangeCpuBytes:0,requiredBytes:0});
    const decoder=new TriplaneDecoder(device);decoder.init();
    const args={device,decoder,memoryBudget,weightsUrl,expectedWeightBytes,expectedSourceETag,
      triplanesBuf,batchPoints:c.batchPoints,onAfterDuty};
    geometry=await timed('complete-geometry',()=>runResidentDecoder({...args,positions,heads:['density','vertex_offset'],
      onBeforeSourceIntake:scoped('geometry-source'),onBeforePhase:scoped('geometry'),onBeforeDuty:dutyScoped('geometry'),
      async withResult(output){
        const sdfLease=await guard('artifact-threshold-sdf',4*c.gridPoints,{gridPoints:c.gridPoints,threshold:c.threshold});
        sdf=new Float32Array(output.density);for(let i=0;i<sdf.length;i++)sdf[i]-=c.threshold;
        mesh=await cpu('complete-marching',()=>runResidentMarchingTetrahedra({handle:meshHandle,memoryBudget,
          gridVertices:positions,sdf,tetIndices:tets,vertexOffsets:output.vertex_offset,resolution:c.gridResolution,
          onBeforePhase}));
        sdf=null;sdfLease.release();
        if(!mesh.numVertices||!mesh.numFaces)throw Error('actual full geometry is empty; no blank/default artifact');
        await onGeometry?.(mesh);
        return {numVertices:mesh.numVertices,numFaces:mesh.numFaces};
      }}));
    positions=null;tets=null;scaledLease.release();
    // Canonical source buffers remain charged until this function completes.
    materials=await timed('actual-clip-materials',()=>runResidentMaterials({device,handle:{device},memoryBudget,
      weightsUrl,expectedWeightBytes,expectedSourceETag,rgba:conditionRgba,
      onBeforePhase:scoped('materials'),onBeforeDuty:dutyScoped('materials'),onAfterDuty}));
    const uvBytes=84*mesh.numVertices+141*mesh.numFaces+36;
    await guard('artifact-uv-complete-typed-backing',uvBytes,{numVertices:mesh.numVertices,numFaces:mesh.numFaces,
      backingAuthority:'all explicit typed UV allocations; BVH/JS metadata excluded, no physical capacity claim'});
    uv=await cpu('complete-uv',()=>unwrapResidentUV(mesh.vertices,mesh.faces,mesh.numVertices,mesh.numFaces));
    // Retain the original mesh through final consumer, not just local UV use.
    await guard('artifact-full-raster',49*c.textureResolution*c.textureResolution,{textureResolution:c.textureResolution});
    raster=await cpu('complete-raster',()=>rasterizeUV(uv.uvs,uv.newVertices,uv.newFaces,uv.newNumFaces,c.textureResolution,uv.faceAssignment));
    let numOccupied=0;for(const bit of raster.mask)if(bit)numOccupied++;
    if(!numOccupied)throw Error('complete raster has no occupied texels; no empty texture fallback');
    await guard('artifact-all-occupied-queries',16*numOccupied,{numOccupied,textureResolution:c.textureResolution});
    occupied=new Uint32Array(numOccupied);queryPositions=new Float32Array(numOccupied*3);
    let j=0;for(let i=0;i<raster.mask.length;i++)if(raster.mask[i]){occupied[j]=i;
      queryPositions.set(raster.positions3D.subarray(i*3,i*3+3),j*3);j++;}
    bake=await timed('complete-texture-queries',()=>runResidentDecoder({...args,positions:queryPositions,
      heads:['features','perturb_normal'],onBeforeSourceIntake:scoped('bake-source'),
      onBeforePhase:scoped('bake'),onBeforeDuty:dutyScoped('bake'),
      async withResult(output){
        await guard('artifact-complete-textures-and-dilation',16*c.textureResolution*c.textureResolution,
          {textureResolution:c.textureResolution,numOccupied});
        dilationScratch=createDilationScratch(c.textureResolution*c.textureResolution);
        texture=await cpu('complete-texture-materialize',()=>materializeTextures({featuresCPU:output.features,
          normalsCPU:output.perturb_normal,occupiedIndices:occupied,tbnData:raster.tbnData,
          mask:raster.mask,resolution:c.textureResolution,numOccupied,dilationScratch}));
        return {numOccupied,albedoBytes:texture.albedo.byteLength,normalBytes:texture.normalMap.byteLength};
      }}));
    const glb=await cpu('actual-glb-export',()=>exportGLB(uv.newVertices,uv.newNormals,uv.newFaces,uv.uvs,
      texture.albedo,texture.normalMap,uv.newNumVertices,uv.newNumFaces,c.textureResolution,
      materials.roughness,materials.metallic,{requireNormalTexture:true,
        onBeforeCpuAllocation:(name,bytes)=>guard(name,bytes)}));
    const metadata={config:c,numVertices:mesh.numVertices,numFaces:mesh.numFaces,
      uvNumVertices:uv.newNumVertices,uvNumFaces:uv.newNumFaces,numOccupied,glbBytes:glb.byteLength,
      geometry,materials,bake,spans,guardPhases,
      authority:'actual complete canonical consumer; output/foreground/native evidence requires caller witness; identified backing excludes opaque JS/canvas/driver allocations'};
    await withResult({glb,mesh,uv,texture,metadata});
    return metadata;
  } finally {
    gridRaw=null;tets=null;positions=null;sdf=null;mesh=null;uv=null;raster=null;
    queryPositions=null;occupied=null;texture=null;dilationScratch=null;
    disposeResidentMarchingTetrahedra(meshHandle);
    for(const lease of leases)lease.release();
  }
}
