import {createWeightPhaseSource} from './weights.js';
import {isLoaderMemoryBudget} from './loader_memory_budget.js';
import {captureGpuBufferAllocations,createEmptyBuffer} from './gpu.js';
import {prepareClipEmbeddings,validateClipEmbeddings,CLIP_PREP_WEIGHT_NAMES} from './clip_prep_core.js';
import {createClipResidentPipelines,encodeClipVisualPhase,projectClipFeatures,runClipMaterialHead} from './clip_estimator.js';

const prefix='image_estimator.model.visual.',tokenBytes=50*768*4;
const pair=(source,name)=>({weight:source.reference(name+'.weight'),bias:source.reference(name+'.bias')});
export function selectResidentClipPhase(source,phase,index){
  if(phase==='prep')return Object.fromEntries(Object.entries(CLIP_PREP_WEIGHT_NAMES).map(([k,n])=>[k,source.reference(n)]));
  if(phase==='pre'||phase==='post')return pair(source,prefix+'ln_'+phase);
  if(phase==='projection')return {projection:source.reference(prefix+'proj')};
  if(phase==='block'){
    if(!Number.isSafeInteger(index)||index<0||index>=12)throw TypeError('complete twelve-block CLIP index required');
    const p=prefix+'transformer.resblocks.'+index;
    return {ln1:pair(source,p+'.ln_1'),ln2:pair(source,p+'.ln_2'),
      qkv:{weight:source.reference(p+'.attn.in_proj_weight'),bias:source.reference(p+'.attn.in_proj_bias')},
      outProj:pair(source,p+'.attn.out_proj'),fc:pair(source,p+'.mlp.c_fc'),proj:pair(source,p+'.mlp.c_proj')};
  }
  if(phase==='roughness'||phase==='metallic'){
    const p='image_estimator.heads.'+phase,result={};
    for(const l of ['0.0','0.2','0.4','1.0','1.2','2.0','2.2'])
      for(const s of ['weight','bias']){const name=p+'.'+l+'.'+s;result[name]=source.reference(name);}
    for(const t of source.describe(result)){
      const output=/\.[12]\.2\./.test(t.name)?1:512;
      const shape=t.name.endsWith('.bias')?[output]:[512,output];
      if(t.shape?.join(',')!==shape.join(','))throw Error('canonical material head shape mismatch: '+t.name);
    }
    return result;
  }
  throw TypeError('canonical complete CLIP phase required');
}

function retireWork(owner,keep=null){
  const errors=[];
  for(const a of [...owner.allocations])if(a.buffer!==keep){
    try{a.buffer.destroy();owner.allocations.splice(owner.allocations.indexOf(a),1);}
    catch(e){errors.push(e);}
  }
  if(errors.length)throw new AggregateError(errors,'CLIP owned retirement unresolved');
}

/** Actual complete GPU CLIP/CPU heads, no eager raw binding or default material. */
export async function runResidentMaterials({device,handle,memoryBudget,weightsUrl,
  expectedWeightBytes,expectedSourceETag,rgba,onBeforePhase,onBeforeDuty,onAfterDuty}){
  if(!isLoaderMemoryBudget(memoryBudget))throw TypeError('authenticated loader budget required');
  memoryBudget.assertDeviceAcquiredHere(device);
  if(handle?.device!==device||handle._residentClipOwner)throw Error('owned nonquarantined CLIP handle required');
  if(!(rgba instanceof Uint8Array||rgba instanceof Uint8ClampedArray)||rgba.length!==512*512*4)
    throw TypeError('complete condition-size canonical RGBA required');
  if(typeof onBeforePhase!=='function'||typeof onBeforeDuty!=='function')throw TypeError('fresh phase and duty guards required');
  const owner={source:null,allocations:[],leases:[],current:null,embeddings:null,tokens:null,features:null};
  handle._residentClipOwner=owner;
  const lease=(size,label)=>{const l=memoryBudget.reserveCpu(size,label);owner.leases.push(l);return l;};
  let failed=false,failure;
  try{
    owner.source=await createWeightPhaseSource(device,weightsUrl,{memoryBudget,expectedWeightBytes,expectedSourceETag});
    const source=owner.source,duties=[];
    const describe=async(name,selection,storageKind,rangeCpuBytes=0)=>{
      await onBeforePhase({name,tensors:source.describe(selection),storageKind,workGpuBytes:0,rangeCpuBytes});
    };
    const prep=selectResidentClipPhase(source,'prep'),prepTransient=512*512*4*4+3*224*224*4;
    await describe('clip-preparation-weights',prep,'cpu-fp32',prepTransient+tokenBytes);
    const embeddingLease=lease(tokenBytes,'clip-embeddings'),prepLease=lease(prepTransient,'clip-preparation-intermediates');
    owner.embeddings=await source.withCpuWeights(prep,w=>validateClipEmbeddings(prepareClipEmbeddings(rgba,512,512,w)));
    prepLease.release();
    await onBeforeDuty({name:'clip-pipeline-acquisition',workGpuBytes:0,rangeCpuBytes:0,requiredBytes:0});
    const pipelines=await createClipResidentPipelines(device,owner);
    const gpuPhase=async(name,kind,selection,workGpuBytes)=>{
      await describe(name+'-weights',selection,'gpu-buffer');
      await source.withWeights(selection,async weights=>{
        const duty={name,kind,workGpuBytes,rangeCpuBytes:0,requiredBytes:workGpuBytes};
        await onBeforeDuty(duty);
        const encoder=device.createCommandEncoder();
        const before=owner.allocations.length;
        const output=captureGpuBufferAllocations(()=>{
          if(kind==='pre'){
            owner.current=createEmptyBuffer(device,tokenBytes,0,'clip-resident-input');
            device.queue.writeBuffer(owner.current,0,owner.embeddings);
          }
          return encodeClipVisualPhase(encoder,device,owner.current,weights,pipelines,kind);
        },{ownedAllocations:owner.allocations}).value;
        const allocated=owner.allocations.slice(before).reduce((n,a)=>n+a.size,0);
        if(allocated>workGpuBytes)throw Error('actual CLIP work demand drift');
        device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
        owner.current=output;retireWork(owner,output);
        duties.push({...duty,actualNewGpuBytes:allocated,queueCompletionAuthority:'actual-prefix-before-retirement'});
        await onAfterDuty?.(duties.at(-1));await new Promise(resolve=>setTimeout(resolve,0));
      });
    };
    await gpuPhase('clip-pre','pre',selectResidentClipPhase(source,'pre'),2*tokenBytes+16);
    owner.embeddings=null;embeddingLease.release();
    for(let b=0;b<12;b++)await gpuPhase('clip-block-'+b,'block',selectResidentClipPhase(source,'block',b),18*tokenBytes+160);
    await gpuPhase('clip-post','post',selectResidentClipPhase(source,'post'),tokenBytes+16);
    const readback={name:'clip-complete-token-readback',workGpuBytes:tokenBytes,rangeCpuBytes:2*tokenBytes,
      requiredBytes:3*tokenBytes};await onBeforeDuty(readback);
    const tokenLease=lease(2*tokenBytes,'clip-complete-token-readback');
    const staging=device.createBuffer({size:tokenBytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    owner.allocations.push({buffer:staging,size:tokenBytes});
    const copy=device.createCommandEncoder();copy.copyBufferToBuffer(owner.current,0,staging,0,tokenBytes);
    device.queue.submit([copy.finish()]);await device.queue.onSubmittedWorkDone();
    await staging.mapAsync(GPUMapMode.READ);
    owner.tokens=new Float32Array(staging.getMappedRange().slice(0));staging.unmap();
    validateClipEmbeddings(owner.tokens);retireWork(owner);owner.current=null;
    const projection=selectResidentClipPhase(source,'projection');
    await describe('clip-visual-projection',projection,'cpu-fp32',512*4);
    const featureLease=lease(512*4,'clip-projected-features');
    owner.features=await source.withCpuWeights(projection,w=>projectClipFeatures(owner.tokens,w.projection));
    if([...owner.features].some(v=>!Number.isFinite(v)))throw Error('nonfinite canonical projected CLIP features');
    owner.tokens=null;tokenLease.release();
    const result={};
    for(const head of ['roughness','metallic']){
      const selection=selectResidentClipPhase(source,head);
      // Existing head: copied512 + three512 shared + two(512 + one scalar).
      const headBytes=(512+3*512+2*(512+1))*4;
      await describe('clip-'+head+'-head',selection,'cpu-fp32',headBytes);
      const headLease=lease(headBytes,'clip-head-intermediates');
      result[head]=await source.withCpuWeights(selection,w=>runClipMaterialHead(owner.features,
        {_rawGetCPU(name){if(!Object.hasOwn(w,name))throw Error('unselected canonical head tensor: '+name);return w[name];}},head));
      headLease.release();if(!Number.isFinite(result[head])||result[head]<0||result[head]>1)
        throw Error('invalid actual '+head+' material');
    }
    owner.features=null;featureLease.release();
    return {...result,prepOffloaded:false,visualBackend:'webgpu-complete-12-blocks',tokens:50,
      duties,loadingReport:source.loadingReport,weightPhases:source.phases,
      authority:'complete canonical material computation; no native fit/numerical-reference/GLB/foreground claim'};
  }catch(error){failed=true;failure=error;throw error;}
  finally{try{await disposeResidentMaterials(handle);}
    catch(error){throw failed?new AggregateError([failure,error],
      'resident CLIP failed and cleanup unresolved',{cause:failure}):error;}}
}

export async function disposeResidentMaterials(handle){
  const owner=handle._residentClipOwner;if(!owner)return;
  await handle.device.queue.onSubmittedWorkDone();retireWork(owner);
  if(owner.source){await owner.source.dispose();owner.source=null;}
  owner.embeddings=null;owner.tokens=null;owner.features=null;owner.current=null;
  for(const l of owner.leases)l.release();owner.leases=[];handle._residentClipOwner=null;
}
