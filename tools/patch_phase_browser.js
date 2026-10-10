import {loadWeightTensorUnit} from '../src/lib/weights.js';
import {createLoaderMemoryBudget,isLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {createStorageBuffer,createEmptyBuffer} from '../src/lib/gpu.js';
import {SF3DImageTokenizer} from '../src/lib/sf3d_backbone.js';
import {resizeBlendNormalize,SF3D_IMAGE_PREPROCESS} from '../src/lib/preprocess_core.js';
import patchWGSL from '../src/shaders/patch_embed_dinov2.wgsl?raw';

export async function runPatchPhase(config,hostGpu=null){
  if(hostGpu){
    if(!isLoaderMemoryBudget(hostGpu.memoryBudget))throw Error('authenticated pre-bound host budget required');
    hostGpu.memoryBudget.assertDevice(hostGpu.device);
    const snapshot=hostGpu.memoryBudget.snapshot();
    if(snapshot.cpu.maxBytes!==config.requested?.cpuBytes||snapshot.gpu.maxBytes!==config.requested?.gpuBytes)
      throw Error('effective host allowance differs from requested phase configuration');
  }
  const admissionResponse=await fetch('/phase-admission',{cache:'no-store'});
  if(!admissionResponse.ok)throw Error('selected phase admission refused: '+await admissionResponse.text());
  const admission=await admissionResponse.json();
  if(admission.verdict!=='admitted'||admission.requiredBytes!==config.demand.requiredBytes)throw Error('bound selected-phase admission required');
  const adapter=hostGpu?null:await navigator.gpu?.requestAdapter();if(!hostGpu&&!adapter)throw Error('actual WebGPU adapter required');
  const info=adapter?.info,backend=hostGpu?hostGpu.backend:{vendor:info.vendor,architecture:info.architecture,description:info.description,isFallbackAdapter:info.isFallbackAdapter??adapter.isFallbackAdapter};
  if(backend.isFallbackAdapter!==false||!/apple/i.test(backend.vendor))throw Error('nonfallback Apple route required');
  const device=hostGpu?hostGpu.device:await adapter.requestDevice({requiredLimits:{maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize}});
  const budget=hostGpu?hostGpu.memoryBudget:createLoaderMemoryBudget(config.requested);
  if(!hostGpu)budget.bindOwnedDevice(device);
  const baseline=budget.snapshot();device.pushErrorScope('validation');
  let unit,inputLease,bitmap,input,output,staging,tokenizer;
  try{
    // Reserve all explicit preprocessing/readback stores before the decode and
    // typed arrays, independently of the loader's own unit reservations.
    inputLease=budget.reserveCpu(config.demand.cpu.filter(r=>!r.name.startsWith('cumulative-')&&!r.name.startsWith('range-')).reduce((s,r)=>s+r.bytes,0),'patch-input-preprocess-readback');
    const response=await fetch('/phase-image',{cache:'no-store'});if(!response.ok)throw Error('phase image unavailable');
    bitmap=await createImageBitmap(await response.blob());
    if(bitmap.width!==config.input.width||bitmap.height!==config.input.height)throw Error('effective decoded input dimensions changed');
    const canvas=new OffscreenCanvas(bitmap.width,bitmap.height),ctx=canvas.getContext('2d');ctx.drawImage(bitmap,0,0);
    const rgba=ctx.getImageData(0,0,bitmap.width,bitmap.height).data,src=new Float32Array(rgba.length);
    for(let i=0;i<rgba.length;i++)src[i]=rgba[i]/255;
    const p=SF3D_IMAGE_PREPROCESS,chw=resizeBlendNormalize(src,bitmap.width,bitmap.height,p.condImageSize,p.bgColor,p.imageMean,p.imageStd);
    const preserve=async(url,bytes)=>{const r=await fetch(url,{method:'POST',body:bytes});if(!r.ok)throw Error('raw phase tensor persistence failed: '+await r.text());};
    await preserve('/phase-input.f32',new Uint8Array(chw.buffer));
    unit=await loadWeightTensorUnit(device,'/canonical-weights.bin',config.requested.tensorNames,{memoryBudget:budget,expectedWeightBytes:config.source.byteLength,expectedSourceETag:config.source.etag});
    const tensors=unit.tensors,names=config.requested.tensorNames;
    const weights={patchEmbed:{weight:tensors.get(names[0]),bias:tensors.get(names[1])},clsToken:tensors.get(names[2]),posEmbed:tensors.get(names[3])};
    input=createStorageBuffer(device,chw,0,'patch-image');output=createEmptyBuffer(device,1297*1024*4,0,'patch-output');
    tokenizer=new SF3DImageTokenizer(device);
    tokenizer.pipelines.patchEmbed=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:patchWGSL}),entryPoint:'main'}});
    const encoder=device.createCommandEncoder();tokenizer._dispatchPatchEmbed(encoder,input,weights,output,36,36);
    const started=performance.now();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
    const kernelMs=performance.now()-started;
    staging=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,label:'patch-readback'});
    const copy=device.createCommandEncoder();copy.copyBufferToBuffer(output,0,staging,0,output.size);device.queue.submit([copy.finish()]);
    await staging.mapAsync(GPUMapMode.READ);await preserve('/phase-output.f32',new Uint8Array(staging.getMappedRange()));staging.unmap();
    const loadingReport=unit.loadingReport;unit.dispose();unit=null;
    staging.destroy();staging=null;output.destroy();output=null;input.destroy();input=null;
    for(const b of tokenizer._uniformCache.values())b.destroy();tokenizer=null;inputLease.release();inputLease=null;
    const validation=await device.popErrorScope();
    return{backend,validationError:validation?.message??null,patchPhase:{shape:[1297,1024],kernelMs,loadingReport,budget:budget.snapshot(),
      deviceOwnership:hostGpu?'pre-bound-caller-host':'phase-owned',hostBaseline:hostGpu?baseline:null,events:budget.events}};
  }finally{
    bitmap?.close();
    // A shared phase retires only its resources, after submitted work drains.
    // It cannot destroy the foreground device or restore its host's guard.
    if(hostGpu&&(unit||input||output||staging||tokenizer))await device.queue.onSubmittedWorkDone();
    const errors=[],attempt=operation=>{try{operation();}catch(error){errors.push(error);}};
    if(unit)attempt(()=>unit.dispose());
    for(const buffer of [staging,output,input,...(tokenizer?tokenizer._uniformCache.values():[])])if(buffer)attempt(()=>buffer.destroy());
    if(inputLease)attempt(()=>inputLease.release());
    if(!hostGpu){attempt(()=>device.destroy());attempt(()=>budget.restore());}
    if(errors.length)throw new AggregateError(errors,'selected-phase resource cleanup failed');
  }
}
