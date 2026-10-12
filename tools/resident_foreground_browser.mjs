// Inert until Kaminos calls its acquisition/mount hooks. No eager producer.
import {createLoaderMemoryBudget,createWebGpuForegroundService,hostTexturePlan} from '/dist-lib/sf3d-producer.js';
import {connectSf3dForeground} from '/foreground/sf3d-host-device.mjs';

export const sharedGpuBufferRequirements=Object.freeze({maxBufferSize:113246208,maxStorageBufferBindingSize:113246208});
let acquisition;
// debugState contains live nested renderer objects. Detach the JSON record at
// the observation boundary, before admission/transfer awaits let rendering
// advance. Only these small state records are copied, not the frame history.
const snapshotVolumeState=prototype=>JSON.parse(JSON.stringify(prototype.debugState()));
const post=async(url,value)=>{const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
  if(!response.ok)throw Error('resident foreground refused: '+await response.text());};
export async function acquireSharedGpuDevice(adapter,descriptor){
  if(acquisition)throw Error('resident foreground device already acquired');
  acquisition={status:'acquiring'};
  const response=await fetch('/resident-foreground-config',{cache:'no-store'});
  if(!response.ok)throw Error('explicit resident foreground config unavailable');
  const config=await response.json(),info=adapter.info;
  const backend={vendor:info.vendor,architecture:info.architecture,description:info.description,isFallbackAdapter:info.isFallbackAdapter??adapter.isFallbackAdapter};
  await post('/backend',backend);
  await post('/phase',{name:'device-acquisition',tensors:[],workGpuBytes:0,rangeCpuBytes:0,requiredBytes:0});
  // Separate fresh join preserves the canonical model phase sequence. All
  // declared host construction must fit the unchanged combined allowance.
  await post('/foreground-admission',{requiredBytes:config.allowance.totalBytes,grid:config.foreground.grid,
    rendererCpuBytes:config.foreground.rendererCpuBytes,sourceDepth:config.foreground.sourceDepth});
  const budget=createLoaderMemoryBudget(config.allowance);
  const rendererCpuLease=budget.reserveCpu(config.foreground.rendererCpuBytes,'resident-ordinary-foreground-initial-component-stores');
  const device=await budget.requestOwnedDevice(adapter,descriptor,{textureBytes:hostTexturePlan});
  acquisition={status:'acquired',config,backend,budget,device,rendererCpuLease,
    descriptor:{requiredLimits:{...descriptor.requiredLimits},requiredFeatures:[...(descriptor.requiredFeatures??[])]}};
  return device;
}

export async function mountComposition({prototype,sharedGpu,host}){
  const a=acquisition;
  if(a?.status!=='acquired'||sharedGpu?.device!==a.device||sharedGpu?.queue!==a.device.queue||host?.device!==a.device)
    throw Error('guarded resident host device/queue mismatch');
  a.budget.assertDeviceAcquiredHere(a.device);
  // Kaminos constructs the prototype before its asynchronous activation.
  // The host owns that activation; wait for it, without starting a second
  // initialization or accepting a merely constructed/uninitialized volume.
  while(!prototype?.foregroundGpuContext?.().active){
    const state=prototype?.debugState?.();
    if(state?.error)throw Error('resident ordinary volume activation failed: '+state.error);
    if(!state)throw Error('resident composition requires actual ordinary volume and Three host');
    await new Promise(resolve=>setTimeout(resolve,100));
  }
  const context=prototype?.foregroundGpuContext?.();
  if(context?.device!==a.device||context?.queue!==a.device.queue||context.renderer!=='ordinary-volume'||context.productFrameOwner!=='prototype')
    throw Error('resident composition requires actual ordinary volume and Three host');
  const service=createWebGpuForegroundService({routeId:'sf3d.image-to-mesh.webgpu-local.v0',device:a.device,queue:a.device.queue});
  const frames=[];
  connectSf3dForeground({device:a.device,requestForegroundOpportunity:service.request},prototype,host,receipt=>frames.push(receipt));
  const status=document.createElement('p');status.id='status';status.textContent='Guarded full-model composition; renderer warming up';
  Object.assign(status.style,{position:'fixed',left:'12px',bottom:'12px',zIndex:100000,color:'white',background:'#111c',padding:'8px',pointerEvents:'none'});
  document.body.append(status);
  let run,finish,before;
  window.__miniResidentForeground={...a,prototype,host,frames,
    snapshot(){return {sameDevice:prototype.foregroundGpuContext().device===a.device&&host.device===a.device,
      sameQueue:prototype.foregroundGpuContext().queue===a.device.queue,state:snapshotVolumeState(prototype),budget:a.budget.snapshot(),frames:frames.length,
      descriptor:a.descriptor,backend:a.backend};},
    async beginRun(runId){
      if(run||runId!==a.config.runId)throw Error('actual resident run identity mismatch');
      before=snapshotVolumeState(prototype);run=await service.beginRun(runId);return run;
    },
    finishRun(){
      if(!run)throw Error('resident foreground run not begun');
      return finish??=run.finish().then(serviceReport=>({status:'finished',runId:run.runId,sameDevice:true,sameQueue:true,
        before,after:snapshotVolumeState(prototype),frames:[...frames],serviceReport,
        deviceRetirement:'host remains live until exact owned-browser exit; no logical/physical reclamation credit'}));
    },
  };
}
