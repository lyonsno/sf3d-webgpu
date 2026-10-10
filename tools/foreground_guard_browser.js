import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';

// Logical format components, not Metal allocation sizes. The WebGPU spec
// permits depth24plus to use depth32float; reserve eight component bytes for
// the combined depth/stencil case. Padding, swapchains and private backing
// remain outside this ledger and are separately observed by the parent.
// https://gpuweb.github.io/gpuweb/#texture-format-caps
const texelBytes={r8unorm:1,r8snorm:1,r8uint:1,r8sint:1,r16uint:2,r16sint:2,r16float:2,
  rg8unorm:2,rg8snorm:2,rg8uint:2,rg8sint:2,r32uint:4,r32sint:4,r32float:4,
  rg16uint:4,rg16sint:4,rg16float:4,rgba8unorm:4,'rgba8unorm-srgb':4,rgba8snorm:4,rgba8uint:4,rgba8sint:4,
  bgra8unorm:4,'bgra8unorm-srgb':4,rgb10a2unorm:4,rg11b10ufloat:4,rgb9e5ufloat:4,
  rg32uint:8,rg32sint:8,rg32float:8,rgba16uint:8,rgba16sint:8,rgba16float:8,
  rgba32uint:16,rgba32sint:16,rgba32float:16,depth16unorm:2,depth32float:4,depth24plus:4,
  'depth24plus-stencil8':8,'depth32float-stencil8':8,stencil8:1};
const positive=(n,label)=>{if(!Number.isSafeInteger(n)||n<1)throw Error('positive exact '+label+' required');return n;};
export function hostTexturePlan(input){
  const descriptor={...input};
  const raw=descriptor.size,iterable=raw?.[Symbol.iterator];
  const size=iterable?[...raw]:[raw?.width,raw?.height??1,raw?.depthOrArrayLayers??1];
  const [width,height,depth]=[positive(size[0],'texture width'),positive(size[1]??1,'texture height'),positive(size[2]??1,'texture depth')];
  const dimension=descriptor.dimension??'2d',mips=positive(descriptor.mipLevelCount??1,'mips'),samples=positive(descriptor.sampleCount??1,'samples');
  const stride=texelBytes[descriptor.format];
  if(!stride||!['1d','2d','3d'].includes(dimension))throw Error('unaccounted host texture format/dimension '+descriptor.format+'/'+dimension);
  if(mips>1+Math.floor(Math.log2(Math.max(width,height,dimension==='3d'?depth:1))))throw Error('impossible texture mip extent');
  let bytes=0;
  for(let level=0;level<mips;level++)bytes+=Math.max(1,Math.floor(width/2**level))*Math.max(1,Math.floor(height/2**level))*(dimension==='3d'?Math.max(1,Math.floor(depth/2**level)):depth)*stride*samples;
  if(!Number.isSafeInteger(bytes))throw Error('texture store byte count overflow');
  return{bytes,descriptor:{...descriptor,size:{width,height,depthOrArrayLayers:depth},dimension,mipLevelCount:mips,sampleCount:samples}};
}

export function createForegroundGuard(config){
  const root=createLoaderMemoryBudget(config.allowance),cpuLease=root.reserveCpu(config.rendererCpuBytes,'ordinary-flame-explicit-initialization-stores');
  const children=[],queueSubmissions=[];
  const guard={root,children,queueSubmissions,config,
    adoptNewDevice(device,adapter){
      const info=adapter.info,backend={vendor:info.vendor,architecture:info.architecture,description:info.description,isFallbackAdapter:info.isFallbackAdapter??adapter.isFallbackAdapter};
      if(backend.isFallbackAdapter!==false||!/apple/i.test(backend.vendor)){device.destroy();throw Error('actual nonfallback Apple foreground adapter required');}
      const memoryBudget=createLoaderMemoryBudget({...config.allowance,parentBudget:root});
      try{memoryBudget.bindOwnedDevice(device,{textureBytes:hostTexturePlan});}catch(error){device.destroy();throw error;}
      const submit=device.queue.submit,deviceIndex=children.length;
      device.queue.submit=function(commands){const result=Reflect.apply(submit,this,[commands]);queueSubmissions.push({deviceIndex,atMs:performance.now()});return result;};
      children.push({device,memoryBudget,backend,deviceIndex});
      return device;
    },
    forDevice(device){const child=children.find(row=>row.device===device);if(!child)throw Error('foreground device not bound before initialization');child.memoryBudget.assertDevice(device);return child;},
    snapshot(){return{root:root.snapshot(),children:children.map(row=>({backend:row.backend,budget:row.memoryBudget.snapshot()})),queueSubmissions:[...queueSubmissions]};},
    async retire(){
      const errors=[];
      for(const row of children){try{await row.device.queue.onSubmittedWorkDone();row.device.destroy();row.memoryBudget.restore();}catch(error){errors.push(error);}}
      cpuLease.release();
      try{root.restore();}catch(error){errors.push(error);}
      if(errors.length)throw new AggregateError(errors,'foreground host retirement failed');
      return guard.snapshot();
    },
  };
  return guard;
}

// Called in the fresh page before any application module executes. Device
// acquisition waits for guard import and binding completes before the device
// can reach Three or the ordinary flame. Navigation gets its own realm.
export function installForegroundAcquisition(config){
  const gpu=navigator.gpu;if(!gpu)throw Error('WebGPU foreground route unavailable');
  const acquire=gpu.requestAdapter;
  const loading=import('/tools/foreground_guard_browser.js').then(module=>module.createForegroundGuard(config));
  window.__miniForegroundGuardPromise=loading;
  gpu.requestAdapter=async function(...args){
    const guard=await loading,adapter=await Reflect.apply(acquire,this,args);if(!adapter)return adapter;
    const request=adapter.requestDevice;
    adapter.requestDevice=async function(...deviceArgs){const device=await Reflect.apply(request,this,deviceArgs);return guard.adoptNewDevice(device,adapter);};
    return adapter;
  };
}
