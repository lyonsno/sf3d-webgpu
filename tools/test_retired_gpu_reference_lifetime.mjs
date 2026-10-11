import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// Observe the actual private ownership inventory; no GC/native-memory claim.
const inventories=[];
class InventoryMap extends Map {constructor(...args){super(...args);inventories.push(this);}}
const source=fs.readFileSync(new URL('../src/lib/loader_memory_budget.js',import.meta.url),'utf8').replace(/^export /gm,'');
const {createLoaderMemoryBudget}=vm.runInNewContext(source+'\n({createLoaderMemoryBudget})',{Map:InventoryMap,performance});
const make=()=>{
  let failDestroy=false,failDevice=false,failHook=false;
  const resources=[];
  const allocate=descriptor=>{
    let calls=0;
    const resource={...descriptor,destroy(){calls++;if(failDestroy)throw Error('native retirement unresolved');return 'native-return';}};
    const original=resource.destroy;resources.push({resource,original,get calls(){return calls;}});
    return resource;
  };
  const device={createBuffer:allocate,createTexture:allocate,destroy(){if(failDevice)throw Error('device retirement unresolved');}};
  const originalCreate=device.createBuffer,budget=createLoaderMemoryBudget({cpuBytes:64,gpuBytes:64});
  budget.bindOwnedDevice(device,{textureBytes:descriptor=>({bytes:16,descriptor})});
  const inventory=inventories.at(-1);
  return {device,budget,inventory,resources,originalCreate,
    setFaults(native,deviceFault,hook){failDestroy=native;failDevice=deviceFault;failHook=hook;},
    blockHook(resource){
      let hook=resource.destroy;
      Object.defineProperty(resource,'destroy',{configurable:true,get(){return hook;},set(next){
        if(failHook)throw Error('retired hook restoration unresolved');hook=next;
      }});
    }};
};
const allocate=(x,kind)=>kind==='buffer'?x.device.createBuffer({size:16}):
  x.device.createTexture({width:2,height:2,depthOrArrayLayers:1,mipLevelCount:1,format:'rgba8unorm'});
for(const kind of ['buffer','texture']){
  let x=make(),resource=allocate(x,kind);
  assert.equal(x.inventory.size,1);assert.equal(x.budget.snapshot().gpu.liveBytes,16);
  const retiredHook=resource.destroy;
  assert.equal(resource.destroy(),'native-return');
  assert.equal(x.inventory.size,0,'successful retirement must release the budget strong resource reference');
  assert.equal(resource.destroy,retiredHook,'successful retirement does not mutate the callable hook');
  resource.destroy();assert.equal(x.budget.snapshot().gpu.liveBytes,0,'repeat native destruction has no second charge');
  for(let i=0;i<32;i++){const next=allocate(x,kind);assert.equal(x.inventory.size,1);next.destroy();assert.equal(x.inventory.size,0);}
  assert.equal(x.budget.events.filter(e=>e.kind==='reserved').length,33);
  assert.equal(x.budget.events.filter(e=>e.kind==='retirement-requested').length,33,'full event evidence is retained');
  assert.equal(x.budget.snapshot().gpu.physicalMemoryMeasured,false);
  x.budget.restore();assert.equal(x.device.createBuffer,x.originalCreate);
  assert.equal(resource.destroy,retiredHook,'forgotten retired wrapper stays inert after allocator restoration');
  const eventCount=x.budget.events.length;resource.destroy();assert.equal(x.budget.events.length,eventCount);

  x=make();resource=allocate(x,kind);x.setFaults(true,true,false);
  assert.throws(()=>resource.destroy(),/native retirement unresolved/);
  assert.equal(x.inventory.size,1);assert.equal(x.budget.snapshot().gpu.liveBytes,16);
  assert.throws(()=>x.device.destroy(),/device retirement unresolved/);
  assert.equal(x.inventory.size,1);assert.equal(x.budget.snapshot().gpu.liveBytes,16);
  assert.throws(()=>x.device.createBuffer({size:64}),/before allocation/);
  assert.equal(x.resources.length,1,'unresolved retirement stays charged before a replacement allocation');
  assert.throws(()=>x.budget.restore(),/retire/);
  x.setFaults(false,false,false);resource.destroy();assert.equal(x.inventory.size,0);x.budget.restore();

  x=make();resource=allocate(x,kind);x.blockHook(resource);x.setFaults(false,false,true);
  const blockedHook=resource.destroy;
  assert.equal(resource.destroy(),'native-return','native success needs no subsequent property write');
  assert.equal(x.inventory.size,0);assert.equal(resource.destroy,blockedHook);
  assert.equal(x.budget.snapshot().gpu.liveBytes,0);
  x.budget.restore();assert.equal(resource.destroy,blockedHook);

  x=make();resource=allocate(x,kind);x.setFaults(true,false,false);
  assert.throws(()=>resource.destroy(),/native retirement unresolved/);x.device.destroy();
  assert.equal(x.budget.snapshot().gpu.liveBytes,0);x.budget.restore();assert.equal(x.inventory.size,0);
  assert.equal(resource.destroy,x.resources[0].original,'whole-device recovery retains original hook restoration');
}
console.log('PASS actual buffer/texture inventory forgets only native success without retired-hook mutation; inert repeated calls, full events, exact failed native/device charges and live-hook recovery preserved; no physical credit');
