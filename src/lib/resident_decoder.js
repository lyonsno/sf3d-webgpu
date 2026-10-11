import {captureGpuBufferAllocations,createEmptyBuffer} from './gpu.js';
import {isLoaderMemoryBudget} from './loader_memory_budget.js';
import {createWeightPhaseSource} from './weights.js';

const channels={density:1,vertex_offset:3,features:3,perturb_normal:3};

export function selectDecoderHeads(template,heads){
  if(!Array.isArray(heads)||!heads.length||new Set(heads).size!==heads.length||
    heads.some(h=>!channels[h]||!Array.isArray(template?.decoder?.heads?.[h])||
      template.decoder.heads[h].length!==(h==='density'||h==='vertex_offset'?3:4)))
    throw TypeError('complete canonical selected decoder heads required');
  return {heads:Object.fromEntries(heads.map(h=>[h,template.decoder.heads[h]]))};
}

/** Canonical range source/selected full heads; no eager complete weight set. */
export async function runResidentDecoder({device,decoder,memoryBudget,weightsUrl,
  expectedWeightBytes,expectedSourceETag,positions,triplanesBuf,heads,batchPoints,
  onBeforePhase,onBeforeDuty,withResult,onBeforeSourceIntake,onAfterDuty}){
  if(!isLoaderMemoryBudget(memoryBudget))throw TypeError('authenticated loader budget required');
  memoryBudget.assertDeviceAcquiredHere(device);
  if(decoder?.device!==device||decoder._residentDecoderAdapterOwner||decoder._residentQueryWorkOwner)
    throw Error('owned nonquarantined decoder handle required');
  if(typeof onBeforePhase!=='function'||typeof onBeforeDuty!=='function'||typeof withResult!=='function')
    throw TypeError('fresh phase/range guards and complete consumer required');
  const owner={source:null};decoder._residentDecoderAdapterOwner=owner;
  let failed=false,failure;
  try{
    owner.source=await createWeightPhaseSource(device,weightsUrl,
      {memoryBudget,expectedWeightBytes,expectedSourceETag,onBeforeSourceIntake});
    const source=owner.source,selection=selectDecoderHeads(source.template,heads);
    await onBeforePhase({name:'decoder-selected-complete-heads',tensors:source.describe(selection),workGpuBytes:0});
    const result=await source.withWeights(selection,weights=>runResidentDecoderQueries({device,decoder,
      memoryBudget,positions,triplanesBuf,weights,heads,batchPoints,onBeforePhase,onBeforeDuty,withResult,onAfterDuty}));
    return {...result,loadingReport:source.loadingReport,weightPhases:source.phases};
  }catch(error){failed=true;failure=error;throw error;}
  finally{try{await disposeResidentDecoder(decoder);}
    catch(error){throw failed?new AggregateError([failure,error],
      'resident decoder failed and owned cleanup unresolved',{cause:failure}):error;}}
}

export async function disposeResidentDecoder(decoder){
  await disposeResidentDecoderQueries(decoder);
  const owner=decoder._residentDecoderAdapterOwner;if(!owner)return;
  if(owner.source){await owner.source.dispose();owner.source=null;}
  decoder._residentDecoderAdapterOwner=null;
}

/** Identified existing decoder allocation graph, not opaque driver memory.
 * Every point is processed; batchPoints bounds reusable scratch, never output. */
export function describeResidentDecoderDemand({numPoints,batchPoints,heads,weights}){
  if(!Number.isSafeInteger(numPoints)||numPoints<1||!Number.isSafeInteger(batchPoints)||batchPoints<1)
    throw TypeError('positive complete point count and explicit batch size required');
  if(!Array.isArray(heads)||!heads.length||new Set(heads).size!==heads.length||
    heads.some(h=>!channels[h]||!Array.isArray(weights?.heads?.[h])||weights.heads[h].length!==(h==='density'||h==='vertex_offset'?3:4)))
    throw TypeError('exact selected canonical decoder heads required');
  const capacity=Math.min(numPoints,batchPoints);
  // scaled xyz, 3xy grids, 3 sampled40 channels, concatenated120 channels.
  let perPoint=12+3*8+3*160+480,base=0;
  // Sum every actual uniform call without relying on cache deduplication:
  // scale16 + extract3*16 + sample3*20 + concat16; head linear20,
  // hidden SiLU16, density bias/exp16+16 or other final activation16.
  let uniformPerShape=16+3*16+3*20+16;
  for(const h of heads){const hidden=weights.heads[h].length-1;
    perPoint+=hidden*2*64*4+channels[h]*4;
    base+=hidden*4; // actual SiLU dummy buffers
    if(h!=='vertex_offset')perPoint+=channels[h]*4;
    uniformPerShape+=weights.heads[h].length*20+hidden*16+
      (h==='density'?32:h==='vertex_offset'?0:16);
  }
  const resultChannels=heads.reduce((n,h)=>n+channels[h],0);
  const outputCpuBytes=numPoints*resultChannels*4;
  const readbackBytes=capacity*Math.max(...heads.map(h=>channels[h]))*4;
  // Only full range and optional tail have distinct N. This is a source-derived
  // allocation bound, not an arbitrary device cap or a physical-memory claim.
  const uniformBytes=uniformPerShape*(numPoints%capacity?2:1);
  const workGpuBytes=(perPoint+12)*capacity+base+uniformBytes+readbackBytes;
  const rangeCpuBytes=2*readbackBytes;
  return {capacity,outputCpuBytes,workGpuBytes,rangeCpuBytes,
    requiredBytes:workGpuBytes+rangeCpuBytes,
    components:{decoderPerPointBytes:perPoint,decoderBaseBytes:base,
      inputGpuBytes:12*capacity,readbackBytes,uniformBytes},
    authority:'identified explicit complete-query backing; no physical reclaim or driver-capacity guarantee'};
}

/** Borrowed weights/triplanes/device remain owned by the awaited parent. */
export async function runResidentDecoderQueries({device,decoder,memoryBudget,positions,triplanesBuf,
  weights,heads,batchPoints,onBeforePhase,onBeforeDuty,withResult,onAfterDuty}){
  if(!isLoaderMemoryBudget(memoryBudget))throw TypeError('authenticated loader budget required');
  memoryBudget.assertDeviceAcquiredHere(device);
  if(decoder?.device!==device||decoder._residentQueryWorkOwner||decoder._slotProvider||decoder._uniformCache.size)
    throw Error('owned empty nonquarantined decoder required');
  if(!(positions instanceof Float32Array)||!positions.length||positions.length%3||
    triplanesBuf?.size!==70778880)throw TypeError('complete xyz positions and borrowed postprocessor shape required');
  if(typeof onBeforePhase!=='function'||typeof onBeforeDuty!=='function'||typeof withResult!=='function')
    throw TypeError('fresh output/range guards and awaited complete consumer required');
  const numPoints=positions.length/3,demand=describeResidentDecoderDemand({numPoints,batchPoints,heads,weights});
  const owner={allocations:[],slots:new Map(),leases:[],previousProvider:decoder._slotProvider};
  decoder._residentQueryWorkOwner=owner;
  let failed=false,failure;
  try{
    await onBeforePhase({name:'decoder-complete-output-allocation',numPoints,heads,
      workGpuBytes:0,rangeCpuBytes:demand.outputCpuBytes,requiredBytes:demand.outputCpuBytes});
    owner.leases.push(memoryBudget.reserveCpu(demand.outputCpuBytes,'complete-decoder-results'));
    const output=Object.fromEntries(heads.map(h=>[h,new Float32Array(numPoints*channels[h])])),ranges=[];
    let input,staging;
    decoder._slotProvider={acquire(key,size){
      let slot=owner.slots.get(key);
      if(!slot){slot=createEmptyBuffer(device,size,0,'resident-decoder:'+key);owner.slots.set(key,slot);}
      if(size>slot.size)throw Error('actual decoder slot capacity drift: '+key);
      return slot;
    }};
    for(let start=0;start<numPoints;start+=demand.capacity){
      const end=Math.min(numPoints,start+demand.capacity),count=end-start;
      const duty={kind:'decoder-point-range',start,end,numPoints,heads,
        workGpuBytes:start===0?demand.workGpuBytes:demand.components.uniformBytes,
        rangeCpuBytes:demand.rangeCpuBytes};
      duty.requiredBytes=duty.workGpuBytes+duty.rangeCpuBytes;
      await onBeforeDuty(duty);
      const encoder=device.createCommandEncoder();
      let decoded;
      captureGpuBufferAllocations(()=>{
        if(!input)input=createEmptyBuffer(device,demand.capacity*12,0,'resident-decoder-input');
        if(!staging){staging=device.createBuffer({size:demand.components.readbackBytes,
          usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,label:'resident-decoder-readback'});
          owner.allocations.push({buffer:staging,size:staging.size});}
        device.queue.writeBuffer(input,0,positions.subarray(start*3,end*3));
        decoded=decoder.decode(encoder,input,triplanesBuf,count,weights,heads);
      },{ownedAllocations:owner.allocations});
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      for(const head of heads){
        const bytes=count*channels[head]*4,copy=device.createCommandEncoder();
        copy.copyBufferToBuffer(decoded[head],0,staging,0,bytes);device.queue.submit([copy.finish()]);
        const lease=memoryBudget.reserveCpu(demand.rangeCpuBytes,'decoder-range-readback');
        try{await staging.mapAsync(GPUMapMode.READ);
          const values=new Float32Array(staging.getMappedRange().slice(0,bytes));
          for(const value of values)if(!Number.isFinite(value))throw Error('nonfinite complete decoder result');
          output[head].set(values,start*channels[head]);staging.unmap();
        }finally{lease.release();}
      }
      ranges.push({start,end,numPoints,heads:[...heads],queueCompletionAuthority:'actual-prefix-before-range-reuse'});
      await onAfterDuty?.(ranges.at(-1));
      await new Promise(resolve=>setTimeout(resolve,0));
    }
    const value=await withResult(output);
    return {value,numPoints,heads:[...heads],batchPoints,demand,ranges,
      slotInventory:[...owner.slots].map(([slotKey,buffer])=>({slotKey,bytes:buffer.size})),
      authority:'complete selected point queries; no mesh/material/GLB or reference parity claim'};
  }catch(error){failed=true;failure=error;throw error;}
  finally{
    try{await disposeResidentDecoderQueries(decoder);}
    catch(error){throw failed?new AggregateError([failure,error],'decoder failed and owned cleanup unresolved',{cause:failure}):error;}
  }
}

export async function disposeResidentDecoderQueries(decoder){
  const owner=decoder._residentQueryWorkOwner;if(!owner)return;
  await decoder.device.queue.onSubmittedWorkDone();
  const errors=[];
  for(const allocation of [...owner.allocations]){
    try{allocation.buffer.destroy();owner.allocations.splice(owner.allocations.indexOf(allocation),1);}
    catch(error){errors.push(error);}
  }
  if(errors.length)throw new AggregateError(errors,'decoder owned retirement unresolved');
  decoder._slotProvider=owner.previousProvider;decoder._uniformCache.clear();
  owner.slots.clear();for(const lease of owner.leases)lease.release();owner.leases=[];
  decoder._residentQueryWorkOwner=null;
}
