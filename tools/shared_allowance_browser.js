import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';

// Deliberately tiny forced-contention diagnostic, not a RAM capacity policy.
// The same guarded devices must then complete actual compute/readback work.
export async function runSharedAllowanceWitness(){
  const requested={cpuBytes:64,gpuBytes:64},parent=createLoaderMemoryBudget(requested);
  const budgets=[0,1].map(()=>createLoaderMemoryBudget({...requested,parentBudget:parent}));
  const devices=[],leases=[];
  const report={requested,outputs:[],validationErrors:[],refusals:{},backends:[]};
  const errorRow=e=>({name:e.name,message:e.message,memoryBudget:e.memoryBudget});
  try{
    for(const budget of budgets){
      // Chromium/Dawn consumes an adapter on requestDevice. Distinct runtimes
      // acquire distinct handles, verifying each effective route independently.
      const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('actual WebGPU adapter required');
      const info=adapter.info,backend={vendor:info.vendor,architecture:info.architecture,description:info.description,isFallbackAdapter:info.isFallbackAdapter??adapter.isFallbackAdapter};
      if(backend.isFallbackAdapter!==false||!/apple/i.test(info.vendor))throw Error('nonfallback Apple route required');
      report.backends.push(backend);report.backend??=backend;
      const device=await adapter.requestDevice();devices.push(device);budget.bindOwnedDevice(device);device.pushErrorScope('validation');
    }
    report.distinctOwnedDevices=devices[0]!==devices[1];if(!report.distinctOwnedDevices)throw Error('two distinct owned devices required');
    const work=async(index,contend=false)=>{
      const device=devices[index],budget=budgets[index],owned=[];
      const cpu=budget.reserveCpu(32,'compute-input-and-readback');leases.push(cpu);
      try{
        const create=(size,usage,label)=>{const b=device.createBuffer({size,usage,label});owned.push(b);return b;};
        const input=create(16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,'shared-input');
        const output=create(16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,'shared-output');
        const staging=create(16,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,'shared-readback');
        device.queue.writeBuffer(input,0,new Float32Array([7,11,13,17]));
        const pipeline=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:'@group(0) @binding(0) var<storage, read> x: array<f32>; @group(0) @binding(1) var<storage, read_write> y: array<f32>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { y[id.x] = x[id.x] * 2.0; }'}),entryPoint:'main'}});
        const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]});
        const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(4);pass.end();encoder.copyBufferToBuffer(output,0,staging,0,16);device.queue.submit([encoder.finish()]);
        if(contend){
          report.held=parent.snapshot();
          try{const unexpected=budgets[1].reserveCpu(40,'concurrent-runtime-input');unexpected.release();throw Error('shared CPU contention did not refuse');}
          catch(error){if(error.name!=='SF3DMemoryBudgetError')throw error;report.refusals.cpu=errorRow(error);}
          try{const unexpected=devices[1].createBuffer({size:32,usage:GPUBufferUsage.STORAGE,label:'concurrent-runtime-buffer'});unexpected.destroy();throw Error('shared GPU contention did not refuse');}
          catch(error){if(error.name!=='SF3DMemoryBudgetError')throw error;report.refusals.gpu=errorRow(error);}
          report.refusedChild=budgets[1].snapshot();
        }
        await staging.mapAsync(GPUMapMode.READ);report.outputs.push(Array.from(new Float32Array(staging.getMappedRange())));staging.unmap();
      }finally{cpu.release();for(const buffer of owned)buffer.destroy();}
    };
    await work(0,true);await work(1);
    for(const device of devices)report.validationErrors.push((await device.popErrorScope())?.message??null);
    report.parent=parent.snapshot();report.children=budgets.map(b=>b.snapshot());return report;
  }finally{
    for(const lease of leases)lease.release();
    const errors=[];for(let i=0;i<devices.length;i++){try{devices[i].destroy();}catch(e){errors.push(e);}try{budgets[i].restore();}catch(e){errors.push(e);}}
    try{parent.restore();}catch(e){errors.push(e);}if(errors.length)throw new AggregateError(errors,'shared owned-device cleanup failed');
  }
}
