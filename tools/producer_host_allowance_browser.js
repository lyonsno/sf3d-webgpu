import {createSf3dProducer,createLoaderMemoryBudget} from '/dist-lib/sf3d-producer.js';

// Actual bundled constructor and guarded-from-acquisition native device. The
// byte-sized synthetic source refusal is not a model run or capacity policy.
export async function runProducerHostAllowance(fixtureBytes){
  const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('actual WebGPU adapter required');
  const info=adapter.info,backend={vendor:info.vendor,architecture:info.architecture,description:info.description,isFallbackAdapter:info.isFallbackAdapter??adapter.isFallbackAdapter};
  if(backend.isFallbackAdapter!==false||!/apple/i.test(backend.vendor))throw Error('actual nonfallback Apple host adapter required');
  const budget=createLoaderMemoryBudget({cpuBytes:1,gpuBytes:128}),device=await budget.requestOwnedDevice(adapter);
  let host,producer;const report={moduleUrl:'/dist-lib/sf3d-producer.js',backend,fetchCount:0};
  device.pushErrorScope('validation');
  try{
    host=device.createBuffer({size:80,usage:GPUBufferUsage.STORAGE,label:'host-baseline-control'});
    report.before=budget.snapshot();
    const original=globalThis.fetch;
    globalThis.fetch=(...args)=>{report.fetchCount++;return original(...args);};
    try{
      producer=await createSf3dProducer({device,adapter,memoryBudget:budget,workers:{},weightsUrl:'/fixture/f32',expectedWeightBytes:fixtureBytes});
      throw Error('expected actual producer source refusal did not occur');
    }catch(error){
      if(error.name!=='SF3DMemoryBudgetError')throw error;
      report.refusal={name:error.name,message:error.message,memoryBudget:error.memoryBudget};
    }finally{globalThis.fetch=original;}
    budget.assertDeviceAcquiredHere(device);report.sameDevice=true;
    report.after=budget.snapshot();
    try{
      const unexpected=device.createBuffer({size:128,usage:GPUBufferUsage.STORAGE,label:'post-producer-guard-control'});
      unexpected.destroy();throw Error('host guard did not survive producer refusal');
    }catch(error){
      if(error.name!=='SF3DMemoryBudgetError')throw error;
      report.postRefusal={name:error.name,message:error.message,memoryBudget:error.memoryBudget};
    }
    const input=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    let output;
    try{
      output=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      device.queue.writeBuffer(input,0,new Uint32Array([7,11,13,17]));
      const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(input,0,output,0,16);device.queue.submit([encoder.finish()]);
      await output.mapAsync(GPUMapMode.READ);report.control=Array.from(new Uint32Array(output.getMappedRange()));output.unmap();
    }finally{input.destroy();output?.destroy();}
    report.validationError=(await device.popErrorScope())?.message??null;
  }finally{
    if(producer)await producer.dispose().completion;
    host?.destroy();device.destroy();report.terminal=budget.snapshot();budget.restore();
  }
  return report;
}
