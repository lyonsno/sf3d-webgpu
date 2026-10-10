import assert from 'node:assert/strict';
import {TwoStreamBackbone} from '../src/lib/two_stream.js';
import {runCooperativeTwoStream,retireTwoStreamWork} from '../src/lib/cooperative_two_stream.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64};
for(const destroyFails of [false,true]){
  const device=fakeWeightDevice(),create=device.createBuffer;let failDestroy=destroyFails;
  device.createBuffer=function(desc){const buffer=create.call(this,desc),destroy=buffer.destroy;
    buffer.getMappedRange=()=>{throw Error('injected uniform mapping failure');};
    buffer.destroy=()=>{if(failDestroy)throw Error('injected uniform retirement failure');destroy.call(buffer);};
    return buffer;
  };
  device.createCommandEncoder=()=>({finish(){return {};}});
  const backbone=new TwoStreamBackbone(device);
  backbone.dispatchAttentionForwardDuty=()=>backbone._cachedUniform(new Uint32Array([1,2,3,4,5,6]));
  const options={device,backbone,imageTokensBuf:{size:16},N_img:1,weights:{},
    dutyGranularity:'attention-tile',residentFFN:true,retireIntermediateBuffers:true,schedulingMode:'disabled'};
  await assert.rejects(runCooperativeTwoStream(options));
  assert.equal(device.buffers.length,1);
  if(!destroyFails){
    assert.equal(device.buffers[0].destroyed,1,'uniform mapping failure must retire its actual allocation');
    assert.equal(backbone._uniformCache.size,0);
  }else{
    assert.ok(backbone._failedUniforms?.has(device.buffers[0]),'failed uniform retirement stays reachable for recovery');
    await assert.rejects(runCooperativeTwoStream(options),/quarantined/);
    assert.throws(()=>backbone._cachedUniform(new Uint32Array([1,2,3,4,5,6])),/quarantined/);
    assert.equal(device.buffers.length,1,'quarantine must not allocate another uniform');
    failDestroy=false;await retireTwoStreamWork(backbone);
    assert.equal(device.buffers[0].destroyed,1);assert.equal(backbone._failedUniforms.size,0);
    assert.equal(backbone._residentWorkOwner,null);
  }
}
console.log('uniform initialization rejection, exact retirement, failed destruction quarantine and recovery');

