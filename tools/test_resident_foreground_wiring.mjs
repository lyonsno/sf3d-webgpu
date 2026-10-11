import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {webcrypto} from 'node:crypto';
import {createWebGpuForegroundService} from '@kaminos/webgpu-inference-kit';

// Execute the actual native browser callback with a pre-acquired host device.
// Synthetic work only checks wiring/ownership, never native model conformance.
const source=fs.readFileSync(new URL('./smoke_resident_dino.mjs',import.meta.url),'utf8');
const start=source.indexOf('Object.assign(report,await page.evaluate(async config=>{');
const end=source.indexOf('\n  },{',start);
assert.ok(start>0&&end>start,'actual native browser callback located');
const body=source.slice(start,end).replace('Object.assign(report,await page.evaluate(async config=>{','async config=>{')
  .replace("await import('/dist-lib/sf3d-producer.js')",'globalThis.__bundle')+'\n}';
let secondAcquisition=0,destroyed=0,restored=0,finished=0;
let actualBoundary;
const options={},opportunities={serviceAtBoundary:async boundary=>{actualBoundary=boundary;}},withForeground=async(_phase,work)=>work();
const device={queue:{submit(){},async onSubmittedWorkDone(){}},pushErrorScope(){},async popErrorScope(){return null;},destroy(){destroyed++;},
  createCommandEncoder(){return {copyBufferToBuffer(){},finish(){return {};}};},
  createBuffer({size}){return {size,async mapAsync(){},getMappedRange(){return new ArrayBuffer(size);},unmap(){},destroy(){}};}};
const budget={assertDeviceAcquiredHere(d){assert.equal(d,device);},reserveCpu(){return {release(){}};},snapshot(){return {cpu:{liveBytes:20},gpu:{liveBytes:40}};},restore(){restored++;}};
const context={device,budget,backend:{vendor:'apple',isFallbackAdapter:false},
  async beginRun(runId){assert.equal(runId,'owned-run');return {foregroundOpportunities:opportunities,withForeground};},
  async finishRun(){finished++;return {status:'finished',sameDevice:true};}};
const bundle={createLoaderMemoryBudget(){throw Error('second budget forbidden');},
  async preprocessConditionImage(){return {chw:new Float32Array(4),rgba:new Float32Array(4)};},
  TwoStreamBackbone:class{constructor(d){assert.equal(d,device);}init(){}},
  async runResidentDino(input){options.dino=input;await input.withResult({tokensBuf:{size:4},N:1297});return {};},
  async runResidentTwoStream(input){options.backbone=input;await input.withResult({buffer:{size:0}});return {};},
  async runResidentPostProcessor(input){options.post=input;await input.withResult({buffer:{size:0}});return {};},
  async runResidentArtifact(input){options.artifact=input;return {};},
  async disposeResidentTwoStream(){},async disposeResidentArtifact(){}};
const callback=vm.runInNewContext('('+body+')',{
  __bundle:bundle,window:{__miniResidentForeground:context},navigator:{gpu:{async requestAdapter(){secondAcquisition++;throw Error('unsafe second device acquisition');}}},
  fetch:async()=>({ok:true,async text(){return '';},async blob(){return {};}}),
  createImageBitmap:async()=>({width:503,height:503,close(){}}),document:{querySelector(){return {textContent:''};}},
  crypto:webcrypto,GPUBufferUsage:{COPY_DST:1,MAP_READ:2},GPUMapMode:{READ:1},Float32Array,Uint8Array,ArrayBuffer,Error,AggregateError,
});
let result,error;
try{result=await callback({runId:'owned-run',requested:{foreground:{grid:32},throughFullModel:true,throughBackbone:true,throughPostProcessor:true},input:{width:503,height:503},source:{},allocatingDutyIndices:[]});}catch(e){error=e;}
assert.equal(secondAcquisition,0,'resident model must use the already guarded exact host device, never request a second adapter/device');
assert.equal(error,undefined,error?.stack);
assert.equal(finished,1,'one actual foreground finish boundary');
for(const name of ['dino','backbone','post'])assert.equal(options[name].foregroundOpportunities,opportunities,name+' receives the common interlock');
assert.equal(options.artifact.withForeground,withForeground,'artifact CPU windows receive the actual common service');
assert.equal(typeof options.artifact.onAfterDuty,'function','actual CLIP and decoder prefixes service foreground');
assert.equal(destroyed,0,'borrowed shared host device must survive model teardown');
assert.equal(restored,0,'live host allocations must not be relabeled retired');
assert.equal(result.foregroundRun.status,'finished');
await options.artifact.onAfterDuty({name:'clip-block-0'});
const actualService=createWebGpuForegroundService({routeId:'synthetic-wiring',device,queue:device.queue});
const actualRun=await actualService.beginRun('synthetic-wiring');
await actualRun.foregroundOpportunities.serviceAtBoundary(actualBoundary);
await actualRun.finish();await actualService.dispose();
// A failed finish must not destroy/restore a live borrowed host or falsely
// retire quarantined artifact resources. The outer owned browser is the exit.
context.finishRun=async()=>{throw Error('foreground settlement unavailable');};
let artifactRetired=false;bundle.disposeResidentArtifact=async()=>{artifactRetired=true;};
await assert.rejects(()=>callback({runId:'owned-run',requested:{foreground:{grid:32},throughFullModel:true,throughBackbone:true,throughPostProcessor:true},input:{width:503,height:503},source:{},allocatingDutyIndices:[]}),/cleanup failed/);
assert.equal(artifactRetired,false);assert.equal(destroyed,0);assert.equal(restored,0);
console.log('Actual resident callback uses guarded exact host device, all stage interlocks and borrowed-device cleanup');
