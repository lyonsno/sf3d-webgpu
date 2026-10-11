import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createLoaderMemoryBudget,isLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';

// Exercise the actual resident consumer up to source construction. This tests
// selection/custody, not native upload conformance or physical memory benefit.
const source=fs.readFileSync(new URL('../src/lib/resident_two_stream.js',import.meta.url),'utf8')
  .replace(/^import .*;\n/gm,'').replace(/^export /gm,'');
const factory=vm.compileFunction(source+'\nreturn {runResidentTwoStream};',
  ['createWeightPhaseSource','isLoaderMemoryBudget','captureGpuBufferAllocations',
    'dispatchTokenizerEmbedding','runCooperativeTwoStream','retireTwoStreamWork']);
const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:256*1024*1024,gpuBytes:1024*1024*1024});
await budget.requestOwnedDevice({requestDevice:async()=>device});
const backbone={device,_uniformCache:new Map()},sentinel=Error('stop after exact resident source selection');
const onBeforeSourceIntake=async()=>{},calls=[];
let drains=0;const fence=device.queue.onSubmittedWorkDone.bind(device.queue);
device.queue.onSubmittedWorkDone=async()=>{drains++;await fence();};
const {runResidentTwoStream}=factory(async(actualDevice,url,options)=>{
  assert.equal(actualDevice,device);assert.equal(url,'canonical-full-weights.bin');
  assert.equal(options.memoryBudget,budget);
  assert.equal(options.expectedWeightBytes,2285308688);
  assert.equal(options.expectedSourceETag,'"sha256-canonical-full-source"');
  assert.equal(options.onBeforeSourceIntake,onBeforeSourceIntake);
  calls.push(options);
  assert.equal(options.streamFp16Uploads,true,
    'actual complete resident backbone must select the authenticated prefix upload path');
  throw sentinel;
},isLoaderMemoryBudget,()=>{throw Error('unexpected GPU allocation');},
()=>{throw Error('unexpected embedding');},()=>{throw Error('unexpected model execution');},
async actual=>assert.equal(actual,backbone));
let failure;
try{
  try{await runResidentTwoStream({device,backbone,memoryBudget:budget,weightsUrl:'canonical-full-weights.bin',
    expectedWeightBytes:2285308688,expectedSourceETag:'"sha256-canonical-full-source"',
    imageTokensBuf:{size:1297*1024*4},N_img:1297,onBeforeSourceIntake,
    onBeforePhase:async()=>{},onBeforeDuty:async()=>{},withResult:async()=>{}});}catch(error){failure=error;}
  if(failure!==sentinel)throw failure;
  assert.equal(calls.length,1);assert.equal(drains,1,'actual consumer cleanup still drains');
  assert.equal(backbone._residentAdapterOwner,null,'exact consumer owner retired after pre-intake refusal');
  assert.equal(device.buffers.length,0);assert.equal(budget.snapshot().cpu.liveBytes,0);assert.equal(budget.snapshot().gpu.liveBytes,0);
}finally{device.destroy();budget.restore();}
console.log('PASS actual complete resident backbone selects full prefix upload and retains exact canonical intake, authenticated budget and cleanup');

