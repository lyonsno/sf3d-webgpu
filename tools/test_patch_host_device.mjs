import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createLoaderMemoryBudget,isLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
// Exercise the actual browser operation before its image allocation. The
// synthetic device establishes local ownership policy only, not native parity.
const source=fs.readFileSync(new URL('./patch_phase_browser.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'').replace('export async function','async function');
let requests=0,phaseFetches=0;
const hostDevice=fakeWeightDevice(),unexpectedDevice=fakeWeightDevice();
for(const device of [hostDevice,unexpectedDevice]){device.pushErrorScope=()=>{};device.popErrorScope=async()=>null;}
const memoryBudget=createLoaderMemoryBudget({cpuBytes:2,gpuBytes:128});memoryBudget.bindOwnedDevice(hostDevice);
const backend={vendor:'apple',isFallbackAdapter:false};
const navigator={gpu:{async requestAdapter(){requests++;return{info:backend,limits:unexpectedDevice.limits,async requestDevice(){return unexpectedDevice;}};}}};
const fetch=async url=>{if(url==='/phase-admission')return{ok:true,json:async()=>({verdict:'admitted',requiredBytes:1})};phaseFetches++;throw Error('fixture reached admitted image fetch');};
const factory=vm.compileFunction(source+';return runPatchPhase;',['createLoaderMemoryBudget','isLoaderMemoryBudget','navigator','fetch']);
const run=factory(createLoaderMemoryBudget,isLoaderMemoryBudget,navigator,fetch);
try{
  await assert.rejects(run({requested:{cpuBytes:2,gpuBytes:128},demand:{requiredBytes:1,cpu:[{name:'input',bytes:1}]}},
    {device:hostDevice,memoryBudget,backend}),/fixture reached admitted image fetch/);
  assert.equal(requests,0,'host-owned learned phase must use the already guarded exact foreground device, not acquire a second one');
  assert.equal(hostDevice.destroyed,0,'selected phase cannot destroy its caller-owned renderer');
  assert.equal(memoryBudget.snapshot().cpu.liveBytes,0,'failed phase releases only its own input reservation');
  const before=phaseFetches;
  await assert.rejects(run({demand:{requiredBytes:1}},{device:hostDevice,memoryBudget:{},backend}),/authenticated.*host/);
  assert.equal(phaseFetches,before,'unknown borrowed-baseline token cannot reach image allocation');
}finally{hostDevice.destroy();memoryBudget.restore();}
console.log('Actual patch operation reuses an authenticated pre-bound foreground device and never retires caller ownership.');
