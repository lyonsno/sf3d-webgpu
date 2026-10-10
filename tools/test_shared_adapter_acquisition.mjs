import assert from 'node:assert/strict';
import {runSharedAllowanceWitness} from './shared_allowance_browser.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
// Replay the consumed-adapter contract observed on Chromium140/Dawn/Metal in
// shared-allowance-7572b1b-exact-20261010/report.json, not a capacity fixture.
const original=Object.getOwnPropertyDescriptor(globalThis,'navigator');let requests=0;
Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu:{async requestAdapter(){
  if(++requests===2)throw Error('fixture reached second independent adapter request');
  let consumed=false;return{info:{vendor:'apple',isFallbackAdapter:false},async requestDevice(){
    if(consumed)throw Error('adapter is consumed: already used to create a device');consumed=true;return fakeWeightDevice();
  }};
}}}});
try{await assert.rejects(runSharedAllowanceWitness(),/fixture reached second independent adapter request/);assert.equal(requests,2);}
finally{if(original)Object.defineProperty(globalThis,'navigator',original);else delete globalThis.navigator;}
console.log('Each actual owned runtime requests its own adapter handle; consumed adapters are not reused.');
