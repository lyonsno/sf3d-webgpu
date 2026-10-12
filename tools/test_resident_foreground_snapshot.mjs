import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {webcrypto} from 'node:crypto';
import {sendForegroundResult,measureJsonUtf8} from './foreground_result_transfer.mjs';

// The observed Kaminos2620 debugState getter shallow-spreads its live state;
// its ordinaryForeground object continues changing during idle rendering.
const live={frameCount:1,error:null,ordinaryForeground:{completedFrames:1,lastReceipt:{requestId:'idle-1'}},
 additive:{unknown:['retained',{value:'héllo'}]}};
const device={queue:{}},runId='snapshot-owned-run',leases=new Set();
const budget={assertDeviceAcquiredHere(value){assert.equal(value,device);},
 reserveCpu(bytes){assert(Number.isSafeInteger(bytes)&&bytes>0);const lease={release(){leases.delete(lease);}};leases.add(lease);return lease;},
 snapshot(){return {cpuBytes:20*1024*1024};}};
const prototype={debugState:()=>({...live}),foregroundGpuContext:()=>({active:true,device,queue:device.queue,renderer:'ordinary-volume',productFrameOwner:'prototype'})};
const serviceReport={receipts:[],services:[],additive:{uncapped:true}};
const context=vm.createContext({window:{},document:{createElement:()=>({style:{}}),body:{append(){}}},
 createWebGpuForegroundService:()=>({request(){},async beginRun(id){assert.equal(id,runId);return {runId:id,finish:async()=>serviceReport};}}),
 connectSf3dForeground(){},setTimeout,Error});
const source=fs.readFileSync(new URL('./resident_foreground_browser.mjs',import.meta.url),'utf8');
const body=source.replace(/^import .*;\n/gm,'').replace(/^export /gm,'').replace('let acquisition;','let acquisition=$acquisition;');
const a={status:'acquired',device,budget,config:{runId},descriptor:{requiredLimits:{},requiredFeatures:[]}};
const mount=vm.compileFunction(body+';return mountComposition;',['$acquisition'],{parsingContext:context})(a);
await mount({prototype,sharedGpu:{device,queue:device.queue},host:{device}});
const mini=context.window.__miniResidentForeground;
const warmup=mini.snapshot();
await mini.beginRun(runId);
live.frameCount=2;live.ordinaryForeground.completedFrames=2;
const finished=await mini.finishRun();
const bytes=measureJsonUtf8(finished);
let received;
const fetch=async(url,options)=>{
 if(url==='/foreground-result-admission'){
  // Actual await boundary: live rendering changes a nested object after the
  // finished result was measured, without changing the finished run itself.
  live.ordinaryForeground.completedFrames=123456789;
  live.ordinaryForeground.lastReceipt={requestId:'idle-next-and-longer'};
  live.additive.unknown.push('new-live-field');
  return {ok:true,json:async()=>({demand:{rangeCpuBytes:4*bytes}})};
 }
 assert.equal(url,'/foreground-result');
 received=JSON.parse(new TextDecoder().decode(options.body));
 return {ok:true,json:async()=>({schema:'sf3d.native-foreground-run-reference.v0',status:'retained',
  runId,bytes:options.body.byteLength,sha256:options.headers['X-Body-Sha256'],path:'/owned/full-history.json'})};
};
const originalCrypto=globalThis.crypto;
Object.defineProperty(globalThis,'crypto',{value:webcrypto,configurable:true});
try{
 await assert.doesNotReject(sendForegroundResult({value:finished,runId,fetch,budget}),
  'a finished renderer snapshot must not change while fresh serialization admission awaits');
}finally{Object.defineProperty(globalThis,'crypto',{value:originalCrypto,configurable:true});}
assert.equal(warmup.state.ordinaryForeground.completedFrames,1);
assert.equal(finished.before.ordinaryForeground.completedFrames,1);
assert.equal(finished.after.ordinaryForeground.completedFrames,2);
assert.deepEqual(JSON.parse(JSON.stringify(received.before.additive)),{unknown:['retained',{value:'héllo'}]});
assert.deepEqual(JSON.parse(JSON.stringify(received.after.additive)),{unknown:['retained',{value:'héllo'}]});
assert.equal(received.before.ordinaryForeground.completedFrames,1);
assert.equal(received.after.ordinaryForeground.completedFrames,2);
assert.equal(measureJsonUtf8(finished),bytes);
assert.equal(leases.size,0);
assert.equal((await mini.finishRun()),finished,'same completed run value is retained');
console.log('PASS actual mount/finish/transfer composition retains complete JSON snapshots across live nested-state changes; no native math or fit claim');
