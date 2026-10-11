import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('./resident_foreground_browser.mjs',import.meta.url),'utf8');
const body=source.replace(/^import .*;\n/gm,'').replace(/^export /gm,'').replace('let acquisition;','let acquisition=$acquisition;');
async function exercise(mode){
  const device={queue:{}},events=[],state={active:false,error:mode==='failed'?'actual-volume-init-error':null};
  const prototype={debugState:()=>({...state}),foregroundGpuContext:()=>({device:state.active?device:undefined,queue:state.active?device.queue:undefined,
    active:state.active,renderer:mode==='alternate'?'alternate-volume':'ordinary-volume',productFrameOwner:mode==='caller'?'caller':'prototype'})};
  const host={device},budget={assertDeviceAcquiredHere(value){assert.equal(value,device);events.push('owned');}};
  const a={status:'acquired',device,budget,config:{runId:'run'},descriptor:{requiredLimits:{},requiredFeatures:[]}};
  const context=vm.createContext({window:{},document:{createElement:()=>({style:{}}),body:{append(){events.push('status');}}},
    createWebGpuForegroundService(){events.push('service');return {request(){},async beginRun(){return {};}};},
    connectSf3dForeground(_producer,p,h){assert.equal(p,prototype);assert.equal(h,host);assert.equal(state.active,true);events.push('connected');},
    setTimeout(resolve){queueMicrotask(()=>{events.push('activation');state.active=true;resolve();});},Error});
  const mount=vm.compileFunction(body+';return mountComposition;', ['$acquisition'],{parsingContext:context})(a);
  if(mode==='valid'){
    await assert.doesNotReject(Promise.resolve().then(()=>mount({prototype,sharedGpu:{device,queue:device.queue},host})),
      'actual mount must await the ordinary volume activation rather than reject its pre-activation context');
    assert.ok(events.indexOf('activation')<events.indexOf('service'));assert.ok(events.includes('connected'));
    assert.ok(context.window.__miniResidentForeground);
  }else{
    await assert.rejects(Promise.resolve().then(()=>mount({prototype,sharedGpu:{device,queue:device.queue},host})),
      mode==='failed'?/actual-volume-init-error/:/actual ordinary volume and Three host/);
    assert.equal(events.includes('service'),false,'failed or alternate host must not receive a mounted service');
  }
}
for(const mode of ['valid','failed','alternate','caller'])await exercise(mode);
console.log('actual resident mount waits for ordinary activation and rejects failed/alternate/caller hosts');

const native=fs.readFileSync(new URL('./smoke_resident_dino.mjs',import.meta.url),'utf8');
const fail=native.match(/const failConsole=message=>\{[^\n]+/)[0];
let error;
vm.runInNewContext(fail+';failConsole(message);',{message:{type:()=> 'error',text:()=> 'composition module failed: Error: actual-mount-failure'},
  failStartup:value=>{error=value;},Error});
assert.match(error?.message??'',/actual-mount-failure/,'actual K composition failure must release awaited startup and reach owned cleanup');
assert.match(native,/page\.goto\(new URL\(report\.foreground\.route,report\.url\)\.href,\{waitUntil:'domcontentloaded',timeout:0\}/,
  'actual mount-state observation must own initialization, not an unending global network-idle predecessor');
console.log('actual composition error reaches startup failure and initialization does not depend on network idle');
