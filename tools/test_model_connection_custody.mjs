import assert from 'node:assert/strict';
import fs from 'node:fs';
const source=fs.readFileSync(new URL('./smoke_resident_dino.mjs',import.meta.url),'utf8');
if(!source.includes('await reconnectModelWithoutNetworkCopies(')){
 const call=source.match(/browser=await (puppeteer\.connect\([^;]+\));/)[1];
 const puppeteer={connect:async options=>({networkEnabled:options.networkEnabled??true})};
 const current=await new Function('puppeteer','endpoint','return '+call)(puppeteer,'ws://owned');
 assert.equal(current.networkEnabled,false,'actual current model connection still duplicates complete binary request bodies');
}
const {reconnectModelWithoutNetworkCopies}=await import('./model_connection_custody.mjs');
assert(source.indexOf('await reconnectModelWithoutNetworkCopies(')>source.indexOf('report.foreground.captures.before='));
assert(source.includes('if(report.requested.observeDino||foregroundRoot)'));
for(const fault of [null,'endpoint','target','url','globals','pages','connect','viewport','geometry']){
 const events=[],host={},volume={},setup={},originalViewport={width:1280,height:960,deviceScaleFactor:1},
  state={__miniResidentForeground:host,__kaminosVolumePrototype:volume,__kaminosCompositionSetup:setup,innerWidth:1280,innerHeight:960,devicePixelRatio:1};
 let currentViewport=originalViewport;
 let activeBrowser;
 const makePage=(target,url)=>({url:()=>url,browser:()=>activeBrowser,viewport:()=>({...currentViewport}),
  async createCDPSession(){return{async send(method){assert.equal(method,'Target.getTargetInfo');return{targetInfo:{targetId:target}};},async detach(){}};},
  async evaluate(fn,arg){const before=globalThis.window;globalThis.window=state;try{return await fn(arg);}finally{globalThis.window=before;}}
 });
 const initial={wsEndpoint:()=>fault==='endpoint'?'ws://wrong':'ws://owned',disconnect(){events.push('old-disconnect');}};
 activeBrowser=initial;const page=makePage('target-owned','http://owned/');
 const next={wsEndpoint:()=> 'ws://owned',disconnect(){events.push('new-disconnect');},
  async pages(){if(fault==='globals')state.__miniResidentForeground={};return Array.from({length:fault==='pages'?2:1},()=>makePage(fault==='target'?'target-other':'target-owned',fault==='url'?'http://other/':'http://owned/'));}};
 const puppeteer={async connect(options){events.push('connect');assert.equal(options.browserWSEndpoint,'ws://owned');assert.equal(options.protocolTimeout,0);assert.equal(options.networkEnabled,false);
  if(fault==='connect')throw Error('injected connect failure');
  currentViewport=fault==='viewport'?{width:800,height:600}:options.defaultViewport??{width:800,height:600};
  state.innerWidth=fault==='geometry'?799:currentViewport.width;state.innerHeight=currentViewport.height;
  activeBrowser=next;return next;}};
 const run=()=>reconnectModelWithoutNetworkCopies({puppeteer,browser:initial,page,endpoint:'ws://owned'});
 if(fault){await assert.rejects(run(),/owned|custody|connect failure|viewport|geometry/);if(!['endpoint','connect'].includes(fault))assert(events.includes('new-disconnect'));}
 else{const r=await run();assert.equal(r.browser,next);assert.equal(r.page.browser(),next);assert.equal(r.receipt.effectiveNetworkEnabled,false);
  assert.equal(r.page.viewport().width,1280,'actual connect default must not reduce witnessed rendering width');
  assert.equal(r.page.viewport().height,960,'actual connect default must not reduce witnessed rendering height');
  assert.deepEqual(r.receipt.viewportBefore,originalViewport);assert.deepEqual(r.receipt.viewportAfter,originalViewport);
  assert.deepEqual(r.receipt.geometryBefore,{width:1280,height:960,deviceScaleFactor:1});assert.deepEqual(r.receipt.geometryAfter,r.receipt.geometryBefore);
  assert.equal(r.receipt.targetBefore,r.receipt.targetAfter);assert.equal(r.receipt.hostReferencesPreserved,true);assert.deepEqual(events,['old-disconnect','connect']);}
 assert.equal(state.__miniModelConnectionCustody,undefined,'ephemeral custody marker always removed');
}
console.log('PASS model reconnect disables duplicated telemetry only after initial checks, exact owned endpoint/target/URL/global custody, fail loud/cleanup');
