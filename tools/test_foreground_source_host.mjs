import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source=fs.readFileSync(new URL('./smoke_loader_memory.mjs',import.meta.url),'utf8');

{
  const start=source.indexOf('let failStartup'),end=source.indexOf('const initial=',start),listeners=new Map();
  const boot=vm.compileFunction('return(async()=>{'+source.slice(start,end)+'})();',['page','report','URL']);
  const failed={status:()=>503,request:()=>({resourceType:()=> 'script'}),url:()=> 'http://fixture.invalid/foreground/node_modules/three-mesh-bvh/build/index.module.js'};
  const brokenPage={on(event,callback){listeners.set(event,callback);},off(){},
    async goto(){listeners.get('response')?.(failed);return{ok:()=>true};},
    async waitForFunction(){throw Error('fixture reached impossible renderer wait');}};
  await assert.rejects(boot(brokenPage,{foreground:{route:'/foreground/index.html'},url:'http://fixture.invalid'},URL),/foreground required module refused: HTTP 503/);
  const caughtFailure={on(event,callback){listeners.set(event,callback);},off(){},
    async goto(){listeners.get('console')?.({type:()=> 'error',text:()=> 'Volume cockpit initialization failed: Error: volume-cockpit-layout-store-response-invalid:409'});return{ok:()=>true};},
    async waitForFunction(){throw Error('fixture reached impossible renderer wait');}};
  await assert.rejects(boot(caughtFailure,{foreground:{route:'/foreground/index.html'},url:'http://fixture.invalid'},URL),/foreground startup exception: .*volume-cockpit-layout-store-response-invalid:409/);
}

const startup=source.match(/try\{\n      (?:const navigation=)?await Promise\.race\(\[page\.goto[\s\S]*?\}finally\{page\.off\('pageerror',failStartup\);[^\n]*\}/)[0];
const run=vm.compileFunction('return (async()=>{'+startup+'})();',['page','report','URL','startupFailure','failStartup','failRequiredResponse','failRequiredRequest','failCaughtStartup','startupError']);
let waits=0;
const page={goto:async()=>({ok:()=>false,status:()=>503}),waitForFunction:async()=>{waits++;throw Error('fixture reached impossible renderer wait');},off(){}};
await assert.rejects(run(page,{foreground:{route:'/foreground/index.html'},url:'http://fixture.invalid'},URL,new Promise(()=>{}),()=>{},()=>{},()=>{},()=>{}),/foreground navigation refused: HTTP 503/);
assert.equal(waits,0,'failed source response cannot await a nonexistent renderer');
await import('./test_foreground_service_route.mjs');
console.log('Actual-service routing and failed-navigation/module terminality are enforced.');
