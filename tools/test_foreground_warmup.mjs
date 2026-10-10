// Exercise the actual settling branch without a browser/GPU. This is local
// caller-config policy; the separately retained native source-depth witness
// supplies effective Kaminos control/pixel evidence.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {execFileSync} from 'node:child_process';
const revision=process.argv[2];
const source=revision?execFileSync('git',['show',revision+':tools/smoke_loader_memory.mjs'],{encoding:'utf8'}):
  fs.readFileSync(new URL('./smoke_loader_memory.mjs',import.meta.url),'utf8');
const start=source.indexOf('    const initial=await page.evaluate(()=>window.__kaminosVolumePrototype.debugState());');
const end=source.indexOf('    await page.screenshot',start);
assert.ok(start>=0&&end>start);
const settle=vm.compileFunction('return (async()=>{'+source.slice(start,end)+'})();',['page','report']);
function fake(depth){
  let calls=0,waits=0;
  return{get waits(){return waits;},async evaluate(){return {active:true,error:null,frameCount:++calls===1?3:200,controls:{emitterSourceDepth:depth}};},
    async waitForFunction(_fn,options,frames){waits++;assert.equal(options.timeout,0);assert.equal(frames,200,'actual caller warmup cannot be replaced by three frames');}};
}
const report=()=>({requested:{foreground:{sourceDepth:0.125,warmupFrames:200}},foreground:{}});
const page=fake(0.125),r=report();await settle(page,r);assert.equal(page.waits,1);
assert.equal(r.foreground.warmup.completedFrameCount,200);
assert.equal(r.foreground.warmup.requestedFrames,200);
const stale=fake(0.006);
await assert.rejects(settle(stale,report()),/did not apply requested source depth/);
assert.equal(stale.waits,0,'wrong effective source must fail before the warmup waiter');
console.log('Actual settling branch honors caller warmup and refuses stale effective source depth.');
