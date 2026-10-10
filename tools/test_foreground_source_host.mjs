import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {createHash} from 'node:crypto';

const source=fs.readFileSync(new URL('./smoke_loader_memory.mjs',import.meta.url),'utf8');
const branch=source.slice(source.indexOf("if(foregroundRoot&&name.startsWith('/foreground/')){"),source.indexOf("if(name==='/'){"));
const serve=vm.compileFunction(branch,['name','foregroundRoot','req','res','path','foregroundSources','report','execFileSync','fs','digest']);
// Observed pinned index.html length, above Node's default child output cap.
// This fixture tests complete-output policy; actual Git/browser conformance
// is independently exercised against Kaminos70624e5 by the native consumer.
const data=Buffer.alloc(1549166,32),calls=[];
const git=(_command,args,options={})=>{
  calls.push({args,options});
  if(args[0]==='cat-file'&&args[1]==='-s')return String(data.length)+'\n';
  if((options.maxBuffer??1048576)<data.length)throw Error('spawnSync git ENOBUFS');
  return data;
};
let status,body;
const res={writeHead(code){status=code;return this;},end(value){body=value;return this;}};
serve('/foreground/index.html','/source',{},res,path,new Map(),{foreground:{source:{revision:'observed'},servedSources:{}}},git,
  {readFileSync:()=>data},bytes=>createHash('sha256').update(bytes).digest('hex'));
if(!process.argv.includes('--navigation-only')){
  assert.equal(status,200,'complete observed cockpit exceeds hidden default output cap and must still be served');
  assert.equal(body.length,1549166);
}

const startup=source.match(/try\{\n      (?:const navigation=)?await Promise\.race\(\[page\.goto[\s\S]*?\}finally\{page\.off\('pageerror',failStartup\);\}/)[0];
const run=vm.compileFunction('return (async()=>{'+startup+'})();',['page','report','URL','startupFailure','failStartup']);
let waits=0;
const page={goto:async()=>({ok:()=>false,status:()=>503}),waitForFunction:async()=>{waits++;throw Error('fixture reached impossible renderer wait');},off(){}};
await assert.rejects(run(page,{foreground:{route:'/foreground/index.html'},url:'http://fixture.invalid'},URL,new Promise(()=>{}),()=>{}),/foreground navigation refused: HTTP 503/);
assert.equal(waits,0,'failed source response cannot await a nonexistent renderer');
console.log('Complete measured foreground source and failed-navigation terminality are enforced.');
