import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// Exercise the actual entrypoint handler, not a second routing implementation.
const source=fs.readFileSync(new URL('./smoke_loader_memory.mjs',import.meta.url),'utf8');
const body=source.slice(source.indexOf('server=http.createServer((req,res)=>{')+'server=http.createServer((req,res)=>{'.length,
  source.indexOf('\n  await new Promise(resolve=>server.listen'));
const make=vm.compileFunction('return(req,res)=>{'+body.replace(/\}\);\s*$/,'')+'};',
  ['foregroundRoot','foregroundService','report','sources','fixtures','patchMode','canonical','URL','path','foregroundSources','foregroundDependencies','execFileSync','fs','digest']);
const forwarded=[];
const service={proxy(req,res){forwarded.push(req.url);res.writeHead(200).end('actual-handler-response');}};
const handler=make('/pinned-kaminos',service,{foreground:{source:{revision:'pinned'},servedSources:{}}},new Map(),{},false,null,URL,
  {},new Map(),new Map(),()=>{throw Error('not used for root API');},{},()=> 'hash');
function request(url,method='GET'){
  const result={};
  handler({url,method},{setHeader(){},writeHead(status){result.status=status;return this;},end(bytes){result.bytes=bytes;return this;}});
  return result;
}
assert.equal(request('/volume-settings-preset-schema-v2.json').status,200,
  'actual cockpit absolute schema URL must reach the actual Kaminos service, not witness404');
assert.deepEqual(forwarded,['/volume-settings-preset-schema-v2.json']);
assert.equal(request('/api/runtime-config').status,200);
assert.equal(request('/api/forge-host/registry').status,200);
assert.equal(request('/foreground/index.html?volume_resolution=32').status,200);
assert.equal(forwarded.at(-1),'/index.html?volume_resolution=32');
assert.equal(request('/api/volume-cockpit-layouts','POST').status,200,
  'observed cockpit startup may write only its caller-owned layout store');
const before=forwarded.length;
for(const url of ['/lib/sf3d/weights.bin','/foreground/lib/sf3d/weights.bin','/lib/sf3d/%77eights%2ebin','/api/read?root=pipeline-runs&path=weights.bin','/api/delete-scene?name=owned','/foreground/%2f127.0.0.1:9999/api/read?root=pipeline-runs&path=weights.bin'])
  assert.equal(request(url).status,409,'unadmitted model/data/mutation route must be refused before upstream');
assert.equal(request('/api/run-pipeline','POST').status,409);
assert.equal(forwarded.length,before,'refused routes must not contact actual server');
console.log('Actual absolute APIs/static roots compose with held full-model and mutation routes.');
