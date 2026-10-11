import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('./smoke_resident_dino.mjs',import.meta.url),'utf8');
const start=source.indexOf('\n',source.indexOf('report.browserVersion=await browser.version();'))+1,end=source.indexOf('  if(foregroundRoot){',start);
assert.ok(start>=0&&end>start,'actual native page acquisition required');
const body=source.slice(start,end),events=[];
const page={url:()=> 'about:blank',browser:()=>browser,async setViewport(value){events.push(['viewport',value]);}};
const browser={async pages(){events.push(['pages']);return [page];},async newPage(){throw Error('unnecessary second native tab');}};
const report={};
await vm.runInNewContext('(async()=>{'+body+'})()',{browser,report,Error});
assert.deepEqual(JSON.parse(JSON.stringify(events)),[['pages'],['viewport',{width:1280,height:960}]]);
assert.equal(report.browserPage.route,'exact-owned-initial-blank-page');
assert.equal(report.browserPage.initialPageCount,1);
for(const kind of ['missing','multiple','nonblank','wrong-owner']){
  const rejected={...page,url:()=>kind==='nonblank'?'https://foreign.invalid/':'about:blank',browser:()=>kind==='wrong-owner'?{}:owned};
  const owned={async pages(){return kind==='missing'?[]:kind==='multiple'?[rejected,rejected]:[rejected];},async newPage(){throw Error('must not silently open fallback page');}};
  await assert.rejects(vm.runInNewContext('(async()=>{'+body+'})()',{browser:owned,report:{},Error}),/exact owned single initial blank page required/);
}
console.log('actual native assay reuses its exact owned initial blank page and refuses ambiguous fallback');
