import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
// Source contract: a live shared host has nonzero logical stores until the
// exact owned browser exits. This is not a claim about physical reclamation.
const source=fs.readFileSync(new URL('./resident_dino_acceptance.mjs',import.meta.url),'utf8');
const end=source.indexOf("'unretired owned resources');")+"'unretired owned resources');".length;
const start=source.lastIndexOf('  require(',end);
assert.ok(start>0&&end>start);
const check=vm.runInNewContext('(report)=>{let result;const require=ok=>{result=ok;};'+source.slice(start,end)+'return Boolean(result);}',{});
const report={requested:{foreground:{grid:32},throughFullModel:true},ownedBrowserPid:123,
  budget:{cpu:{liveBytes:20},gpu:{liveBytes:40}},foregroundRun:{status:'finished',sameDevice:true,sameQueue:true},
  foreground:{deviceRetirement:{route:'exact-owned-browser-exit',ownedPid:123}},cleanup:{browser:{exitObserved:true,ownedPid:123}}};
assert.equal(check(report),true,'observed exact owned-browser exit retires the borrowed host without fabricating logical-zero stores');
for(const change of [r=>r.cleanup.browser.exitObserved=false,r=>r.foreground.deviceRetirement.ownedPid=124,
  r=>r.ownedBrowserPid=124,r=>r.requested.foreground=null,r=>r.foregroundRun.status='failed']){
  const bad=structuredClone(report);change(bad);assert.equal(check(bad),false,'missing/mismatched/failed shared host retirement must refuse');
}
assert.equal(check({...report,requested:{},budget:{cpu:{liveBytes:0},gpu:{liveBytes:0}}}),true,'ordinary standalone zero-store invariant remains');
console.log('Actual native retirement assertion preserves nonzero live-host ledgers and requires exact owned exit');
