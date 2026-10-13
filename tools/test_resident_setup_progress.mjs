import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('./smoke_resident_dino.mjs',import.meta.url),'utf8');
const start=source.indexOf('const setForegroundSetupPhase=async name=>{');
const end=source.indexOf('\nlet source,monitor',start);
assert.ok(start>0&&end>start,'actual setup must durably distinguish mount, warmup, capture and reconnect waits before inference');
const report={foregroundSetup:{status:'not-started',phases:[]}};
let at=1000,writes=0;
const phase=vm.runInNewContext(source.slice(start,end)+';setForegroundSetupPhase;',{report,
  persist:async()=>{writes++;},Date:{now:()=>at},console:{log(){}}});
await phase('actual-foreground-mount-wait');
assert.equal(report.phase,'actual-foreground-mount-wait');
assert.equal(report.foregroundSetup.status,'running');
assert.equal(report.foregroundSetup.phases[0].startedAtUnixMs,1000);
assert.equal(report.foregroundSetup.phases[0].completedAtUnixMs,undefined,'a pending setup is not model execution or success');
at=4000;await phase('actual-foreground-warmup-wait');
assert.equal(report.foregroundSetup.phases[0].completedAtUnixMs,4000);
at=5000;await phase('native-dino');
assert.equal(report.foregroundSetup.status,'completed');
assert.equal(report.foregroundSetup.completedAtUnixMs,5000);
assert.equal(writes,3);
for(const name of ['actual-foreground-navigation','actual-foreground-mount-wait','actual-foreground-warmup-wait',
  'actual-foreground-snapshot','actual-foreground-first-capture','actual-foreground-model-reconnect','native-dino'])
  assert.ok(source.includes("setForegroundSetupPhase('"+name+"')"),'actual live path records '+name);
assert.ok(source.includes("report.foregroundSetup.status='failed'"),'setup failure remains a durable failure, not completed preparation');
console.log('PASS actual durable setup boundaries distinguish pending preparation, completed waits and model start without an execution cap');
