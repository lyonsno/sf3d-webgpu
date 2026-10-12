import assert from 'node:assert/strict';
import fs from 'node:fs';
const source=fs.readFileSync(new URL('./smoke_resident_dino.mjs',import.meta.url),'utf8');
const start=source.indexOf("report.phase='actual-foreground-post-model-capture'");
const end=source.indexOf("    if(report.foreground.pageErrors.length)",start);
assert(start>=0&&end>start);const body=source.slice(start,end);
const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;
const png=Buffer.alloc(32);png.writeUInt32BE(0x89504e47);png.writeUInt32BE(1280,16);png.writeUInt32BE(960,20);
for(const verdict of ['admitted','refused']){
 const report={runId:'owned',foreground:{displayAssertion:{},captures:{before:{sha256:'owned-png'}}},
  evidencePaths:{foregroundBefore:'/owned/before.png',foregroundAfter:'/owned/after.png'}};
 const calls=[];const page={evaluate:async()=>({sameDevice:true,sameQueue:true}),screenshot:async()=>calls.push('screenshot')};
 const freshObservation=async(phase,demand)=>{calls.push('fresh-admission');assert.equal(demand.requiredBytes,7*1280*960*4);return{verdict,demand};};
 const run=new AsyncFunction('report','persist','displayChild','page','fs','digest','freshObservation',body);
 if(verdict==='refused'){await assert.rejects(run(report,()=>{}, {exitCode:null,signalCode:null},page,{readFileSync:()=>png},()=> 'owned-png',freshObservation),/capture.*refused/);assert.deepEqual(calls,['fresh-admission']);}
 else{await run(report,()=>{}, {exitCode:null,signalCode:null},page,{readFileSync:()=>png},()=> 'owned-png',freshObservation);
  assert.deepEqual(calls,['fresh-admission','screenshot'],'actual native capture needs fresh baseline plus identified NEW before screenshot');
  assert.equal(report.foreground.captureAdmission.verdict,'admitted');}
}
console.log('PASS actual native post-model screenshot fresh prospective admission/refusal before allocation; no original guard weakening');
