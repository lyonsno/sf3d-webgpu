import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import {EventEmitter} from 'node:events';
import {memoryStopAction} from './owned_browser_stop.mjs';
import {writeJsonReportAtomic} from './json_report_atomic.mjs';
import {createParentPhaseJournal} from './parent_phase_journal.mjs';

const testParent=path.join(os.homedir(),'.local/state/sf3d');
fs.mkdirSync(testParent,{recursive:true});
const fixture=fs.mkdtempSync(path.join(testParent,'test-native-phase-journal-'));
const nativeSource=fs.readFileSync(new URL('./smoke_resident_dino.mjs',import.meta.url),'utf8');
const declaration=nativeSource.match(/const persist=([^;]+);/)?.[0];
assert.ok(declaration,'actual native persistence declaration must be exercised');
const modulePath=new URL('./native_phase_journal.mjs',import.meta.url);
const implementation=fs.existsSync(modulePath)?await import(modulePath.href):{};
const report={schema:'fixture-full-native',status:'running',phase:'arguments',runId:'full-native-fixture',requested:{appendPhaseJournal:true},
  evidencePaths:{report:path.join(fixture,'report.json')},phaseObservations:[]};
const reportPath=report.evidencePaths.report;
const context={reportPath,report,writeJsonReportAtomic,...implementation};
const persist=vm.runInNewContext(declaration+';persist;',context);
let counts=[0,0,0];
try{
  persist();
  for(let i=0;i<counts.length;i++){
    report.phaseObservations.push({phase:'complete-phase-'+i,toJSON(){counts[i]++;return {phase:'complete-phase-'+i,descriptor:{full:true},raw:'uncapped-'+i};}});
    report.phase='phase-complete-phase-'+i;persist();
  }
  for(let i=0;i<4;i++){report.readbackBytes=(i+1)*524288;persist();}
  assert.deepEqual(counts,[1,1,1],'running persistence must serialize every complete observation exactly once, not rewrite its growing history');
  const live=JSON.parse(fs.readFileSync(reportPath,'utf8'));
  assert.equal(live.phaseObservations,undefined,'live checkpoint must not impersonate complete history');
  assert.equal(live.phaseJournal.writtenPhaseCount,3);
  assert.equal(live.phaseJournal.retention,'uncapped');
  const rows=fs.readFileSync(live.evidencePaths.phaseJournal,'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(rows.length,4,'the reused canonical journal also preserves invocation identity');
  const observed=rows.filter(r=>r.type==='phase-observation');
  assert.deepEqual(observed.map(r=>r.payload.observation.phase),['complete-phase-0','complete-phase-1','complete-phase-2']);
  assert.ok(rows.every(r=>r.invocationId===report.runId&&r.schema==='sf3d.parent-phase-journal-event.v0'));
  report.status='failed';report.phase='primary-consumer-failure';report.error={message:'real failure before GLB',lastTrustworthyPhase:report.phase};report.terminalAt='2026-10-11T00:00:00.000Z';
  persist();
  const terminal=JSON.parse(fs.readFileSync(reportPath,'utf8'));
  assert.equal(terminal.status,'failed');assert.equal(terminal.error.message,'real failure before GLB');
  assert.equal(terminal.phaseObservations.length,3);assert.equal(terminal.phaseJournal.status,'complete');
  assert.equal(terminal.phaseJournal.verification,'terminal-hash-chain-and-complete-observation-order');
  assert.deepEqual(counts,[3,3,3],'terminal verifies and consolidates complete observations only after the hot path ends');
  assert.equal(terminal.readbackBytes,4*524288,'all readback progress survives terminal consolidation');
  function fresh(label,options={}){
    const output=path.join(fixture,label+'.json');
    const value={schema:'native-fixture',runId:label,status:'running',phase:'before-real-consumer',
      requested:{appendPhaseJournal:true},evidencePaths:{report:output},phaseObservations:[]};
    return {value,output,writer:implementation.createNativePhaseEvidenceWriter({reportPath:output,report:value,...options})};
  }
  for(const damage of ['truncate','append','same-byte-tamper','changed-observation','shrink']){
    const x=fresh(damage);x.writer();x.value.phaseObservations.push({phase:'phase-a',full:'retained'});x.writer();
    const journal=x.value.evidencePaths.phaseJournal;
    if(damage==='truncate')fs.truncateSync(journal,fs.statSync(journal).size-1);
    if(damage==='append')fs.appendFileSync(journal,'unexpected');
    if(damage==='same-byte-tamper')fs.writeFileSync(journal,fs.readFileSync(journal,'utf8').replace('phase-a','phase-b'));
    if(damage==='changed-observation')x.value.phaseObservations[0].phase='phase-b';
    if(damage==='shrink')x.value.phaseObservations=[];
    x.value.status='passed';x.value.verdict={ok:true,errors:[]};x.value.terminalAt='terminal';
    assert.throws(x.writer,/journal|history/);
    const failed=JSON.parse(fs.readFileSync(x.output,'utf8'));
    assert.equal(failed.status,'failed');assert.equal(failed.verdict.ok,false);
    assert.equal(failed.phaseJournal.status,'failed');assert.ok(failed.persistenceFailure.lastTrustworthyPhase);
    assert.deepEqual(failed.phaseObservations,x.value.phaseObservations,'terminal preserves even the disputed complete input');
  }
  const partial=fresh('partial',{createJournal:args=>{
    const actual=createParentPhaseJournal(args);
    return {...actual,append(type,payload){const event=actual.append(type,payload);
      fs.truncateSync(actual.path,fs.statSync(actual.path).size-1);return event;}};
  }});
  partial.writer();partial.value.phaseObservations.push({phase:'never-admitted-consumer'});
  assert.throws(partial.writer,/partial write/);
  const partialReport=JSON.parse(fs.readFileSync(partial.output,'utf8'));
  assert.equal(partialReport.status,'failed');assert.equal(partialReport.phaseJournal.writtenPhaseCount,0);
  assert.equal(partialReport.phaseObservations,undefined,'failed hot checkpoint still does not counterfeit history');
  const owned=new EventEmitter();Object.assign(owned,{pid:12345,exitCode:null,signalCode:null,
    kill(signal){assert.equal(signal,'SIGTERM');queueMicrotask(()=>{owned.exitCode=0;owned.emit('exit',0,null);});return true;}});
  await assert.rejects(memoryStopAction({child:()=>owned,report:partial.value,persist:partial.writer})({reason:'process-footprint-budget'}),/partial write/);
  assert.equal(owned.exitCode,0,'journal failure must not bypass exact owned browser stop');
  const stopped=JSON.parse(fs.readFileSync(partial.output,'utf8'));
  assert.equal(stopped.memorySafety.stop.exitObserved,true);assert.equal(stopped.memorySafety.stop.ownedPid,12345);
  const intact=fs.readFileSync(journalPathFrom(live));
  const reused=implementation.createNativePhaseEvidenceWriter({reportPath,report:{...report,status:'running',terminalAt:undefined}});
  assert.throws(reused,/EEXIST/);assert.deepEqual(fs.readFileSync(journalPathFrom(live)),intact,'occupied journal must never be replaced');
  function journalPathFrom(value){return value.evidencePaths.phaseJournal;}
  const ordinary={schema:'unchanged-default',status:'running',phase:'ordinary',runId:'ordinary',requested:{appendPhaseJournal:false},evidencePaths:{},phaseObservations:[{phase:'existing'}]};
  const ordinaryPath=path.join(fixture,'ordinary.json');
  const ordinaryPersist=vm.runInNewContext(declaration+';persist;',{report:ordinary,reportPath:ordinaryPath,writeJsonReportAtomic,
    createNativePhaseEvidenceWriter(){assert.fail('unselected default must never switch persistence route');}});
  ordinaryPersist();assert.deepEqual(JSON.parse(fs.readFileSync(ordinaryPath,'utf8')).phaseObservations,ordinary.phaseObservations);
  assert.equal(fs.existsSync(ordinaryPath+'.phases.jsonl'),false);
  console.log('Actual native append-only phase persistence/full terminal failure contracts passed');
}finally{fs.rmSync(fixture,{recursive:true,force:true});}
