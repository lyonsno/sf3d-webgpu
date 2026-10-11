import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import {EventEmitter} from 'node:events';
import {createNativePhaseEvidenceWriter} from './native_phase_journal.mjs';
import {stopOwnedBrowser} from './owned_browser_stop.mjs';

const source=fs.readFileSync(new URL('./smoke_resident_dino.mjs',import.meta.url),'utf8');
const start=source.indexOf('server=http.createServer(async(req,res)=>{');
const end=source.indexOf('\n  await new Promise(resolve=>server.listen',start);
assert.ok(start>0&&end>start,'exercise the actual HTTP handler');
const observeDeclaration=source.match(/const observe=async phase=>\{[\s\S]*?\};/)?.[0];
assert.ok(observeDeclaration,'exercise the actual browser phase refusal');
const catchStart=source.lastIndexOf('}catch(error){report.status=');
const cleanupEnd=source.indexOf('  report.verdict=',catchStart);
assert.ok(catchStart>0&&cleanupEnd>catchStart,'exercise the actual outer failure and cleanup');
const parent=path.join(os.homedir(),'.local/state/sf3d');
fs.mkdirSync(parent,{recursive:true});
const fixture=fs.mkdtempSync(path.join(parent,'test-native-http-failure-'));
try{
  for(const mode of ['sticky-journal','secondary-report']){
    const reportPath=path.join(fixture,mode+'.json');
    const report={status:'running',phase:'native-dino',runId:mode,requested:{appendPhaseJournal:true},
      evidencePaths:{report:reportPath},phaseObservations:[]};
    const writer=createNativePhaseEvidenceWriter({reportPath,report});
    writer();
    // Real sticky journal fault: change the uncapped source prefix, not a mock
    // that merely assumes the implementation's throwing behavior.
    fs.appendFileSync(report.evidencePaths.phaseJournal,'damaged');
    let calls=0,handler;
    const persist=()=>{calls++;if(mode==='secondary-report'&&calls===2)throw Error('secondary report unavailable');return writer();};
    vm.runInNewContext(source.slice(start,end),{
      http:{createServer(callback){handler=callback;return {};}},
      foregroundRoot:null,
      URL,report,persist,readBody:async()=>Buffer.from(JSON.stringify({vendor:'apple',isFallbackAdapter:false})),
    });
    let status,body;
    const response={writeHead(code){status=code;return this;},end(value){body=value;return this;}};
    let escaped;
    try{await handler({url:'/backend',method:'POST'},response);}catch(error){escaped=error;}
    assert.equal(escaped,undefined,'journal fault must not escape the unowned async HTTP callback');
    assert.equal(status,409,'latched journal failure must deliver HTTP refusal instead of rejecting an unowned handler');
    assert.match(body,/journal changed prefix/,'the primary journal fault reaches the caller');
    assert.equal(calls,2,'one secondary report attempt, never silent fallback');
    assert.equal(report.status,'failed');
    assert.match(report.persistenceFailure.message,/journal changed prefix/);
    assert.match(report.lastServerError.reportError,mode==='secondary-report'?/secondary report unavailable/:/journal changed prefix/);

    const fetch=async()=>({ok:false,text:async()=>body});
    const observe=vm.runInNewContext(observeDeclaration+';observe;',{fetch,JSON,Error});
    let hostStopped=false,processStopped=false,serverClosed=false,sourceClosed=false;
    const child=new EventEmitter();
    Object.assign(child,{pid:12345,exitCode:null,signalCode:null,
      kill(signal){assert.equal(signal,'SIGTERM');queueMicrotask(()=>{child.exitCode=0;child.emit('exit',0,null);});return true;}});
    const cleanup=vm.runInNewContext('(async()=>{try{await pageEvaluate();'+source.slice(catchStart,cleanupEnd)+'}})',{
      report,pageEvaluate:()=>observe({name:'real-phase'}),
      foregroundRoot:null,foregroundService:null,displayChild:null,
      hostMonitor:{async stop(){hostStopped=true;return {status:'stopped'};}},safetyStop:null,
      monitor:{async stop(){processStopped=true;return {status:'observed'};}},
      child,stopOwnedBrowser,server:{close(done){serverClosed=true;done();}},
      closeCanonicalSource(){sourceClosed=true;},source:{},profile:null,fs,
    });
    await cleanup();
    assert.match(report.error.message,/fresh phase observation refused: native phase journal changed prefix/);
    assert.equal(report.status,'failed');
    assert.deepEqual([hostStopped,processStopped,serverClosed,sourceClosed],[true,true,true,true]);
    assert.equal(report.cleanup.browser.ownedPid,12345);
    assert.equal(report.cleanup.browser.exitObserved,true,'actual outer cleanup stops the exact owned child after actual browser refusal');
    assert.equal(child.exitCode,0);
    // The sticky fault still refuses terminal persistence; its failed full
    // report is written only after owned cleanup, never converted to success.
    report.terminalAt='fixture-terminal';
    assert.throws(writer,/journal changed prefix/);
    const failed=JSON.parse(fs.readFileSync(reportPath,'utf8'));
    assert.equal(failed.status,'failed');assert.equal(failed.phaseJournal.status,'failed');
    assert.equal(failed.cleanup.browser.exitObserved,true);
    assert.equal(failed.phaseObservations.length,0);
  }
  console.log('Actual HTTP journal-fault refusal, browser failure and exact-owned outer cleanup contracts passed');
}finally{fs.rmSync(fixture,{recursive:true,force:true});}
