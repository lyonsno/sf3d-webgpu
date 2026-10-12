import fs from 'node:fs';
import {createParentPhaseJournal,replayParentPhaseJournal} from './parent_phase_journal.mjs';
import {writeJsonReportAtomic} from './json_report_atomic.mjs';

// Hot-path observations are appended once. The running checkpoint explicitly
// omits their history; the terminal report remains complete and uncapped.
export function createNativePhaseEvidenceWriter({reportPath,report,writeReport=writeJsonReportAtomic,
  createJournal=createParentPhaseJournal,replayJournal=replayParentPhaseJournal}){
  const journalPath=reportPath+'.phases.jsonl';
  let journal,written=0,bytes=0,diagnosticEvents=0,terminalRecorded=false,fault=null;
  const append=(type,payload)=>{
    const event=journal.append(type,payload);
    bytes+=Buffer.byteLength(JSON.stringify(event)+'\n');
    if(fs.statSync(journalPath).size!==bytes)throw Error('native phase journal partial write or changed prefix');
  };
  const persist=()=>{
    try{
      if(fault)throw fault;
      if(!journal){
        journal=createJournal({journalPath,invocationId:report.runId,requested:report.requested??{}});
        bytes=fs.statSync(journalPath).size;
        const initial=replayJournal(journalPath);
        if(!initial.integrityOk||initial.hasPartialTail||initial.invocationId!==report.runId||initial.events.length!==1)
          throw Error('native invocation journal is incomplete or mismatched');
      }
      if(fs.statSync(journalPath).size!==bytes)throw Error('native phase journal changed prefix');
      if(report.phaseObservations.length<written)throw Error('native phase history shrank');
      if(terminalRecorded&&report.phaseObservations.length!==written)throw Error('native phase appended after terminal');
      while(written<report.phaseObservations.length){
        append('phase-observation',{observation:report.phaseObservations[written]});
        written++;
      }
      if(report.terminalAt&&!terminalRecorded){
        append('terminal',{status:report.status,phase:report.phase,phaseCount:written});
        journal.close();terminalRecorded=true;
        const replay=replayJournal(journalPath),rows=replay.events.filter(e=>e.type==='phase-observation');
        if(!replay.integrityOk||replay.hasPartialTail||replay.invocationId!==report.runId||
          replay.terminal?.status!==report.status||replay.terminal?.phase!==report.phase||
          replay.terminal?.phaseCount!==written||rows.length!==written||
          replay.events.filter(e=>e.type.startsWith('dino-observation-')).length!==diagnosticEvents||
          rows.some((e,i)=>JSON.stringify(e.payload.observation)!==JSON.stringify(report.phaseObservations[i])))
          throw Error('native terminal journal differs from complete observed phase order');
      }
    }catch(error){
      fault??=error;
      try{journal?.close();}catch{}
      report.status='failed';
      report.persistenceFailure={phase:'native-phase-journal',message:String(fault.message??fault),
        lastTrustworthyPhase:report.phase,writtenPhaseCount:written,journalPath};
      if(report.verdict)report.verdict={...report.verdict,ok:false,errors:[...(report.verdict.errors??[]),report.persistenceFailure.message]};
    }
    report.evidencePaths.phaseJournal=journalPath;
    report.phaseJournal={schema:'sf3d.native-phase-journal-reference.v0',path:journalPath,
      status:fault?'failed':report.terminalAt?'complete':'running',retention:'uncapped',
      writtenPhaseCount:written,observedPhaseCount:report.phaseObservations.length,
      diagnosticEventCount:diagnosticEvents,
      verification:terminalRecorded&&!fault?'terminal-hash-chain-and-complete-observation-order':'not-terminally-verified',
      authority:report.terminalAt&&!fault?'complete retained observations; not model-success evidence':
        'running checkpoint excludes history; complete observations remain in source-linked journal'};
    const checkpoint=report.terminalAt?report:{...report,phaseObservations:undefined,
      ...(report.requested?.observeDino===true&&report.dinoDiagnostics?{dinoDiagnostics:{...report.dinoDiagnostics,observations:undefined,
        observedObservationCount:report.dinoDiagnostics.observations.length,
        diagnosticEventCount:diagnosticEvents,journalPath}}:{})};
    writeReport(reportPath,checkpoint);
    if(fault)throw fault;
  };
  persist.appendDiagnostic=(type,payload)=>{
    if(terminalRecorded)throw Error('diagnostic event after terminal');
    if(fault)throw fault;
    if(report.requested?.observeDino!==true||!type.startsWith('dino-observation-'))
      throw Error('unselected diagnostic journal route');
    if(!journal)persist();
    try{append(type,payload);diagnosticEvents++;}
    catch(error){fault??=error;throw error;}
  };
  return persist;
}
