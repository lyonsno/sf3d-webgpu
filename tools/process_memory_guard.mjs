// Adapted from Kaminos models/trellis2/process-memory.mjs @ 5c9a2fc2939e0a28ac3560a1e9108e81ba245fa7.
// Source provenance retained; SF3D schema/name adaptation, not a new physical-capacity claim.
import fs from 'node:fs/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {randomUUID} from 'node:crypto';
const execute=promisify(execFile);
export async function startProcessMemory({python,script,rootPid=process.pid,rawPath,summaryPath=rawPath+'.summary.json',runId=randomUUID(),periodMs=1000,probe,maxFootprintBytes,onUnsafe}={}){
  if(!rawPath||!Number.isSafeInteger(rootPid)||rootPid<1||!Number.isFinite(periodMs)||periodMs<=0)
    throw Error('explicit memory output/owner and positive sample interval required');
  if(maxFootprintBytes!==undefined){
    if(!Number.isSafeInteger(maxFootprintBytes)||maxFootprintBytes<1)throw TypeError('positive caller-selected process memory budget required');
    if(typeof onUnsafe!=='function')throw TypeError('budgeted process observation requires onUnsafe stop action');
  }
  const collect=probe??(async()=>{
    const {stdout,stderr}=await execute(python,[script,'--root-pid',String(rootPid),'--run-id',runId],{maxBuffer:Infinity});
    return {stdout,stderr,exitStatus:0};
  });
  const summary={schema:'sf3d.process-memory.v0',status:'running',runId,rootPid,periodMs,sampleCount:0,
    sampledPeakAggregatePhysicalFootprintBytes:null,processes:{},rawPath,summaryPath,
    unavailableProcessObservations:[],coverage:'sampled-owned-process-tree',
    meaning:'sampled simultaneous owned-process footprint including observer; per-process kernel peaks are not summed; not machine-capacity certification'};
  await fs.writeFile(rawPath,'');let pending,stopped=false,timer,failed,safetyStopped=false;
  const persist=()=>fs.writeFile(summaryPath,JSON.stringify(summary,null,2)+'\n');
  const stopUnsafe=async reason=>{
    if(maxFootprintBytes===undefined||safetyStopped)return;
    safetyStopped=true;clearInterval(timer);
    summary.safety={...reason,maxFootprintBytes,atUnixMs:Date.now(),actionStatus:'requested'};
    // Report I/O must not be a predecessor of the safety intervention.
    // Preserve the logging failure in memory and still perform the stop.
    try{await persist();}catch(error){summary.safety.reportError=error.message;}
    try{await onUnsafe(summary.safety);summary.safety.actionStatus='returned';}
    catch(error){summary.safety.actionStatus='failed';summary.safety.actionError=error.message;throw error;}
    finally{await persist();}
  };
  const sample=async()=>{
    let row,transport;
    try{
      const result=await collect();
      if(typeof result?.stdout==='string')transport={stdout:result.stdout,stderr:result.stderr??'',exitStatus:result.exitStatus??0};
      else row=result;
    }catch(e){
      transport={stdout:e.stdout??'',stderr:e.stderr??'',exitStatus:e.code??null,signal:e.signal??null,error:e.message};
    }
    if(transport){
      try{row=JSON.parse(transport.stdout);}catch{
        row={status:'unavailable',error:transport.error??'malformed process-memory probe output'};
      }
      row={...row,transport};
    }
    if(!row||typeof row!=='object'||Array.isArray(row))row={status:'unavailable',error:'missing or malformed probe report',output:row};
    // Preserve the complete attempted observation before admitting it to any arithmetic.
    await fs.appendFile(rawPath,JSON.stringify(row??{status:'unavailable',error:'missing probe output'})+'\n');
    summary.lastObservation=row;
    for(const missing of row.unavailableProcesses??[]){
      const exit=missing.exitEvidence;
      const retired=missing.pid!==rootPid&&missing.errno===3&&missing.exitedBeforeMeasurement===true&&
        exit?.route==='ps-pid-status'&&!exit.stderr&&typeof exit.status==='string'&&
        ((exit.returnCode===1&&exit.status==='')||(exit.returnCode===0&&exit.status.startsWith('Z')));
      if(!missing.expectedProbeExit&&!retired){
      summary.unavailableProcessObservations.push({atUnixMs:row.atUnixMs,...missing});summary.coverage='partial-process-coverage';
      }
    }
    if((transport&&transport.exitStatus!==0)||row.runId!==runId||row.rootPid!==rootPid||row.status!=='observed'||!Array.isArray(row.processes)||
      !row.processes.some(p=>p.pid===rootPid)||!Number.isSafeInteger(row.sampledAggregatePhysicalFootprintBytes)||row.sampledAggregatePhysicalFootprintBytes<1)
      throw Error('observed current-owner process-memory sample required; stale/missing output is not zero memory'+(row.error?': '+row.error:''));
    for(const p of row.processes)if(!Number.isSafeInteger(p.physicalFootprintBytes)||p.physicalFootprintBytes<0||
      !Number.isSafeInteger(p.kernelLifetimePeakPhysicalFootprintBytes)||p.kernelLifetimePeakPhysicalFootprintBytes<0||!p.processStartAbstime)
      throw Error('actual process footprint/start identity required');
    if(row.processes.reduce((total,p)=>total+p.physicalFootprintBytes,0)!==row.sampledAggregatePhysicalFootprintBytes)
      throw Error('process-memory aggregate disagrees with observed process rows');
    summary.sampleCount++;
    summary.sampledPeakAggregatePhysicalFootprintBytes=Math.max(summary.sampledPeakAggregatePhysicalFootprintBytes??0,row.sampledAggregatePhysicalFootprintBytes);
    for(const p of row.processes){
      const key=p.pid+':'+p.processStartAbstime,old=summary.processes[key];
      summary.processes[key]={...p,observedPeakPhysicalFootprintBytes:Math.max(old?.observedPeakPhysicalFootprintBytes??0,p.physicalFootprintBytes),
        kernelLifetimePeakPhysicalFootprintBytes:Math.max(old?.kernelLifetimePeakPhysicalFootprintBytes??0,p.kernelLifetimePeakPhysicalFootprintBytes)};
    }
    if(maxFootprintBytes!==undefined){
      if(summary.coverage!=='sampled-owned-process-tree'){
        await stopUnsafe({reason:'memory-observation-unavailable',error:'partial process coverage'});
        throw Error('bounded memory observation unavailable: partial process coverage');
      }
      if(row.sampledAggregatePhysicalFootprintBytes>maxFootprintBytes){
        await stopUnsafe({reason:'process-footprint-budget',observedBytes:row.sampledAggregatePhysicalFootprintBytes});
        throw Error('SF3D process memory budget exceeded: '+row.sampledAggregatePhysicalFootprintBytes+' > '+maxFootprintBytes);
      }
    }
  };
  const request=()=>{
    if(stopped||failed||pending)return pending;
    pending=sample().catch(async e=>{
      failed=e;summary.status=summary.safety?.reason==='process-footprint-budget'?'budget-refused':'failed';summary.error=e.message;clearInterval(timer);
      try{await stopUnsafe({reason:'memory-observation-unavailable',error:e.message});}catch(actionError){summary.stopActionError=actionError.message;}
      await persist();
    }).finally(()=>pending=null);
    return pending;
  };
  try{await request();if(failed)throw failed;}catch(e){e.memorySummary=summary;throw e;}
  timer=setInterval(request,periodMs);
  return{sample:request,async stop(){if(stopped)return summary;await request();stopped=true;clearInterval(timer);if(pending)await pending;
    summary.status=failed?(summary.safety?.reason==='process-footprint-budget'?'budget-refused':'failed'):'observed';await persist();return summary;}};
}
