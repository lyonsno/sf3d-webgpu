import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {measureJsonUtf8} from './foreground_result_transfer.mjs';
import {createForegroundResultReceiver} from './native_foreground_result.mjs';
const arg=name=>process.argv[process.argv.indexOf(name)+1];
if(!process.argv.includes('--observed-report')||!process.argv.includes('--output'))throw Error('explicit observed real report/output required');
const input=path.resolve(arg('--observed-report')),out=path.resolve(arg('--output'));
fs.mkdirSync(out,{recursive:true});const report={schema:'sf3d.foreground-result-observed-conformance.v0',status:'running',phase:'read-observed-source',input,
 claim:'complete real retained Kit/foreground JSON transport/restoration conformance; CPU only, no new model/native fit'};
const persist=()=>fs.writeFileSync(out+'/report.json',JSON.stringify(report,null,2)+'\n');persist();
try{
 const raw=fs.readFileSync(input),source=JSON.parse(raw);report.source={sha256:createHash('sha256').update(raw).digest('hex'),revision:source.source.revision,runId:source.runId};
 assert.equal(source.source.revision,'9e6e5143edb1c35486e57d2a3fe94f937ea865fc');assert.equal(source.foregroundRun.runId,source.runId);
 report.phase='measure-and-stream-complete-original-result';persist();
 const value=source.foregroundRun,bytes=Buffer.from(JSON.stringify(value)),measured=measureJsonUtf8(value);assert.equal(measured,bytes.length);
 const receiver=createForegroundResultReceiver({filename:out+'/full-original-foreground.json',runId:source.runId});
 await receiver.admit({runId:source.runId,bytes:measured},async(_phase,demand)=>({verdict:'admitted',demand,authority:'CPU replay only; not live process admission'}));
 const ref=await receiver.receive(bytes,{runId:source.runId,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});
 const restored=await receiver.restore({exitObserved:true,ownedPid:123},123);
 assert.deepEqual(restored,value);assert.equal(restored.frames.length,9171);assert.equal(restored.serviceReport.receipts.length,value.serviceReport.receipts.length);
 report.scope={frames:restored.frames.length,receipts:restored.serviceReport.receipts.length,services:restored.serviceReport.services.length,bytes:measured};
 report.reference=ref;report.status='passed';report.phase='complete';
}catch(error){report.status='failed';report.error={message:error.message,stack:error.stack,lastTrustworthyPhase:report.phase};}
report.terminalAt=new Date().toISOString();persist();console.log(JSON.stringify({status:report.status,scope:report.scope,error:report.error?.message,report:out+'/report.json'}));process.exitCode=report.status==='passed'?0:1;
