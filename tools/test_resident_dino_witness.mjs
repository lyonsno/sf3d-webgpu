import assert from 'node:assert/strict';
import * as witness from './resident_dino_acceptance.mjs';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
assert.equal(typeof witness.dinoPhaseDemand,'function','phase demand must be described before allocation');
const phase={name:'dino-block-0',tensors:[{name:'x',size:16,dtype:1}],isFirst:true};
const demand=witness.dinoPhaseDemand(phase);
assert.equal(demand.weightGpuBytes,32);assert.equal(demand.rangeCpuBytes,64);
assert.ok(demand.workGpuBytes>200000000,'complete native work/diagnostic buffers included');
assert.ok(witness.dinoPhaseDemand({...phase,name:'dino-block-1',isFirst:false}).requiredBytes<demand.requiredBytes);
assert.throws(()=>witness.dinoPhaseDemand({name:'dino-block-25',tensors:[]}),/phase/);
assert.equal(witness.acceptResidentDino({status:'passed',phase:'complete'}).ok,false,'blank successful summary cannot close');
const leaf=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-dino-witness-contract-'));
try{
  const output=path.join(leaf,'output'),input=path.join(leaf,'input'),bytes=Buffer.alloc(1297*1024*4);bytes.writeFloatLE(1,0);
  fs.writeFileSync(output,bytes);fs.writeFileSync(input,Buffer.alloc(3*512*512*4));
  const phaseNames=['preprocess','camera',...Array.from({length:24},(_,i)=>'dino-block-'+i),'dino-output'];
  const observation=phase=>({phase,verdict:'admitted',host:{source:'live-macos',hostname:'host',model:'Mac14,9',processor:'Apple M2 Pro',observerErrors:[],hostFreeBytes:100},
    demand:{requiredBytes:10},process:{coverage:'sampled-owned-process-tree',lastObservation:{runId:'current',rootPid:42,status:'observed',sampledAggregatePhysicalFootprintBytes:10}}});
  const valid={status:'passed',phase:'complete',source:{clean:true,revision:'head',hostname:'host'},requested:{revision:'head',weightsSha256:'weights',processBudgetBytes:100},
    backend:{vendor:'apple',isFallbackAdapter:false},artifact:{sha256:'bundle',servedSha256:'bundle',sourceRevision:'head',kitVersion:'0.1.53'},
    canonicalSource:{sha256:'weights',etag:'"weights"',byteLength:200},dino:{weightPhases:Array.from({length:25},()=>({status:'completed-retired'})),
      loadingReport:{sourceETag:'"weights"',expectedWeightBytes:200},cooperative:{status:'succeeded',schedulingMode:'cooperative',queueCompletionAuthority:'per-gpu-duty-prefix-fence',
        boundaries:[{completedItems:24,totalItems:24,actualRangeCount:24}]}},phaseObservations:phaseNames.map(observation),runId:'current',rootPid:42,
    validationError:null,budget:{cpu:{liveBytes:0},gpu:{liveBytes:0}},processObservation:{status:'observed',coverage:'sampled-owned-process-tree',sampledPeakAggregatePhysicalFootprintBytes:20},
    cleanup:{browser:{exitObserved:true},server:'closed'},evidencePaths:{input,output},output:{sha256:createHash('sha256').update(bytes).digest('hex')}};
  assert.equal(witness.acceptResidentDino(valid).ok,true,'synthetic policy fixture, not native conformance evidence');
  for(const mutate of [r=>r.backend.isFallbackAdapter=true,r=>r.artifact.servedSha256='old',r=>r.requested.revision='other',
    r=>r.dino.cooperative.boundaries[0].completedItems=23,r=>r.dino.weightPhases.pop(),r=>r.phaseObservations.pop(),
    r=>r.phaseObservations[0].process.lastObservation.runId='stale',r=>r.phaseObservations[0].host.hostFreeBytes=0,
    r=>r.phaseObservations[0].process.lastObservation.sampledAggregatePhysicalFootprintBytes=101,
    r=>r.memorySafety={reason:'original-process-stop'},r=>r.validationError='invalid',r=>r.budget.gpu.liveBytes=4,
    r=>r.dino.loadingReport.sourceETag='"other"',r=>r.evidencePaths.output=path.join(leaf,'missing'),r=>r.output.sha256='cached']){
    const report=structuredClone(valid);mutate(report);assert.equal(witness.acceptResidentDino(report).ok,false);
  }
  fs.writeFileSync(output,Buffer.alloc(4));assert.equal(witness.acceptResidentDino(valid).ok,false,'partial raw output rejects');
  const failure=path.join(leaf,'failure.json');
  assert.throws(()=>execFileSync(process.execPath,['tools/smoke_resident_dino.mjs','--report',failure],{stdio:'pipe'}));
  const failed=JSON.parse(fs.readFileSync(failure));assert.equal(failed.status,'failed');assert.equal(failed.error.lastTrustworthyPhase,'arguments');
  const retained=Buffer.from('previous evidence');fs.writeFileSync(failure,retained);
  assert.throws(()=>execFileSync(process.execPath,['tools/smoke_resident_dino.mjs','--report',failure],{stdio:'pipe'}));
  assert.deepEqual(fs.readFileSync(failure),retained,'occupied evidence is preserved rather than relabeled');
  assert.equal(fs.readdirSync(leaf).filter(n=>n.startsWith('failure.json.refused-')).length,1);
}finally{fs.rmSync(leaf,{recursive:true});}
console.log('full encoder phase components and blank-witness refusal pass');
