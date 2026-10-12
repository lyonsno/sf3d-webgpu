import assert from 'node:assert/strict';
import fs from 'node:fs';import os from 'node:os';import path from 'node:path';import {spawnSync} from 'node:child_process';
for(const selected of [true,false]){
 const output=fs.mkdtempSync(path.join(os.tmpdir(),'projection-rows-cli-')),report=path.join(output,'failure.json');
 try{const args=['--import','./tools/wgsl-raw-loader-register.mjs','tools/smoke_resident_dino.mjs','--report',report,
  '--through-backbone','--process-budget-bytes','2147483648'];
  if(selected)args.push('--reuse-attention-projection-storage');
  const result=spawnSync(process.execPath,args,{encoding:'utf8'});
  assert.equal(result.status,1);const r=JSON.parse(fs.readFileSync(report));
  assert.equal(r.requested.reuseAttentionProjectionStorage,selected,'actual native parser must preserve selected projection storage route and default');
  assert.equal(r.requested.processBudgetBytes,2147483648);assert.equal(r.requested.cpuBytes,268435456);
  assert.equal(r.requested.gpuBytes,1073741824);assert.equal(r.requested.totalBytes,1342177280);
  assert.equal(r.cleanup.browser.status,'not-started','malformed selection cannot launch browser/GPU');
 }finally{fs.rmSync(output,{recursive:true});}
}
console.log('PASS actual projection storage CLI requested/default route and unchanged guard constants; invalid invocation leaves durable report, no browser');
