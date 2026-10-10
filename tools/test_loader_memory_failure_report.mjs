import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const root=path.resolve(new URL('..',import.meta.url).pathname),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-loader-failure-'));
for(const [name,args,phase]of [
  ['missing arguments',[],'arguments'],
  ['wrong source',['--repo-root',root,'--expected-revision','b'.repeat(40),'--chrome','/nonexistent/chrome','--process-budget-bytes','1'],'source-identity'],
  ['canonical missing allowances',['--canonical-tensor-unit','--repo-root',root,'--expected-revision','b'.repeat(40),'--chrome','/nonexistent/chrome','--process-budget-bytes','1'],'arguments'],
]){
  const report=path.join(tmp,phase+'.json');
  const run=spawnSync(process.execPath,[path.join(root,'tools/smoke_loader_memory.mjs'),...args,'--report',report],{encoding:'utf8'});
  assert.equal(run.status,1,name+' refuses');const receipt=JSON.parse(fs.readFileSync(report));
  assert.equal(receipt.phase,phase);assert.equal(receipt.status,'failed');assert.ok(receipt.error.message);
  assert.equal(receipt.ownedBrowserPid,undefined,'no browser before prerequisites');assert.equal(receipt.verdict.ok,false);
  assert.equal(receipt.reportPath,report);assert.ok(receipt.evidencePaths.process);assert.ok(receipt.terminalAt);
}
console.log('Native loader witness failures before primary output leave a durable phase, last evidence, and exact output paths.');
