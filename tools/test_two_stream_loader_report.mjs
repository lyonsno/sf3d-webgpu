import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
const leaf=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-loader-report-contract-')),failure=path.join(leaf,'failure.json');
try{
  assert.throws(()=>execFileSync(process.execPath,['tools/smoke_resident_dino.mjs','--through-backbone','--repo-root',process.cwd(),
    '--expected-revision',execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),'--input','not-read.png','--chrome','not-launched',
    '--process-budget-bytes','2147483648','--report',failure],{stdio:'pipe'}));
  assert.ok(fs.existsSync(failure),'missing raw-WGSL loader must still produce a durable terminal report before any browser launch');
  const failed=JSON.parse(fs.readFileSync(failure));assert.equal(failed.status,'failed');assert.equal(failed.error.lastTrustworthyPhase,'backbone-graph-import');
  assert.match(failed.error.message,/UNKNOWN_FILE_EXTENSION|Unknown file extension/);
}finally{fs.rmSync(leaf,{recursive:true});}
console.log('raw shader loader failure produces terminal report before native/browser work');
