import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('./smoke_loader_memory.mjs',import.meta.url),'utf8');
const start=source.indexOf("report.phase='native-process-stop'"),end=source.indexOf("report.status='passed';",start);
const sequence=vm.compileFunction('return(async()=>{'+source.slice(start,end)+'})();',
  ['report','monitor','persist','startProcessMemory','script','reportPath','stop']);
let controls=0;
const observed={status:'budget-refused',safety:{reason:'process-footprint-budget'},lastObservation:{sampledAggregatePhysicalFootprintBytes:1192699536}};
await assert.rejects(sequence({runId:'observed-native',observer:{python:'/fixture'}},{stop:async()=>observed},async()=>{},
  async()=>{controls++;throw Object.assign(Error('fixture refusal'),{memorySummary:{safety:{reason:'process-footprint-budget'}}});},'/fixture','/fixture',()=>{}),
  /consumer memory guard did not complete normally/);
assert.equal(controls,0,'actual threshold failure cannot enter or overwrite evidence with a diagnostic control');
console.log('Actual consumer memory failure is preserved before any deliberate refusal control.');
