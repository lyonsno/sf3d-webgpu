import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {acceptResidentForeground} from './resident_foreground_service.mjs';
const index=process.argv.indexOf('--runtime-config');
assert.ok(index>=0&&process.argv[index+1],'caller-selected actual observed runtime-config required');
const filename=process.argv[index+1],bytes=fs.readFileSync(filename),effective=JSON.parse(bytes),source=effective.source;
assert.equal(effective.schema,'kaminos.runtime-config.v0');
assert.equal(typeof source.repoRoot,'string');assert.match(source.commit,/^[a-f0-9]{40}$/);assert.equal(source.dirty,false);
const report={requested:{foreground:{repoRoot:source.repoRoot,revision:source.commit}},
  foreground:{source:{repoRoot:source.repoRoot,revision:source.commit,trackedClean:true},
    service:{route:'owned-actual-kaminos-handler.v0',effective}}};
const errors=acceptResidentForeground(report).errors;
assert.equal(errors.includes('actual foreground handler identity missing'),false,
  'actual observed K2620 nested runtime-config source must not be rejected');
assert.ok(errors.length>0,'an observed source identity alone must not admit a full native composition');
for(const mutate of [s=>s.repoRoot='/wrong',s=>s.commit='a'.repeat(40),s=>s.dirty=true,s=>delete s.commit]){
  const bad=structuredClone(report);mutate(bad.foreground.service.effective.source);
  assert.ok(acceptResidentForeground(bad).errors.includes('actual foreground handler identity missing'));
}
console.log(JSON.stringify({status:'passed-source-conformance-only',runtimeConfig:filename,
  sha256:createHash('sha256').update(bytes).digest('hex'),repoRoot:source.repoRoot,revision:source.commit}));
