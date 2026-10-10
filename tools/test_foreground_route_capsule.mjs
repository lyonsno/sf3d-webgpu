import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// Replay the actual entrypoint's parser. Whitespace and route fields observed
// in Kaminos70624e5:sf3d-elfinblue.html; this reduced fixture proves parser
// policy, while the exact committed full capsule remains the native witness.
const source=fs.readFileSync(new URL('./smoke_loader_memory.mjs',import.meta.url),'utf8');
const parser=source.match(/const match=html\.match[\s\S]*?report\.foreground\.variant=[^\n]+/)[0];
const parse=vm.compileFunction(parser+';return report.foreground;',['html','report','URL']);
const original='/?kaminos_volume_smoke=1&amp;volume_resolution=160&amp;volume_physical_mode=2&amp;settings_preset=observed&amp;settings_preset_authority=shared-volume-settings-preset-v2#composition_module_url=./sf3d-live-flame-inject.mjs';
for(const whitespace of [' ','','\t']){
  const report={requested:{foreground:{grid:32}},foreground:{}};
  const result=parse('<meta http-equiv="refresh" content="0;'+whitespace+'url='+original+'">',report,URL);
  assert.equal(result.originalGrid,160);
  const effective=new URL(result.route,'http://witness.invalid');
  assert.equal(effective.pathname,'/foreground/index.html');
  assert.equal(effective.searchParams.get('volume_resolution'),'32');
  assert.equal(effective.searchParams.get('volume_physical_mode'),'2');
  assert.equal(effective.searchParams.get('kaminos_volume_smoke'),'1');
  assert.equal(effective.searchParams.has('settings_preset'),false);
  assert.equal(effective.searchParams.has('settings_preset_authority'),false);
  assert.equal(effective.hash,'');
  assert.equal(new URL(result.originalRoute).hash,'#composition_module_url=./sf3d-live-flame-inject.mjs');
}
console.log('Observed refresh whitespace and ordinary controls survive explicit grid variant parsing.');
