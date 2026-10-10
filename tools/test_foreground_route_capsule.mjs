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
{
  const capsule='<meta content="0; url=/?volume_resolution=160&amp;volume_emitter_source_depth=0.006">';
  const report={requested:{foreground:{grid:32,sourceDepth:0.125,warmupFrames:200}},foreground:{}};
  const result=parse(capsule,report,URL);
  assert.equal(new URL(result.originalRoute).searchParams.get('volume_emitter_source_depth'),'0.006');
  assert.equal(new URL(result.route,'http://witness.invalid').searchParams.get('volume_emitter_source_depth'),'0.125',
    'caller-selected resolved source depth must reach the actual route, not silently preserve the under-resolved capsule');
  const unchanged=parse(capsule,{requested:{foreground:{grid:32}},foreground:{}},URL);
  assert.equal(new URL(unchanged.route,'http://witness.invalid').searchParams.get('volume_emitter_source_depth'),'0.006',
    'omitted source-depth request must preserve the original capsule');
}
