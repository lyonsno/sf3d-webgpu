#!/usr/bin/env node
/**
 * Parity evidence admission (tools/parity_compare_core.mjs judgeParityEvidence).
 *
 * r3 MEDIUM (2026-09-16): a successful parity report could be marked
 * evidentiary with required WebGPU stages missing (the page stops exposing a
 * stage → compareStages lists it under missing.webgpu → report still said
 * evidentiary: true). Admission now requires verified provenance, a parity
 * object, every required stage compared on both sides, the four element-wise
 * comparisons and the one stats-only comparison.
 */
import assert from 'node:assert/strict';
import { PARITY_REQUIRED_STAGE_IDS, judgeParityEvidence } from './parity_compare_core.mjs';

assert.deepEqual(PARITY_REQUIRED_STAGE_IDS, { elementwise: ['density', 'vertex_offset', 'grid_positions', 'camera_embed'], statsOnly: ['scene_codes'] });

function completeParity() {
  const stages = {};
  for (const id of PARITY_REQUIRED_STAGE_IDS.elementwise) stages[id] = { stageId: id, mode: 'elementwise' };
  for (const id of PARITY_REQUIRED_STAGE_IDS.statsOnly) stages[id] = { stageId: id, mode: 'stats-only' };
  return { stages, summary: { compared: 4, statsOnly: 1 }, missing: { webgpu: [], reference: [] } };
}
const provenanceOk = { ok: true };

// Complete comparison with verified provenance → evidentiary.
{
  const a = judgeParityEvidence({ provenance: provenanceOk, parity: completeParity() });
  assert.deepEqual(a, { evidentiary: true, reasons: [] });
  console.log('ok  complete comparison with verified provenance is evidentiary');
}
const falsifiers = [
  ['parity null', { provenance: provenanceOk, parity: null }, /no parity comparison/],
  ['provenance not verified', { provenance: { ok: false, errors: ['x'] }, parity: completeParity() }, /reference provenance not verified/],
  ['provenance missing', { provenance: null, parity: completeParity() }, /reference provenance not verified/],
  ['webgpu stage missing', { provenance: provenanceOk, parity: { ...completeParity(), missing: { webgpu: ['vertex_offset'], reference: [] } } }, /WebGPU side missing required stage\(s\): vertex_offset/],
  ['reference stage missing', { provenance: provenanceOk, parity: { ...completeParity(), missing: { webgpu: [], reference: ['camera_embed'] } } }, /reference side missing required stage\(s\): camera_embed/],
  ['stage absent from stages map', { provenance: provenanceOk, parity: (() => { const p = completeParity(); delete p.stages.density; return p; })() }, /required stage density was not compared/],
  ['element-wise count short', { provenance: provenanceOk, parity: { ...completeParity(), summary: { compared: 3, statsOnly: 1 } } }, /3 element-wise comparisons completed, expected 4/],
  ['stats-only count short', { provenance: provenanceOk, parity: { ...completeParity(), summary: { compared: 4, statsOnly: 0 } } }, /0 stats-only comparisons completed, expected 1/],
];
for (const [name, input, re] of falsifiers) {
  const a = judgeParityEvidence(input);
  assert.equal(a.evidentiary, false, `${name} must not be evidentiary`);
  assert.match(a.reasons.join('\n'), re, `${name} names its reason`);
  console.log(`ok  not evidentiary: ${name}`);
}
console.log('\nPARITY EVIDENCE ADMISSION CONTRACT PASSED');
