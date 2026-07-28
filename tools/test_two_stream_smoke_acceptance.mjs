import assert from 'node:assert/strict';

import {
  evaluateTwoStreamSmokeAcceptance,
} from './two_stream_smoke_acceptance.mjs';

const canonicalSingleArm = evaluateTwoStreamSmokeAcceptance({
  armSelection: 'candidate',
  outputIdentical: null,
  outputCanonical: true,
  cooperativeComplete: true,
  progressHonest: true,
});
assert.equal(canonicalSingleArm.paired, false);
assert.equal(canonicalSingleArm.ok, false);
console.log('ok  canonical one-arm diagnostics cannot satisfy A/B acceptance');

const canonicalPair = evaluateTwoStreamSmokeAcceptance({
  armSelection: 'both',
  outputIdentical: true,
  outputCanonical: true,
  cooperativeComplete: true,
  progressHonest: true,
});
assert.equal(canonicalPair.paired, true);
assert.equal(canonicalPair.ok, true);
console.log('ok  exact canonical paired parity satisfies A/B acceptance');

const mismatchedPair = evaluateTwoStreamSmokeAcceptance({
  armSelection: 'both',
  outputIdentical: false,
  outputCanonical: true,
  cooperativeComplete: true,
  progressHonest: true,
});
assert.equal(mismatchedPair.ok, false);
console.log('ok  mismatched paired output fails A/B acceptance');

console.log('\nALL TWO-STREAM SMOKE ACCEPTANCE CHECKS PASSED');
