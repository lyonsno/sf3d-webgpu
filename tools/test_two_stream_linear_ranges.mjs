/**
 * Contract for uncapped, exact row-range linear dispatch.
 *
 * Run:
 *   node --import ./tools/wgsl-raw-loader-register.mjs \
 *     tools/test_two_stream_linear_ranges.mjs
 */
import assert from 'node:assert/strict';

import {
  createLinearRowRanges,
  createTwoStreamAttentionDutyPlan,
  normalizeLinearRowRange,
} from '../src/lib/two_stream.js';

assert.deepEqual(
  normalizeLinearRowRange(27648, 0, 128),
  { totalRows: 27648, rowStart: 0, rowCount: 128, rowEnd: 128 },
);
assert.deepEqual(
  normalizeLinearRowRange(27648, 27616, 32),
  { totalRows: 27648, rowStart: 27616, rowCount: 32, rowEnd: 27648 },
);
assert.throws(() => normalizeLinearRowRange(27648, 27616, 33), /exceeds totalRows/);
assert.throws(() => normalizeLinearRowRange(27648, -1, 1), /rowStart/);
assert.throws(() => normalizeLinearRowRange(27648, 0, 0), /rowCount/);
console.log('ok  row normalization preserves exact global bounds');

const ranges = createLinearRowRanges(27648, 128);
assert.equal(ranges.length, 216);
assert.deepEqual(ranges[0], {
  rangeIndex: 0,
  totalRows: 27648,
  rowStart: 0,
  rowCount: 128,
  rowEnd: 128,
});
assert.deepEqual(ranges.at(-1), {
  rangeIndex: 215,
  totalRows: 27648,
  rowStart: 27520,
  rowCount: 128,
  rowEnd: 27648,
});
for (let index = 0; index < ranges.length; index++) {
  assert.equal(ranges[index].rangeIndex, index);
  assert.equal(ranges[index].rowStart, index * 128);
  assert.equal(ranges[index].rowEnd, ranges[index].rowStart + ranges[index].rowCount);
  if (index > 0) assert.equal(ranges[index - 1].rowEnd, ranges[index].rowStart);
}
assert.equal(ranges.reduce((sum, range) => sum + range.rowCount, 0), 27648);
console.log('ok  row plan covers all 27,648 rows once without a hidden cap');

const partial = createLinearRowRanges(3089, 128);
assert.equal(partial.length, 25);
assert.deepEqual(partial.at(-1), {
  rangeIndex: 24,
  totalRows: 3089,
  rowStart: 3072,
  rowCount: 17,
  rowEnd: 3089,
});
console.log('ok  final partial range is retained exactly');

const decomposedPlan = createTwoStreamAttentionDutyPlan(1297);
assert.equal(decomposedPlan.length, 4218);
assert.equal(
  decomposedPlan.filter(duty => duty.kind.endsWith('-linear-range')).length,
  2592,
);
for (const block of [0, 1, 2, 3]) {
  const prefix = `block-${block}-fuse-out`;
  assert.equal(
    decomposedPlan.filter(
      duty => duty.kind === 'fuse-attention-linear-range'
        && duty.ownerId === `${prefix}-attention-projection`,
    ).length,
    216,
  );
  assert.equal(
    decomposedPlan.filter(
      duty => duty.kind === 'fuse-geglu-linear-range'
        && duty.ownerId === `${prefix}-geglu-expansion`,
    ).length,
    216,
  );
  assert.equal(
    decomposedPlan.filter(
      duty => duty.kind === 'fuse-ffn-linear-range'
        && duty.ownerId === `${prefix}-ffn-projection`,
    ).length,
    216,
  );
}
console.log('ok  four fuse-out tails expose all 2,592 linear row duties');

const coarsePlan = createTwoStreamAttentionDutyPlan(1297, {
  linearRowsPerDuty: 256,
});
assert.equal(coarsePlan.length, 2922);
assert.equal(
  coarsePlan.filter(duty => duty.kind.endsWith('-linear-range')).length,
  1296,
);
assert.equal(
  coarsePlan.filter(duty => duty.kind === 'fuse-geglu-linear-range')[0].rowCount,
  256,
);
console.log('ok  explicit row granularity changes work without hiding rows');

console.log('\nALL TWO-STREAM LINEAR RANGE CONTRACT CHECKS PASSED');
