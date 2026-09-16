#!/usr/bin/env node
/** Contract for tools/cooperative_identity_acceptance.mjs (kit-validator wrapper). */
import assert from 'node:assert/strict';
import { acceptCooperativeMechanismReport, cooperativeMechanismExpectations } from './cooperative_identity_acceptance.mjs';
import { createProductRouteOptions } from '../src/lib/product_route.js';

const requested = createProductRouteOptions({});
// Expectations carry the exact identities for the product route.
const ts = cooperativeMechanismExpectations('two-stream-backbone', requested, { expectedGpuDutyCount: 2922 });
assert.equal(ts.expectedManifestId, 'sf3d.two-stream-attention-cooperative-boundaries.v0');
assert.equal(ts.expectedInvocationId, 'sf3d:two-stream:cooperative');
assert.equal(ts.expectedCompletionPolicy, 'strict-prefix');
assert.equal(ts.expectedGpuDutyCount, 2922);
assert.equal(cooperativeMechanismExpectations('texture-bake', requested).expectedGpuDutyCount, undefined);
assert.throws(() => cooperativeMechanismExpectations('post-processor', { cooperativePostProcessor: false }), /not a requested cooperative mechanism/);
console.log('ok  expectations bound to the requested route');

// The verdict comes from the kit validator: a wrong identity is named, an
// object that is not a kit report is rejected, nothing is waved through.
const wrong = acceptCooperativeMechanismReport('dinov2-tokenizer', { schema: 'nope', routeId: 'other', manifestId: 'x', invocationId: 'y', status: 'succeeded' }, requested);
assert.equal(wrong.ok, false);
assert.ok(wrong.errors.some(e => /routeId/.test(e)), 'names the route identity mismatch');
assert.ok(wrong.errors.some(e => /manifestId/.test(e)), 'names the manifest identity mismatch');
assert.equal(acceptCooperativeMechanismReport('texture-bake', null, requested).ok, false);
console.log('ok  kit validator verdict preserved (wrong identity rejected)');

console.log('\nCOOPERATIVE IDENTITY ACCEPTANCE CONTRACT PASSED');
