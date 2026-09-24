#!/usr/bin/env node
import assert from 'node:assert/strict';
import { assembleBrowserKitIdentity } from './browser_kit_identity.js';

const shared = {
  exportedVersion: '0.1.52',
  exportNames: ['WEBGPU_INFERENCE_KIT_VERSION', 'createWebGpuForegroundService'],
  witnessModuleUrl: 'http://127.0.0.1:4173/tools/browser_kit_identity.js',
};
const before = await assembleBrowserKitIdentity({
  ...shared,
  servedModules: [{ url: '/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc', bytes: 20, sha256: 'a'.repeat(64) }],
});
const after = await assembleBrowserKitIdentity({
  ...shared,
  servedModules: [{ url: '/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc', bytes: 20, sha256: 'b'.repeat(64) }],
});
assert.equal(before.exportFingerprint, after.exportFingerprint, 'same-version unchanged export surface keeps the API fingerprint');
assert.notEqual(before.servedModuleSetSha256, after.servedModuleSetSha256, 'changed served implementation bytes change the effective module identity');
assert.equal(before.servedModuleCount, 1);
console.log('browser-kit identity binds same-version/same-export served implementation changes');
