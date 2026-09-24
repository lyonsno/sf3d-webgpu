#!/usr/bin/env node
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { createHash } from 'node:crypto';
import { assembleBrowserKitIdentity } from './browser_kit_identity.js';
import { createBrowserExecutedKitModuleCapture, isRelevantExecutedModuleUrl } from './browser_executed_module_identity.mjs';

const shared = {
  exportedVersion: '0.1.52',
  exportNames: ['WEBGPU_INFERENCE_KIT_VERSION', 'createWebGpuForegroundService'],
  witnessModuleUrl: 'http://127.0.0.1:4173/tools/browser_kit_identity.js',
};
const before = await assembleBrowserKitIdentity({
  ...shared,
  executedModules: [{ scriptId: '1', url: '/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc', bytes: 20, sha256: 'a'.repeat(64) }],
});
const after = await assembleBrowserKitIdentity({
  ...shared,
  executedModules: [{ scriptId: '1', url: '/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc', bytes: 20, sha256: 'b'.repeat(64) }],
});
assert.equal(before.exportFingerprint, after.exportFingerprint, 'same-version unchanged export surface keeps the API fingerprint');
assert.notEqual(before.executedModuleSetSha256, after.executedModuleSetSha256, 'changed executed implementation source changes its module-set identity');
assert.equal(before.executedModuleCount, 1);
assert.equal(before.identityBasis, 'chrome-debugger-executed-module-source');

const cdp = new EventEmitter();
const scriptSource = 'export const version = "0.1.52";';
cdp.send = async (method, { scriptId }) => {
  assert.equal(method, 'Debugger.getScriptSource');
  assert.equal(scriptId, 'kit-script');
  return { scriptSource };
};
assert.equal(isRelevantExecutedModuleUrl('/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc', 'http://127.0.0.1:4173/'), true);
assert.equal(isRelevantExecutedModuleUrl('/src/main.js', 'http://127.0.0.1:4173/'), false);
const capture = createBrowserExecutedKitModuleCapture(cdp, { baseUrl: 'http://127.0.0.1:4173/' });
cdp.emit('Debugger.scriptParsed', { scriptId: 'ignored-script', url: 'http://127.0.0.1:4173/src/main.js', type: 'module' });
cdp.emit('Debugger.scriptParsed', { scriptId: 'kit-script', url: 'http://127.0.0.1:4173/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc', type: 'module' });
const captured = await capture.snapshot();
assert.equal(captured.length, 1, 'non-kit app script is not included in the kit implementation identity');
assert.equal(captured[0].sha256, createHash('sha256').update(scriptSource, 'utf8').digest('hex'), 'identity hashes source returned for the parsed Chrome scriptId, not a second URL fetch');
capture.dispose();

const missingCdp = new EventEmitter();
missingCdp.send = async () => ({ scriptSource: 'export {};' });
const missingCapture = createBrowserExecutedKitModuleCapture(missingCdp, { baseUrl: 'http://127.0.0.1:4173/' });
missingCdp.emit('Debugger.scriptParsed', { scriptId: 'dependency-only', url: 'http://127.0.0.1:4173/node_modules/.vite/deps/chunk-abc.js', type: 'module' });
await assert.rejects(() => missingCapture.snapshot(), /did not parse an @kaminos\/webgpu-inference-kit module/);
missingCapture.dispose();
console.log('browser-kit identity binds Chrome-parsed module source and rejects dependency-only capture');
