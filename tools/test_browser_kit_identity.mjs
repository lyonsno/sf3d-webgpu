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
const moduleCapture = {
  modules: [{ scriptId: '1', executionContextId: 4, url: '/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc', bytes: 20, sha256: 'a'.repeat(64) }],
  producerBinding: {
    executionContextId: 4, kitModuleScriptId: '1', kitModuleUrl: '/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc',
    producerModuleScriptId: '2', producerModuleUrl: '/src/lib/sf3d_producer.js',
    producerModuleExecutionContextId: 4, producerImportsKitModuleUrl: '/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc',
    identityHelperScriptId: '3', identityHelperModuleUrl: '/tools/browser_kit_identity.js',
    identityHelperExecutionContextId: 4, identityHelperImportsKitModuleUrl: '/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc',
    linkBasis: 'same-context-static-import-url',
  },
};
const before = await assembleBrowserKitIdentity({
  ...shared,
  executedModuleCapture: moduleCapture,
});
const after = await assembleBrowserKitIdentity({
  ...shared,
  executedModuleCapture: { ...moduleCapture, modules: [{ ...moduleCapture.modules[0], sha256: 'b'.repeat(64) }] },
});
assert.equal(before.exportFingerprint, after.exportFingerprint, 'same-version unchanged export surface keeps the API fingerprint');
assert.notEqual(before.executedModuleSetSha256, after.executedModuleSetSha256, 'changed executed implementation source changes its module-set identity');
assert.equal(before.executedModuleCount, 1);
assert.equal(before.identityBasis, 'chrome-debugger-executed-module-source');

const cdp = new EventEmitter();
const sources = {
  'kit-script': 'export const version = "0.1.52";',
  'producer-script': 'import "/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc";',
  'identity-helper-script': 'import "/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc";',
};
cdp.send = async (method, { scriptId }) => {
  assert.equal(method, 'Debugger.getScriptSource');
  return { scriptSource: sources[scriptId] };
};
assert.equal(isRelevantExecutedModuleUrl('/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc', 'http://127.0.0.1:4173/'), true);
assert.equal(isRelevantExecutedModuleUrl('/src/main.js', 'http://127.0.0.1:4173/'), false);
const capture = createBrowserExecutedKitModuleCapture(cdp, { baseUrl: 'http://127.0.0.1:4173/' });
cdp.emit('Debugger.scriptParsed', { scriptId: 'ignored-script', url: 'http://127.0.0.1:4173/src/main.js', isModule: true, executionContextId: 4 });
cdp.emit('Debugger.scriptParsed', { scriptId: 'kit-script', url: 'http://127.0.0.1:4173/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc', isModule: true, executionContextId: 4 });
cdp.emit('Debugger.scriptParsed', { scriptId: 'producer-script', url: 'http://127.0.0.1:4173/src/lib/sf3d_producer.js', isModule: true, executionContextId: 4 });
cdp.emit('Debugger.scriptParsed', { scriptId: 'identity-helper-script', url: 'http://127.0.0.1:4173/tools/browser_kit_identity.js', isModule: true, executionContextId: 4 });
const captured = await capture.snapshot();
assert.equal(captured.modules.length, 1, 'non-kit app scripts are not included in the kit implementation identity');
assert.equal(captured.modules[0].sha256, createHash('sha256').update(sources['kit-script'], 'utf8').digest('hex'), 'identity hashes source returned for the parsed Chrome scriptId, not a second URL fetch');
assert.equal(captured.producerBinding.linkBasis, 'same-context-static-import-url', 'capture binds the producer and identity helper to the unique kit module URL');
capture.dispose();

const missingCdp = new EventEmitter();
missingCdp.send = async () => ({ scriptSource: 'export {};' });
const missingCapture = createBrowserExecutedKitModuleCapture(missingCdp, { baseUrl: 'http://127.0.0.1:4173/' });
missingCdp.emit('Debugger.scriptParsed', { scriptId: 'dependency-only', url: 'http://127.0.0.1:4173/node_modules/.vite/deps/chunk-abc.js', isModule: true, executionContextId: 4 });
await assert.rejects(() => missingCapture.snapshot(), /did not parse an @kaminos\/webgpu-inference-kit module/);
missingCapture.dispose();

const failedCdp = new EventEmitter();
failedCdp.send = async () => { throw new Error('source retrieval denied'); };
const failedCapture = createBrowserExecutedKitModuleCapture(failedCdp, { baseUrl: 'http://127.0.0.1:4173/' });
let unhandled = null;
const onUnhandled = reason => { unhandled = reason; };
process.on('unhandledRejection', onUnhandled);
failedCdp.emit('Debugger.scriptParsed', { scriptId: 'failed-kit', url: 'http://127.0.0.1:4173/node_modules/.vite/deps/@kaminos_webgpu-inference-kit.js?v=abc', isModule: true, executionContextId: 4 });
await new Promise(resolve => setImmediate(resolve));
await assert.rejects(() => failedCapture.snapshot(), /source retrieval denied/);
await new Promise(resolve => setImmediate(resolve));
process.off('unhandledRejection', onUnhandled);
assert.equal(unhandled, null, 'early CDP source retrieval rejection must be captured for the parent phase report');
failedCapture.dispose();
console.log('browser-kit identity binds Chrome-parsed module source and rejects dependency-only capture');
