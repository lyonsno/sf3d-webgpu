import * as kit from '@kaminos/webgpu-inference-kit';

const bytesToHex = bytes => [...bytes]
  .map(value => value.toString(16).padStart(2, '0'))
  .join('');

async function sha256(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return bytesToHex(new Uint8Array(digest));
}

export async function assembleBrowserKitIdentity({ exportedVersion, exportNames, executedModuleCapture, witnessModuleUrl }) {
  const executedModules = executedModuleCapture?.modules ?? [];
  const modules = [...executedModules]
    .map(module => ({ scriptId: String(module.scriptId), executionContextId: Number(module.executionContextId), url: module.url, bytes: Number(module.bytes), sha256: module.sha256 }))
    .sort((a, b) => a.url.localeCompare(b.url) || a.scriptId.localeCompare(b.scriptId));
  const contentRows = [...new Map(modules.map(({ url, bytes, sha256 }) => [`${url}\0${sha256}`, { url, bytes, sha256 }])).values()]
    .sort((a, b) => a.url.localeCompare(b.url) || a.sha256.localeCompare(b.sha256));
  const canonicalModules = JSON.stringify(contentRows);
  const executedModuleSetSha256 = await sha256(new TextEncoder().encode(canonicalModules));
    const canonicalApi = JSON.stringify({
    packageName: '@kaminos/webgpu-inference-kit', exportedVersion, exportNames,
  });
  const exportFingerprint = await sha256(new TextEncoder().encode(canonicalApi));
  return Object.freeze({
    packageName: '@kaminos/webgpu-inference-kit',
    exportedVersion,
    exportNames,
    exportFingerprint,
    identityBasis: 'chrome-debugger-executed-module-source',
    executedModuleSetSha256,
    executedModuleCount: modules.length,
    kitModuleCount: modules.filter(module => /@kaminos_webgpu-inference-kit|\/node_modules\/@kaminos\/webgpu-inference-kit\//.test(module.url)).length,
    executedModules: Object.freeze(modules.map(module => Object.freeze(module))),
    producerBinding: executedModuleCapture?.producerBinding ? Object.freeze({ ...executedModuleCapture.producerBinding }) : null,
    witnessModuleUrl,
  });
}

export async function readBrowserKitIdentity({ executedModuleCapture }) {
  const exportNames = Object.keys(kit).sort();
  const exportedVersion = kit.WEBGPU_INFERENCE_KIT_VERSION;
  if (!Array.isArray(executedModuleCapture?.modules) || executedModuleCapture.modules.length === 0) throw new Error('Chrome did not provide executed kit module sources');
  const identity = await assembleBrowserKitIdentity({
    exportedVersion, exportNames, executedModuleCapture, witnessModuleUrl: import.meta.url,
  });
  return identity;
}
