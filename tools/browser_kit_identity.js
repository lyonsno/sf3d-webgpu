import * as kit from '@kaminos/webgpu-inference-kit';

const bytesToHex = bytes => [...bytes]
  .map(value => value.toString(16).padStart(2, '0'))
  .join('');

async function sha256(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return bytesToHex(new Uint8Array(digest));
}

export async function assembleBrowserKitIdentity({ exportedVersion, exportNames, servedModules, witnessModuleUrl }) {
  const modules = [...servedModules]
    .map(module => ({ url: module.url, bytes: Number(module.bytes), sha256: module.sha256 }))
    .sort((a, b) => a.url.localeCompare(b.url));
  const canonicalModules = JSON.stringify(modules);
  const servedModuleSetSha256 = await sha256(new TextEncoder().encode(canonicalModules));
  const canonicalApi = JSON.stringify({
    packageName: '@kaminos/webgpu-inference-kit', exportedVersion, exportNames,
  });
  const exportFingerprint = await sha256(new TextEncoder().encode(canonicalApi));
  return Object.freeze({
    packageName: '@kaminos/webgpu-inference-kit',
    exportedVersion,
    exportNames,
    exportFingerprint,
    servedModuleSetSha256,
    servedModuleCount: modules.length,
    servedModules: Object.freeze(modules.map(module => Object.freeze(module))),
    witnessModuleUrl,
  });
}

export async function readBrowserKitIdentity() {
  const exportNames = Object.keys(kit).sort();
  const exportedVersion = kit.WEBGPU_INFERENCE_KIT_VERSION;
  const entries = performance.getEntriesByType('resource');
  const isOptimizerModule = (entry) => {
    const url = new URL(entry.name, location.href);
    return /\.m?js$/i.test(url.pathname)
      && (url.pathname.includes('/node_modules/.vite/deps/')
        || url.pathname.includes('/node_modules/@kaminos/webgpu-inference-kit/'));
  };
  const kitModuleEntries = entries.filter(entry => {
    const url = new URL(entry.name, location.href);
    return /\.m?js$/i.test(url.pathname)
      && (url.pathname.includes('@kaminos_webgpu-inference-kit')
        || url.pathname.includes('/node_modules/@kaminos/webgpu-inference-kit/'));
  });
  if (!kitModuleEntries.length) throw new Error('browser did not expose a loaded @kaminos/webgpu-inference-kit module response');
  const moduleUrls = [...new Set(entries.filter(isOptimizerModule).map(entry => entry.name))].sort();
  const servedModules = [];
  for (const name of moduleUrls) {
    const url = new URL(name, location.href);
    const response = await fetch(url.href, { cache: 'no-store' });
    if (!response.ok) throw new Error(`could not read browser-served module ${url.pathname}: HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (!bytes.byteLength) throw new Error(`browser-served module is empty: ${url.pathname}`);
    servedModules.push({ url: `${url.pathname}${url.search}`, bytes: bytes.byteLength, sha256: await sha256(bytes) });
  }
  const identity = await assembleBrowserKitIdentity({
    exportedVersion, exportNames, servedModules, witnessModuleUrl: import.meta.url,
  });
  return identity;
}
