import * as kit from '@kaminos/webgpu-inference-kit';

const bytesToHex = bytes => [...bytes]
  .map(value => value.toString(16).padStart(2, '0'))
  .join('');

export async function readBrowserKitIdentity() {
  const exportNames = Object.keys(kit).sort();
  const exportedVersion = kit.WEBGPU_INFERENCE_KIT_VERSION;
  const canonical = JSON.stringify({
    packageName: '@kaminos/webgpu-inference-kit',
    exportedVersion,
    exportNames,
  });
  const digest = await crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(canonical),
  );
  return Object.freeze({
    packageName: '@kaminos/webgpu-inference-kit',
    exportedVersion,
    exportNames,
    exportFingerprint: bytesToHex(new Uint8Array(digest)),
    witnessModuleUrl: import.meta.url,
  });
}
