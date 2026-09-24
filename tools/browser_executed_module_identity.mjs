import { createHash } from 'node:crypto';

const isScriptPath = pathname => /\.(?:m?js)$/i.test(pathname);
const isViteModulePath = pathname => pathname.includes('/node_modules/.vite/deps/')
  || pathname.includes('/node_modules/@kaminos/webgpu-inference-kit/');
const isKitModulePath = pathname => pathname.includes('@kaminos_webgpu-inference-kit')
  || pathname.includes('/node_modules/@kaminos/webgpu-inference-kit/');

export function isRelevantExecutedModuleUrl(value, baseUrl) {
  const url = new URL(value, baseUrl);
  return isScriptPath(url.pathname) && isViteModulePath(url.pathname);
}

export function createBrowserExecutedKitModuleCapture(cdp, { baseUrl }) {
  const scriptSources = new Map();
  const onScriptParsed = event => {
    if (event.type !== 'module' || !event.url || !isRelevantExecutedModuleUrl(event.url, baseUrl)) return;
    const scriptId = String(event.scriptId);
    if (scriptSources.has(scriptId)) return;
    const source = cdp.send('Debugger.getScriptSource', { scriptId }).then(result => {
      if (typeof result.scriptSource !== 'string' || result.scriptSource.length === 0) {
        throw new Error(`Chrome returned no executed source for ${event.url}`);
      }
      const sourceBytes = Buffer.from(result.scriptSource, 'utf8');
      return Object.freeze({
        scriptId,
        url: new URL(event.url, baseUrl).pathname + new URL(event.url, baseUrl).search,
        bytes: sourceBytes.byteLength,
        sha256: createHash('sha256').update(sourceBytes).digest('hex'),
      });
    });
    scriptSources.set(scriptId, source);
  };
  cdp.on('Debugger.scriptParsed', onScriptParsed);

  return Object.freeze({
    async snapshot() {
      const modules = (await Promise.all(scriptSources.values()))
        .sort((a, b) => a.url.localeCompare(b.url) || a.scriptId.localeCompare(b.scriptId));
      if (!modules.some(module => isKitModulePath(module.url))) {
        throw new Error('Chrome did not parse an @kaminos/webgpu-inference-kit module');
      }
      return Object.freeze(modules);
    },
    dispose() { cdp.off('Debugger.scriptParsed', onScriptParsed); },
  });
}
