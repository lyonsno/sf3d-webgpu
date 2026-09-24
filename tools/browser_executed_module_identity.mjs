import { createHash } from 'node:crypto';

const isScriptPath = pathname => /\.(?:m?js)$/i.test(pathname);
const isKitDependencyPath = pathname => pathname.includes('/node_modules/.vite/deps/')
  || pathname.includes('/node_modules/@kaminos/webgpu-inference-kit/');
const isKitModulePath = pathname => pathname.includes('@kaminos_webgpu-inference-kit')
  || pathname.includes('/node_modules/@kaminos/webgpu-inference-kit/');
const isImporterPath = pathname => pathname.endsWith('/src/lib/sf3d_producer.js')
  || pathname.endsWith('/tools/browser_kit_identity.js');
const escapeRegex = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const importsExactUrl = (source, url) => new RegExp(`(?:\\bfrom\\s*|\\bimport\\s*)["']${escapeRegex(url)}["']`).test(source);

export function isRelevantExecutedModuleUrl(value, baseUrl) {
  const url = new URL(value, baseUrl);
  return isScriptPath(url.pathname) && (isKitDependencyPath(url.pathname) || isImporterPath(url.pathname));
}

export function createBrowserExecutedKitModuleCapture(cdp, { baseUrl }) {
  const scriptSources = new Map();
  const captureErrors = new Map();
  const onScriptParsed = event => {
    if (event.isModule !== true || !event.url || !isRelevantExecutedModuleUrl(event.url, baseUrl)) return;
    const scriptId = String(event.scriptId);
    if (scriptSources.has(scriptId)) return;
    const source = Promise.resolve().then(() => cdp.send('Debugger.getScriptSource', { scriptId })).then(result => {
      if (typeof result.scriptSource !== 'string' || result.scriptSource.length === 0) {
        throw new Error(`Chrome returned no executed source for ${event.url}`);
      }
      const sourceBytes = Buffer.from(result.scriptSource, 'utf8');
      return Object.freeze({
        scriptId,
        executionContextId: event.executionContextId,
        url: new URL(event.url, baseUrl).pathname + new URL(event.url, baseUrl).search,
        bytes: sourceBytes.byteLength,
        sha256: createHash('sha256').update(sourceBytes).digest('hex'),
        source: result.scriptSource,
      });
    }).catch(error => {
      captureErrors.set(scriptId, error);
      return null;
    });
    scriptSources.set(scriptId, source);
  };
  cdp.on('Debugger.scriptParsed', onScriptParsed);

  return Object.freeze({
    async snapshot() {
      const captured = (await Promise.all(scriptSources.values())).filter(Boolean);
      if (captureErrors.size) {
        const [scriptId, error] = captureErrors.entries().next().value;
        throw new Error(`Chrome executed-module source retrieval failed for script ${scriptId}: ${error.message}`);
      }
      const kitModules = captured.filter(module => isKitModulePath(module.url));
      if (kitModules.length === 0) {
        throw new Error('Chrome did not parse an @kaminos/webgpu-inference-kit module');
      }
      if (kitModules.length !== 1) throw new Error(`Chrome parsed ${kitModules.length} kit module instances; producer binding is ambiguous`);
      const kitModule = kitModules[0];
      const producerModules = captured.filter(module => new URL(module.url, baseUrl).pathname.endsWith('/src/lib/sf3d_producer.js'));
      const helperModules = captured.filter(module => new URL(module.url, baseUrl).pathname.endsWith('/tools/browser_kit_identity.js'));
      if (producerModules.length !== 1 || helperModules.length !== 1) {
        throw new Error(`Chrome parsed ${producerModules.length} producer modules and ${helperModules.length} identity-helper modules; producer binding is ambiguous`);
      }
      const producerModule = producerModules[0], helperModule = helperModules[0];
      if (!Number.isInteger(kitModule.executionContextId)
        || producerModule.executionContextId !== kitModule.executionContextId
        || helperModule.executionContextId !== kitModule.executionContextId) {
        throw new Error('Chrome kit, producer, and identity-helper modules are not in one execution context');
      }
      if (!importsExactUrl(producerModule.source, kitModule.url) || !importsExactUrl(helperModule.source, kitModule.url)) {
        throw new Error('Chrome producer and identity helper do not both import the unique parsed kit module URL');
      }
      const modules = captured.filter(module => isKitDependencyPath(new URL(module.url, baseUrl).pathname))
        .sort((a, b) => a.url.localeCompare(b.url) || a.scriptId.localeCompare(b.scriptId))
        .map(({ source, ...module }) => Object.freeze(module));
      return Object.freeze({
        modules: Object.freeze(modules),
        producerBinding: Object.freeze({
          executionContextId: kitModule.executionContextId,
          kitModuleScriptId: kitModule.scriptId,
          kitModuleUrl: kitModule.url,
          producerModuleScriptId: producerModule.scriptId,
          producerModuleUrl: producerModule.url,
          producerModuleExecutionContextId: producerModule.executionContextId,
          producerImportsKitModuleUrl: kitModule.url,
          identityHelperScriptId: helperModule.scriptId,
          identityHelperModuleUrl: helperModule.url,
          identityHelperExecutionContextId: helperModule.executionContextId,
          identityHelperImportsKitModuleUrl: kitModule.url,
          linkBasis: 'same-context-static-import-url',
        }),
      });
    },
    dispose() { cdp.off('Debugger.scriptParsed', onScriptParsed); },
  });
}
