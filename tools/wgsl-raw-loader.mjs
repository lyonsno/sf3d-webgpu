/**
 * Minimal Node ESM loader hook so contract tests can import modules that use
 * Vite's `import x from './foo.wgsl?raw'` syntax. Shader text is irrelevant to
 * the dispatch-sequence contract (the tests stub the _dispatch* helpers), so we
 * resolve any .wgsl (with or without ?raw) to an empty-string default export.
 */
export async function resolve(specifier, context, next) {
  if (specifier.includes('.wgsl')) {
    const clean = specifier.replace(/\?.*$/, '');
    const resolved = await next(clean, context);
    return { ...resolved, url: `${resolved.url}?wgsl-raw`, shortCircuit: true };
  }
  return next(specifier, context);
}

export async function load(url, context, next) {
  if (url.endsWith('?wgsl-raw')) {
    return {
      format: 'module',
      source: 'export default "";',
      shortCircuit: true,
    };
  }
  return next(url, context);
}
