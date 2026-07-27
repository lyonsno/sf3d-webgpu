import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  loadTetData,
  resolveSourcePublicBaseUrl,
} from '../src/lib/marching_tet.js';

const moduleUrl = new URL('../src/lib/marching_tet.js', import.meta.url);
const publicBaseUrl = new URL('/', moduleUrl);
const expectedGridUrl = new URL('tets/_grid_vertices.bin', publicBaseUrl).href;
const expectedIndicesUrl = new URL('tets/indices.bin', publicBaseUrl).href;
const originalFetch = globalThis.fetch;
const inferenceSource = readFileSync(new URL('../src/lib/inference.js', import.meta.url), 'utf8');

assert.match(inferenceSource, /loadTetData\(\)/, 'the production inference route must use the module-owned tet asset origin');
assert.doesNotMatch(inferenceSource, /loadTetData\(['"]tets\/['"]\)/, 'the production route must not reintroduce document-relative tet assets');
assert.equal(
  resolveSourcePublicBaseUrl('./', 'https://source.example/sf3d/assets/index.js', false).href,
  'https://source.example/sf3d/',
  'relative production bases must resolve from the built bundle directory to the source public root',
);
assert.equal(
  resolveSourcePublicBaseUrl('', 'https://source.example/sf3d/assets/index.js', false).href,
  'https://source.example/sf3d/',
  'empty production bases must preserve relative-build public-root semantics',
);
assert.equal(
  resolveSourcePublicBaseUrl('./', 'https://source.example/src/lib/marching_tet.js', true).href,
  'https://source.example/',
  'development modules must resolve public assets from the source origin',
);

function response(bytes, {
  ok = true,
  status = 200,
  statusText = 'OK',
  url = 'fixture://tet',
  contentType = 'application/octet-stream',
} = {}) {
  return {
    ok,
    status,
    statusText,
    url,
    headers: {
      get(name) {
        return name.toLowerCase() === 'content-type' ? contentType : null;
      },
    },
    async arrayBuffer() {
      return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    },
  };
}

try {
  const requested = [];
  globalThis.fetch = async url => {
    requested.push(String(url));
    if (String(url).endsWith('_grid_vertices.bin')) {
      return response(new Uint8Array(new Float32Array([0, 0.5, 1]).buffer), { url: String(url) });
    }
    return response(new Uint8Array(new Int32Array([0, 1, 2, 3]).buffer), { url: String(url) });
  };
  const loaded = await loadTetData();
  assert.deepEqual(requested, [expectedGridUrl, expectedIndicesUrl]);
  assert.deepEqual([...loaded.gridVertices], [0, 0.5, 1]);
  assert.deepEqual([...loaded.indices], [0, 1, 2, 3]);

  requested.length = 0;
  await loadTetData('custom-tets/');
  assert.deepEqual(requested, [
    new URL('custom-tets/_grid_vertices.bin', publicBaseUrl).href,
    new URL('custom-tets/indices.bin', publicBaseUrl).href,
  ]);

  requested.length = 0;
  await loadTetData('tets/');
  assert.deepEqual(requested, [
    expectedGridUrl,
    expectedIndicesUrl,
  ]);

  globalThis.fetch = async url => response(new Uint8Array(335), {
    ok: false,
    status: 404,
    statusText: 'Not Found',
    url: String(url),
  });
  await assert.rejects(
    () => loadTetData(),
    error => /404/.test(error.message) && /_grid_vertices\.bin/.test(error.message),
    'HTML 404 bodies must fail as fetch errors before typed-array parsing',
  );

  globalThis.fetch = async url => response(
    String(url).endsWith('_grid_vertices.bin') ? new Uint8Array(335) : new Uint8Array(16),
    { url: String(url) },
  );
  await assert.rejects(
    () => loadTetData(),
    error => /335/.test(error.message) && /multiple of 4/.test(error.message) && /_grid_vertices\.bin/.test(error.message),
    'misaligned tet assets must identify the malformed binary rather than throw an anonymous TypedArray error',
  );

  globalThis.fetch = async url => response(new Uint8Array(336), {
    url: String(url),
    contentType: 'text/html; charset=utf-8',
  });
  await assert.rejects(
    () => loadTetData(),
    error => /text\/html/.test(error.message) && /_grid_vertices\.bin/.test(error.message),
    'aligned HTML fallback bodies must not masquerade as tet binaries',
  );

  globalThis.fetch = async url => response(
    String(url).endsWith('_grid_vertices.bin')
      ? new Uint8Array(new Float32Array([0, 0.5, 1, 1]).buffer)
      : new Uint8Array(new Int32Array([0, 1, 2, 3]).buffer),
    { url: String(url) },
  );
  await assert.rejects(
    () => loadTetData(),
    /expected xyz triples/,
    'malformed vertex record shape must fail',
  );

  globalThis.fetch = async url => response(
    String(url).endsWith('_grid_vertices.bin')
      ? new Uint8Array(new Float32Array([0, 0.5, 1]).buffer)
      : new Uint8Array(new Int32Array([0, 1, 2, 3, 4]).buffer),
    { url: String(url) },
  );
  await assert.rejects(
    () => loadTetData(),
    /expected tetrahedra quads/,
    'malformed index record shape must fail',
  );
} finally {
  globalThis.fetch = originalFetch;
}
