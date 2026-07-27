import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  createValidatedClipPipeline,
  loadClipShaderSources,
  resolveClipShaderUrls,
} from '../src/lib/clip_estimator.js';

const clipSource = readFileSync(new URL('../src/lib/clip_estimator.js', import.meta.url), 'utf8');
assert.doesNotMatch(
  clipSource,
  /fetch\(\s*['"]\/src\/shaders\//,
  'CLIP shaders must not resolve from the embedding document origin',
);

const moduleUrl = 'https://source.example/sf3d/src/lib/clip_estimator.js';
assert.deepEqual(resolveClipShaderUrls(moduleUrl), {
  linear: 'https://source.example/sf3d/src/shaders/linear.wgsl',
  layernorm: 'https://source.example/sf3d/src/shaders/layernorm_vit.wgsl',
});

function response(body, {
  ok = true,
  status = 200,
  statusText = 'OK',
  url = 'fixture://shader',
  contentType = 'text/wgsl',
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
    async text() {
      return body;
    },
  };
}

const requested = [];
const validWgsl = '@compute @workgroup_size(1) fn main() {}';
const sources = await loadClipShaderSources({
  moduleUrl,
  fetchImpl: async url => {
    requested.push(String(url));
    return response(validWgsl, { url: String(url) });
  },
});
assert.deepEqual(requested, [
  'https://source.example/sf3d/src/shaders/linear.wgsl',
  'https://source.example/sf3d/src/shaders/layernorm_vit.wgsl',
]);
assert.deepEqual(sources, {
  linear: validWgsl,
  layernorm: validWgsl,
});

await assert.rejects(
  () => loadClipShaderSources({
    moduleUrl,
    fetchImpl: async url => response('<!DOCTYPE HTML><title>missing</title>', {
      ok: false,
      status: 404,
      statusText: 'Not Found',
      url: String(url),
      contentType: 'text/html',
    }),
  }),
  error => /404/.test(error.message) && /linear\.wgsl/.test(error.message),
  'HTTP shader failures must identify the source-owned URL before compilation',
);

await assert.rejects(
  () => loadClipShaderSources({
    moduleUrl,
    fetchImpl: async url => response('<!DOCTYPE HTML><title>fallback</title>', {
      url: String(url),
      contentType: 'text/html; charset=utf-8',
    }),
  }),
  error => /text\/html/.test(error.message) && /linear\.wgsl/.test(error.message),
  'HTML fallback bodies must not masquerade as shader source',
);

await assert.rejects(
  () => loadClipShaderSources({
    moduleUrl,
    fetchImpl: async url => response('not actually wgsl', { url: String(url) }),
  }),
  error => /WGSL/.test(error.message) && /linear\.wgsl/.test(error.message),
  'non-WGSL source must fail before pipeline creation',
);

let errorScopeDepth = 0;
await assert.rejects(
  () => createValidatedClipPipeline({
    pushErrorScope(kind) {
      assert.equal(kind, 'validation');
      errorScopeDepth++;
    },
    async popErrorScope() {
      errorScopeDepth--;
      return null;
    },
    createShaderModule() {
      return {
        async getCompilationInfo() {
          return {
            messages: [{
              type: 'error',
              lineNum: 7,
              linePos: 3,
              message: 'fixture compilation failure',
            }],
          };
        },
      };
    },
    createComputePipeline() {
      throw new Error('pipeline creation must not run after compilation failure');
    },
  }, validWgsl, 'linear.wgsl'),
  error => /fixture compilation failure/.test(error.message) && /linear\.wgsl/.test(error.message),
  'WGSL compilation errors must fail loud with shader identity',
);
assert.equal(errorScopeDepth, 0, 'failed shader compilation must balance the validation scope');

const effectiveShaderUrl = 'https://source.example/sf3d/src/shaders/linear.wgsl';
let synchronousFailurePopCount = 0;
await assert.rejects(
  () => createValidatedClipPipeline({
    pushErrorScope(kind) {
      assert.equal(kind, 'validation');
    },
    async popErrorScope() {
      synchronousFailurePopCount++;
      return null;
    },
    createShaderModule() {
      return {
        async getCompilationInfo() {
          return { messages: [] };
        },
      };
    },
    createComputePipeline() {
      throw new Error('fixture synchronous pipeline failure');
    },
  }, validWgsl, effectiveShaderUrl),
  error => (
    /fixture synchronous pipeline failure/.test(error.message)
    && error.message.includes(effectiveShaderUrl)
  ),
  'synchronous pipeline failures must retain the effective shader URL',
);
assert.equal(synchronousFailurePopCount, 1, 'synchronous pipeline failure must pop its scope once');

let rejectedPopCount = 0;
await assert.rejects(
  () => createValidatedClipPipeline({
    pushErrorScope(kind) {
      assert.equal(kind, 'validation');
    },
    async popErrorScope() {
      rejectedPopCount++;
      throw new Error('fixture validation scope rejection');
    },
    createShaderModule() {
      return {
        async getCompilationInfo() {
          return { messages: [] };
        },
      };
    },
    createComputePipeline() {
      return { fixture: 'untrusted-pipeline' };
    },
  }, validWgsl, effectiveShaderUrl),
  error => (
    /fixture validation scope rejection/.test(error.message)
    && error.message.includes(effectiveShaderUrl)
  ),
  'scope-pop rejection must retain the effective shader URL',
);
assert.equal(rejectedPopCount, 1, 'scope-pop rejection must attempt to pop its scope once');

const expectedPipeline = { fixture: 'pipeline' };
const pipeline = await createValidatedClipPipeline({
  pushErrorScope(kind) {
    assert.equal(kind, 'validation');
    errorScopeDepth++;
  },
  async popErrorScope() {
    errorScopeDepth--;
    return null;
  },
  createShaderModule({ code, label }) {
    assert.equal(code, validWgsl);
    assert.match(label, /linear\.wgsl/);
    return {
      async getCompilationInfo() {
        return { messages: [] };
      },
    };
  },
  createComputePipeline({ layout, compute }) {
    assert.equal(layout, 'auto');
    assert.equal(compute.entryPoint, 'main');
    return expectedPipeline;
  },
}, validWgsl, 'linear.wgsl');
assert.equal(pipeline, expectedPipeline);
assert.equal(errorScopeDepth, 0, 'successful pipeline creation must balance the validation scope');
