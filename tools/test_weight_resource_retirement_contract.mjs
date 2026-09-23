/** Real loader/CLIP initialization/producer lifecycle; fake GPU, no numerical or memory-fit claim. */
import assert from 'node:assert/strict';
import test from 'node:test';
import { loadWeights } from '../src/lib/weights.js';
import { createSf3dProducer, finishProducerRunWithEvidence, releaseLoadedWeights, releaseProducerResources } from '../src/lib/sf3d_producer.js';
import { createProductRouteWorkers } from '../src/lib/product_route.js';
import { runClipPrep } from '../src/lib/clip_estimator.js';
import { weightFixture, fakeWeightDevice, installWeightFetch } from './fixtures/weight_resource_fixture.mjs';

globalThis.GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, MAP_READ: 1 };
globalThis.GPUShaderStage = { COMPUTE: 4 };
let moduleId = 0;
const freshClip = () => import(`../src/lib/clip_estimator.js?retirement-test=${++moduleId}`);
const prepName = 'image_estimator.model.visual.conv1.weight';
const warmClip = (clip, device, weights) => assert.rejects(
  clip.estimateMaterials(device, new Uint8Array(4), 1, 1, weights),
  /clip conv1 weight must be Float32Array\[2359296\], got 1/,
  'the tiny fixture intentionally stops at prep, after real CLIP resource initialization',
);

for (const normX of [true, false]) test(`optional normalization present=${normX}: allocate once and retire`, async () => {
  const fixture = weightFixture({ normX });
  const restore = installWeightFetch(fixture);
  try {
    const device = fakeWeightDevice();
    const weights = await loadWeights(device, 'fixture://weights');
    for (let b = 0; b < 4; b++) for (const fuse of ['fuse_block_in', 'fuse_block_out']) {
      const norm = weights.backbone.mainBlocks[b][fuse === 'fuse_block_in' ? 'fuseBlockIn' : 'fuseBlockOut'].normX;
      if (!normX) { assert.equal(norm, null); continue; }
      const value = fixture.values.get(`backbone.main_blocks.${b}.${fuse}.norm_x.weight`);
      assert.equal(device.buffers.filter(buffer => new Float32Array(buffer.data)[0] === value).length, 1,
        'a presence probe must not allocate a discarded normalization buffer');
    }
    releaseLoadedWeights(weights);
    assert.ok(device.buffers.every(buffer => buffer.destroyed === 1), 'every uploaded tensor retires');
  } finally { restore(); }
});

test('producer disposal drains foreground, retires all 148 lazy buffers once, and keeps injected device', async () => {
  const restore = installWeightFetch(weightFixture({ normX: false }));
  try {
    const clip = await freshClip();
    const device = fakeWeightDevice();
    device.destroy = () => assert.fail('host device is borrowed');
    const producer = await createSf3dProducer({ device, weightsUrl: 'fixture://weights', workers: {} });
    const eagerCount = device.buffers.length;
    await warmClip(clip, device, producer.weights);
    const lazy = device.buffers.slice(eagerCount);
    assert.equal(lazy.length, 148, 'real CLIP initializer uploaded all transformer weights');
    let unblock;
    let started;
    const gate = new Promise(resolve => { unblock = resolve; });
    const began = new Promise(resolve => { started = resolve; });
    const frame = producer.requestForegroundOpportunity({ requestId: 'retirement-frame', async run() { started(); await gate; } });
    await began;
    const disposal = producer.dispose();
    assert.equal(producer.dispose().completion, disposal.completion);
    await Promise.resolve();
    assert.ok(device.buffers.every(buffer => buffer.destroyed === 0), 'admitted foreground holds retirement');
    unblock();
    await frame.completion;
    assert.deepEqual(await disposal.completion, { status: 'released' });
    assert.ok(lazy.every(buffer => buffer.destroyed === 1), 'all lazy CLIP weights retired before completion');
    assert.ok(device.buffers.every(buffer => buffer.destroyed === 1), 'eager and lazy resources retired exactly once');
    await producer.dispose().completion;
    assert.ok(device.buffers.every(buffer => buffer.destroyed === 1));
  } finally { restore(); }
});

test('retained disposed producer cannot access CPU payload or allocate any lazy weight', async () => {
  const restore = installWeightFetch(weightFixture({ normX: false }));
  try {
    const device = fakeWeightDevice();
    const producer = await createSf3dProducer({ device, weightsUrl: 'fixture://weights', workers: {} });
    assert.equal(producer.weights._rawGetCPU(prepName).length, 1);
    await producer.dispose().completion;
    const count = device.buffers.length;
    for (const method of ['_rawGet', '_rawGetCPU', '_rawTryGet', '_rawHas']) {
      for (const name of [prepName, 'missing']) assert.throws(() => producer.weights[method](name), /SF3D weights are disposed/);
    }
    assert.equal(device.buffers.length, count, 'post-disposal access cannot upload');
  } finally { restore(); }
});

test('CLIP pipelines are device-scoped and lazy weights are weight-set-scoped', async () => {
  const restore = installWeightFetch(weightFixture({ normX: false }));
  try {
    const clip = await freshClip();
    const a = fakeWeightDevice();
    const b = fakeWeightDevice();
    const first = await loadWeights(a, 'fixture://first');
    await warmClip(clip, a, first);
    const count = a.buffers.length;
    await warmClip(clip, a, first);
    assert.equal(a.buffers.length, count, 'one weight identity reuses its lazy set');
    assert.equal(a.clipPipelines.length, 2);
    releaseLoadedWeights(first);
    const second = await loadWeights(a, 'fixture://second');
    const secondStart = a.buffers.length;
    await warmClip(clip, a, second);
    assert.equal(a.buffers.length - secondStart, 148, 'replacement weights do not borrow the disposed set');
    assert.equal(a.clipPipelines.length, 2, 'same device may reuse immutable pipelines');
    const other = await loadWeights(b, 'fixture://other');
    const otherStart = b.buffers.length;
    await warmClip(clip, b, other);
    assert.equal(b.buffers.length - otherStart, 148);
    assert.equal(b.clipPipelines.length, 2, 'new device compiles its own CLIP pipelines');
    releaseLoadedWeights(second);
    releaseLoadedWeights(other);
    assert.ok([...a.buffers, ...b.buffers].every(buffer => buffer.destroyed === 1));
  } finally { restore(); }
});

test('partial lazy initialization can retry without a partial cache or duplicate uploads', async () => {
  const restore = installWeightFetch(weightFixture({ normX: false }));
  try {
    const clip = await freshClip();
    const device = fakeWeightDevice();
    const weights = await loadWeights(device, 'fixture://weights');
    const start = device.buffers.length;
    device.failBufferAt = start + 5;
    await assert.rejects(clip.estimateMaterials(device, new Uint8Array(4), 1, 1, weights), /injected allocation failure/);
    device.failBufferAt = null;
    await warmClip(clip, device, weights);
    assert.equal(device.buffers.length - start, 148, 'retry completes the set without leaking duplicate partial uploads');
    releaseLoadedWeights(weights);
    assert.ok(device.buffers.every(buffer => buffer.destroyed === 1));
  } finally { restore(); }
});

test('injected weight sets remain host-owned, including their lazy buffers and raw payload', async () => {
  const restore = installWeightFetch(weightFixture({ normX: false }));
  try {
    const clip = await freshClip();
    const device = fakeWeightDevice();
    const weights = await loadWeights(device, 'fixture://host-weights');
    const producer = await createSf3dProducer({ device, weights, workers: {} });
    await warmClip(clip, device, weights);
    await producer.dispose().completion;
    assert.ok(device.buffers.every(buffer => buffer.destroyed === 0), 'borrower must not retire host weights');
    assert.equal(weights._rawGetCPU(prepName).length, 1);
    releaseLoadedWeights(weights); // the host, not the disposed borrower, releases its set
    assert.ok(device.buffers.every(buffer => buffer.destroyed === 1));
  } finally { restore(); }
});

test('producer disposal resets copied preparation weights in an injected worker after foreground drain', async () => {
  const device = fakeWeightDevice();
  const listeners = new Set();
  let resetCount = 0;
  const worker = {
    addEventListener(type, listener) { if (type === 'message') listeners.add(listener); },
    removeEventListener(type, listener) { if (type === 'message') listeners.delete(listener); },
    postMessage(message) {
      const reply = data => { for (const listener of [...listeners]) listener({ data: { id: message.id, ok: true, ...data } }); };
      if (message.type === 'init') { queueMicrotask(() => reply({ initialized: true })); return; }
      if (message.type === 'reset') { resetCount += 1; queueMicrotask(() => reply({ reset: true })); return; }
      queueMicrotask(() => reply({ embeddings: new Float32Array(50 * 768).buffer }));
    },
  };
  const prep = {
    conv1W: new Float32Array(768 * 3072),
    classEmb: new Float32Array(768),
    posEmb: new Float32Array(50 * 768),
  };
  const weights = { _rawGetCPU(name) {
    if (name.endsWith('conv1.weight')) return prep.conv1W;
    if (name.endsWith('class_embedding')) return prep.classEmb;
    if (name.endsWith('positional_embedding')) return prep.posEmb;
    throw new Error(`unexpected prep tensor ${name}`);
  } };
  const producer = await createSf3dProducer({ device, weights, workers: { clipPrepWorker: worker } });
  await runClipPrep(worker, new Uint8ClampedArray(4), 1, 1, weights);
  await producer.dispose().completion;
  assert.equal(resetCount, 1,
    'producer disposal acknowledges release of the copied tensors without terminating the injected worker');
});

test('negative CLIP reset acknowledgment rejects disposal after owned workers and weights are retired', async () => {
  const restoreFetch = installWeightFetch(weightFixture({ normX: false, fullClipPrep: true }));
  const OriginalWorker = globalThis.Worker;
  const createdWorkers = [];
  globalThis.Worker = class {
    constructor(url, options) {
      this.url = String(url);
      this.name = options?.name || '';
      this.listeners = { message: new Set(), error: new Set(), messageerror: new Set() };
      this.terminated = 0;
      this.installed = null;
      createdWorkers.push(this);
    }
    addEventListener(type, listener) { this.listeners[type]?.add(listener); }
    removeEventListener(type, listener) { this.listeners[type]?.delete(listener); }
    emit(type, data) { for (const listener of [...this.listeners[type]]) listener(data); }
    postMessage(message) {
      const reply = payload => queueMicrotask(() => this.emit('message', { data: { id: message.id, ok: true, ...payload } }));
      if (message.type === 'init') {
        this.installed = { conv1W: message.conv1W, classEmb: message.classEmb, posEmb: message.posEmb };
        reply({ initialized: true });
      } else if (message.type === 'reset') {
        this.installed = null;
        reply({ reset: false });
      } else {
        reply({ embeddings: new Float32Array(50 * 768).buffer });
      }
    }
    terminate() { this.terminated += 1; }
  };

  try {
    const device = fakeWeightDevice();
    const producer = await createSf3dProducer({ device, weightsUrl: 'fixture://weights' });
    const clipWorker = producer.workers.clipPrepWorker;
    assert.equal(createdWorkers.length, 5, 'producer owns the five product-route workers');
    await runClipPrep(clipWorker, new Uint8ClampedArray(4), 1, 1, producer.weights, { timeoutMs: 5000 });
    assert.ok(clipWorker.installed, 'CLIP worker holds producer-owned preparation tensors before release');

    await assert.rejects(
      () => producer.dispose().completion,
      /clip prep reset not acknowledged/,
      'the reset failure remains visible to the lifecycle caller');
    assert.deepEqual(createdWorkers.map(worker => worker.terminated), [1, 1, 1, 1, 1],
      'every producer-owned worker receives its termination attempt despite reset failure');
    assert.ok(device.buffers.every(buffer => buffer.destroyed === 1),
      'producer-owned eager GPU weight buffers retire despite reset failure');
    assert.throws(() => producer.weights._rawGetCPU(prepName), /SF3D weights are disposed/,
      'producer-owned retained preparation bytes are cleared despite reset failure');
  } finally {
    globalThis.Worker = OriginalWorker;
    restoreFetch();
  }
});

test('failed foreground drain retains owned workers and weights', async () => {
  const finishError = new Error('foreground drain did not settle');
  let clipWorkerReads = 0;
  let workerTerminations = 0;
  let weightDisposals = 0;
  const routeWorkers = { get clipPrepWorker() {
    clipWorkerReads += 1;
    return { terminate() { workerTerminations += 1; } };
  } };
  await assert.rejects(() => releaseProducerResources({
    foreground: { dispose() { throw finishError; } },
    retainedClipPrepWorker: true,
    routeWorkers,
    modelWeights: { dispose() { weightDisposals += 1; } },
    ownsWorkers: true,
    ownsWeights: true,
  }), error => error === finishError);
  assert.equal(clipWorkerReads, 0, 'failed drain cannot start CLIP reset or worker retirement');
  assert.equal(workerTerminations, 0);
  assert.equal(weightDisposals, 0);
});

test('later worker constructor failure terminates every worker created before the throw', () => {
  const OriginalWorker = globalThis.Worker;
  const created = [];
  globalThis.Worker = class {
    constructor() {
      if (created.length === 1) throw new Error('second worker failed');
      this.terminated = 0;
      created.push(this);
    }
    terminate() { this.terminated += 1; }
  };
  try {
    assert.throws(() => createProductRouteWorkers(), /second worker failed/);
    assert.equal(created.length, 1);
    assert.equal(created[0].terminated, 1);
  } finally { globalThis.Worker = OriginalWorker; }
});

test('inference and foreground-finish failures preserve both causes and last run evidence', async () => {
  const originalPerformance = globalThis.performance;
  const inferenceError = new Error('inference phase failed');
  const finishError = new Error('foreground finish clock failed');
  let failNextClock = false;
  globalThis.performance = {
    timeOrigin: originalPerformance.timeOrigin,
    now() {
      if (failNextClock) { failNextClock = false; throw finishError; }
      return originalPerformance.now();
    },
  };
  try {
    const device = fakeWeightDevice();
    const producer = await createSf3dProducer({ device, weights: {}, workers: {} });
    let thrown;
    try {
      await producer.run({ width: 1, height: 1 }, {
        runId: 'dual-failure',
        onProgress() {
          producer.requestForegroundOpportunity({ requestId: 'dual-failure-frame', run() { return null; } });
          failNextClock = true;
          throw inferenceError;
        },
      });
    } catch (error) { thrown = error; }
    assert.ok(thrown instanceof AggregateError, `neither failure may replace the other (got ${thrown?.name}: ${thrown?.message})`);
    assert.deepEqual(thrown.errors, [inferenceError, finishError]);
    assert.equal(thrown.sf3dRun?.runId, 'dual-failure');
    assert.equal(thrown.sf3dRun?.lastProgress, 'Preprocessing image...');
    assert.equal(thrown.sf3dRun?.foregroundOpportunityReport, null);
    assert.equal(producer.quarantined, true, 'failed finish quarantines model execution');
    assert.equal(producer.disposed, false, 'failed finish does not detach the foreground host');
    await assert.rejects(producer.dispose().completion, error => error === finishError);
  } finally { globalThis.performance = originalPerformance; }
});

test('falsy foreground-finish rejection remains a distinct dual-failure cause', async () => {
  const originalPerformance = globalThis.performance;
  const inferenceError = new Error('inference phase failed with null finish');
  let failNextClock = false;
  globalThis.performance = {
    timeOrigin: originalPerformance.timeOrigin,
    now() {
      if (failNextClock) { failNextClock = false; throw null; }
      return originalPerformance.now();
    },
  };
  try {
    const producer = await createSf3dProducer({ device: fakeWeightDevice(), weights: {}, workers: {} });
    let thrown;
    try {
      await producer.run({ width: 1, height: 1 }, {
        runId: 'dual-null-finish',
        onProgress() {
          producer.requestForegroundOpportunity({ requestId: 'dual-null-frame', run() { return null; } });
          failNextClock = true;
          throw inferenceError;
        },
      });
    } catch (error) { thrown = error; }
    assert.ok(thrown instanceof AggregateError);
    assert.deepEqual(thrown.errors, [inferenceError, null]);
    assert.equal(thrown.sf3dRun?.runId, 'dual-null-finish');
    assert.equal(producer.quarantined, true);
    const disposal = producer.dispose();
    assert.equal(disposal.status, 'quarantined');
    assert.deepEqual(await disposal.completion.then(
      () => ({ rejected: false }), reason => ({ rejected: true, reason })),
    { rejected: true, reason: null });
  } finally { globalThis.performance = originalPerformance; }
});

test('finish-only failure retains completed inference run evidence', async () => {
  const finishError = new Error('foreground finish rejected after inference');
  let releaseCalls = 0;
  await assert.rejects(() => finishProducerRunWithEvidence({
    prepared: { release() { releaseCalls += 1; throw finishError; } },
    pipelineFailed: false,
    runId: 'finish-only',
    lastProgress: 'GLB export complete',
    startedAtMs: 1,
    deviceInjected: true,
    commit: 'test-commit',
  }), error => {
    assert.equal(error, finishError);
    assert.equal(error.sf3dRun?.runId, 'finish-only');
    assert.equal(error.sf3dRun?.lastProgress, 'GLB export complete');
    assert.equal(error.sf3dRun?.foregroundOpportunityReport, null);
    assert.equal(error.sf3dRun?.inferenceCompleted, true);
    return true;
  });
  assert.equal(releaseCalls, 1);
});

test('inference-only failure keeps its original error and a settled foreground report', async () => {
  const inferenceError = new Error('inference callback failed');
  const producer = await createSf3dProducer({ device: fakeWeightDevice(), weights: {}, workers: {} });
  await assert.rejects(() => producer.run({ width: 1, height: 1 }, {
    runId: 'single-failure',
    onProgress() { throw inferenceError; },
  }), error => {
    assert.equal(error, inferenceError);
    assert.equal(error.sf3dRun?.runId, 'single-failure');
    assert.equal(error.sf3dRun?.foregroundOpportunityReport?.status, 'succeeded');
    return true;
  });
  assert.equal(producer.quarantined, false);
  await producer.dispose().completion;
});

test('known loader weights cannot be injected into another GPU device', async () => {
  const restore = installWeightFetch(weightFixture({ normX: false }));
  try {
    const a = fakeWeightDevice();
    const b = fakeWeightDevice();
    const weights = await loadWeights(a, 'fixture://weights');
    await assert.rejects(createSf3dProducer({ device: b, weights, workers: {} }), /SF3D weights belong to a different GPUDevice/);
    assert.ok(a.buffers.every(buffer => buffer.destroyed === 0), 'rejection cannot destroy borrowed weights');
    releaseLoadedWeights(weights);
  } finally { restore(); }
});

test('producer initialization failure releases its loaded weights but not injected ones', async () => {
  const restore = installWeightFetch(weightFixture({ normX: false }));
  try {
    const device = fakeWeightDevice();
    device.createComputePipeline = () => { throw new Error('producer pipeline setup failed'); };
    await assert.rejects(createSf3dProducer({ device, weightsUrl: 'fixture://weights', workers: {} }), /producer pipeline setup failed/);
    assert.ok(device.buffers.length > 0);
    assert.ok(device.buffers.every(buffer => buffer.destroyed === 1), 'failed producer construction cannot strand owned weights');
    const host = fakeWeightDevice();
    const weights = await loadWeights(host, 'fixture://host');
    host.createComputePipeline = device.createComputePipeline;
    await assert.rejects(createSf3dProducer({ device: host, weights, workers: {} }), /producer pipeline setup failed/);
    assert.ok(host.buffers.every(buffer => buffer.destroyed === 0));
    releaseLoadedWeights(weights);
  } finally { restore(); }
});

test('failed CLIP pipeline initialization retries; concurrent calls share only their own device', async () => {
  const restore = installWeightFetch(weightFixture({ normX: false }));
  try {
    const clip = await freshClip();
    const device = fakeWeightDevice();
    const weights = await loadWeights(device, 'fixture://weights');
    device.failClipPipeline = true;
    await assert.rejects(clip.estimateMaterials(device, new Uint8Array(4), 1, 1, weights), /injected pipeline failure/);
    device.failClipPipeline = false;
    await Promise.all([warmClip(clip, device, weights), warmClip(clip, device, weights)]);
    assert.equal(device.clipPipelines.length, 3, 'one failed attempt plus exactly two shared successful pipelines');
    releaseLoadedWeights(weights);
    assert.ok(device.buffers.every(buffer => buffer.destroyed === 1));
  } finally { restore(); }
});

test('partial eager load failure retires allocations even though no weight set was returned', async () => {
  const restore = installWeightFetch(weightFixture());
  try {
    const device = fakeWeightDevice();
    device.failBufferAt = 5;
    await assert.rejects(loadWeights(device, 'fixture://weights'), /injected allocation failure/);
    assert.equal(device.buffers.length, 5);
    assert.ok(device.buffers.every(buffer => buffer.destroyed === 1));
  } finally { restore(); }
});

test('CLIP dispatch keeps per-run device bindings across asynchronous prep boundaries', async () => {
  const restore = installWeightFetch(weightFixture());
  globalThis.GPUMapMode = { READ: 1 };
  try {
    const clip = await freshClip();
    let prepCount = 0;
    let resumePrep;
    const bothPrepping = new Promise(resolve => { resumePrep = resolve; });
    const run = async () => {
      const device = fakeWeightDevice();
      let dispatches = 0;
      const createPipeline = device.createComputePipeline;
      device.createComputePipeline = desc => {
        const pipeline = createPipeline(desc);
        pipeline.getBindGroupLayout = () => ({ device });
        return pipeline;
      };
      const createBuffer = device.createBuffer;
      device.createBuffer = desc => ({ ...createBuffer(desc), async mapAsync() {} });
      device.createBindGroup = desc => {
        assert.equal(desc.layout.device, device, 'pipeline layout belongs to this run');
        for (const entry of desc.entries) assert.equal(entry.resource.buffer.device, device, 'every bound buffer belongs to this run');
        return desc;
      };
      device.createCommandEncoder = () => ({
        copyBufferToBuffer() {}, finish() { return {}; },
        beginComputePass() {
          return { setPipeline(pipeline) { assert.equal(pipeline.device, device); },
            setBindGroup() {}, dispatchWorkgroups() { dispatches++; }, end() {} };
        },
      });
      const borrowed = [];
      const weights = {
        _rawGet() { const buffer = device.createBuffer({ size: 4 }); borrowed.push(buffer); return buffer; },
        _rawGetCPU(name) {
          if (name === prepName) return new Float32Array(768 * 3072);
          if (name.endsWith('class_embedding')) return new Float32Array(768);
          if (name.endsWith('positional_embedding')) return new Float32Array(50 * 768);
          if (name.endsWith('visual.proj')) return new Float32Array(512 * 768);
          return new Float32Array(name.endsWith('.bias') ? 1 : 512);
        },
      };
      const listeners = new Set();
      const worker = {
        addEventListener(type, listener) { if (type === 'message') listeners.add(listener); },
        removeEventListener(type, listener) { if (type === 'message') listeners.delete(listener); },
        postMessage(message) {
          const reply = data => { for (const listener of [...listeners]) listener({ data: { id: message.id, ok: true, ...data } }); };
          if (message.type === 'init') { queueMicrotask(() => reply({ initialized: true })); return; }
          if (++prepCount === 2) resumePrep();
          bothPrepping.then(() => reply({ embeddings: new Float32Array(50 * 768).buffer }));
        },
      };
      const result = await clip.estimateMaterials(device, new Uint8Array(4), 1, 1, weights, { clipPrepWorker: worker });
      assert.equal(result.prepOffloaded, true);
      assert.equal(dispatches, 122, 'all 12 transformer blocks and pre/post normalization dispatched');
      assert.equal(borrowed.length, 148);
      assert.ok(borrowed.every(buffer => buffer.destroyed === 0), 'estimator must not infer ownership of arbitrary injected accessors');
    };
    await Promise.all([run(), run()]);
  } finally { restore(); }
});
