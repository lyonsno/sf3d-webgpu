import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {webcrypto} from 'node:crypto';

// Execute the actual browser callback with synthetic resources. This is a
// consumer-order/borrowed-custody test, not model arithmetic or native fit.
const source = fs.readFileSync(new URL('./smoke_resident_dino.mjs', import.meta.url), 'utf8');
const startText = 'Object.assign(report,await page.evaluate(async config=>{';
const start = source.indexOf(startText), end = source.indexOf('\n  },{requested:report.requested', start);
assert(start >= 0 && end > start, 'actual native callback source required');
const body = source.slice(start + startText.length, end)
  .replace("await import('/dist-lib/sf3d-producer.js')", '$module');

async function exercise({failRetirement = false} = {}) {
  const events = []; let backbone, retired = false, postComplete = false, deviceDestroyed = false;
  const postBuffer = {size: 4, destroyed: false};
  const device = {
    queue: {submit() {}, async onSubmittedWorkDone() {events.push('prefix');}},
    pushErrorScope() {}, async popErrorScope() {return null;},
    destroy() {deviceDestroyed = true; events.push('device-destroy');},
    createCommandEncoder() {return {copyBufferToBuffer() {}, finish() {return {};}};},
    createBuffer() {return {size: 4, async mapAsync() {}, getMappedRange() {return new ArrayBuffer(4);},
      unmap() {}, destroy() {events.push('readback-destroy');}};},
  };
  const adapter = {info: {vendor: 'apple', architecture: 'metal-3', isFallbackAdapter: false},
    limits: {maxStorageBufferBindingSize: 1, maxBufferSize: 1}};
  const budget = {async requestOwnedDevice() {return device;}, reserveCpu() {return {release() {}};},
    snapshot() {return {};}, restore() {}};
  const module = {
    createLoaderMemoryBudget() {return budget;},
    TwoStreamBackbone: class {constructor(value) {assert.equal(value, device); backbone = this;} init() {}},
    async preprocessImage() {return new Float32Array([1]);},
    async preprocessConditionImage() {return {chw: new Float32Array([1]), rgba: new Uint8Array(4)};},
    async runResidentDino(options) {return {value: await options.withResult({tokensBuf: {size: 4}, N: 1297})};},
    async runResidentTwoStream(options) {try {return {value: await options.withResult({buffer: {size: 4}})};}
      finally {await module.disposeResidentTwoStream(backbone);}},
    async runResidentPostProcessor(options) {assert.equal(retired, false, 'input must survive all postprocessor work');
      postComplete = true; events.push('complete-post-prefix');
      return {value: await options.withResult({buffer: postBuffer})};},
    async disposeResidentTwoStream(value) {
      assert.equal(value, backbone); assert.equal(postComplete, true);
      if (retired) return;
      if (failRetirement) throw Error('injected upstream retirement failure');
      retired = true; events.push('retire-backbone');
    },
    async runResidentArtifact(options) {
      assert.equal(retired, true, 'obsolete backbone must retire before complete artifact allocation');
      assert.equal(options.triplanesBuf, postBuffer); assert.equal(postBuffer.destroyed, false);
      assert.equal(deviceDestroyed, false); events.push('artifact'); return {complete: true};
    },
    async disposeResidentArtifact() {assert.equal(postBuffer.destroyed, false);},
  };
  const context = vm.createContext({crypto: webcrypto, document: {querySelector() {return {textContent: ''};}},
    navigator: {gpu: {async requestAdapter() {return adapter;}}},
    GPUBufferUsage: {COPY_DST: 1, MAP_READ: 2}, GPUMapMode: {READ: 1},
    async createImageBitmap() {return {width: 503, height: 503, close() {}};},
    async fetch(url) {events.push('fetch:' + url); return {ok: true, async blob() {return {};}};},
  });
  const callback = vm.compileFunction('return async config => {' + body + '\n};', ['$module'],
    {parsingContext: context})(module);
  const config = {requested: {throughBackbone: true, throughPostProcessor: true, throughFullModel: true,
    attentionRowsPerDuty: 32, postChannelsPerDuty: 16, reuseDeadTriplaneStorage: true, reuseAttentionResidualStorage: true},
    input: {width: 503, height: 503}, source: {byteLength: 2285308688, etag: 'exact-source'}, allocatingDutyIndices: []};
  if (failRetirement) {
    await assert.rejects(callback(config), /injected upstream retirement failure/);
    assert.equal(events.includes('artifact'), false);
  } else {
    await callback(config);
    assert(events.indexOf('fetch:/postprocessor.f32') < events.indexOf('retire-backbone'));
    assert(events.indexOf('retire-backbone') < events.indexOf('artifact'));
    assert.equal(events.filter(x => x === 'retire-backbone').length, 1);
  }
  assert.equal(postBuffer.destroyed, false); assert.equal(deviceDestroyed, true);
}
await exercise();
console.log('ok actual native callback retires only obsolete owned input after complete post consumption/readback');
await exercise({failRetirement: true});
console.log('ok failed upstream retirement prevents downstream artifact launch without early borrowed disposal');
