import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
import { execFileSync } from 'node:child_process';

const arg = name => { const i = process.argv.indexOf(name); return i < 0 ? null : process.argv[i + 1]; };
const outputDir = arg('--output-dir'), chrome = arg('--chrome');
assert.ok(outputDir && chrome);
fs.mkdirSync(outputDir, { recursive: true });
const report = { ok: false, phase: 'startup', executable: chrome,
  sourceCommit: null,
  route: 'native-webgpu-spatial-postprocessor' };
const write = () => fs.writeFileSync(path.join(outputDir, 'report.json'), JSON.stringify(report, null, 2));
write();
let server, browser;
try {
  report.phase = 'source-preflight'; write();
  const git = (...args) => execFileSync('git', args, { encoding: 'utf8' }).trim();
  report.sourceCommit = git('rev-parse', 'HEAD');
  report.sourceStatus = git('status', '--porcelain', '--untracked-files=all');
  if (report.sourceStatus) throw new Error('native parity requires clean committed source');
  if (!process.argv.includes('--source-preflight-only')) {
  assert.ok(!chrome.startsWith('/Applications/Google Chrome.app/'));
  server = await createServer({ configFile: false, server: { host: '127.0.0.1', port: 0 } });
  await server.listen();
  browser = await puppeteer.launch({ executablePath: chrome, headless: true,
    args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
  const page = await browser.newPage();
  await page.goto(server.resolvedUrls.local[0] + 'src/lib/gpu.js');
  report.phase = 'row-kernel'; write();
  report.result = await page.evaluate(async () => {
    const gpu = await import('/src/lib/gpu.js');
    const ops = await import('/src/lib/shader_ops.js');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener('uncapturederror', e => errors.push(e.error.message));
    const check = (truth, message) => { if (!truth) throw new Error(message); };
    const equal = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
    try {
      const params = { inC: 3, outC: 5, inH: 19, inW: 17, kH: 3, kW: 3,
        padH: 1, padW: 1, strideH: 1, strideW: 1, applyRelu: true };
      const input = gpu.createStorageBuffer(device, Float32Array.from({ length: 3 * 19 * 17 }, (_, i) => Math.sin(i * .3)));
      const weight = gpu.createStorageBuffer(device, Float32Array.from({ length: 5 * 3 * 9 }, (_, i) => Math.cos(i * .7) / 9));
      const bias = gpu.createStorageBuffer(device, new Float32Array([.1, -.2, .3, -.4, .5]));
      let encoder = device.createCommandEncoder();
      const whole = ops.dispatchConv2d(device, encoder, input, weight, bias, params);
      const activated = ops.dispatchActivation(device, encoder, whole.buffer, null, 5 * 19 * 17, 0);
      const output = gpu.createStorageBuffer(device, new Float32Array(5 * 19 * 17).fill(-999), GPUBufferUsage.COPY_DST);
      ops.dispatchConv2dChannelRange(device, encoder, input, weight, bias, output, params,
        { channelStart: 0, channelCount: 5, rowStart: 3, rowCount: 4 });
      device.queue.submit([encoder.finish()]);
      const expected = await gpu.readBuffer(device, activated, output.size);
      const partial = await gpu.readBuffer(device, output, output.size);
      for (let c = 0; c < 5; c++) for (let y = 0; y < 19; y++) for (let x = 0; x < 17; x++) {
        const i = (c * 19 + y) * 17 + x;
        check(partial[i] === (y >= 3 && y < 7 ? expected[i] : -999),
          `row frontier violated at channel ${c}, row ${y}, column ${x}`);
      }
      encoder = device.createCommandEncoder();
      for (const [rowStart, rowCount] of [[0, 3], [7, 12]]) {
        ops.dispatchConv2dChannelRange(device, encoder, input, weight, bias, output, params,
          { channelStart: 0, channelCount: 5, rowStart, rowCount });
      }
      device.queue.submit([encoder.finish()]);
      check(equal(expected, await gpu.readBuffer(device, output, output.size)), 'split convolution differs from whole');
      const { streamPostProcessor } = await import('/src/lib/post_processor_spatial.js');
      const config = { channels: 3, outChannels: 2, size: 19, scale: 2 };
      const pixels = config.size ** 2, width = config.size * config.scale;
      const source = Float32Array.from({ length: 3 * config.channels * pixels }, (_, i) => Math.sin(i * .07));
      const sourceBuffer = gpu.createStorageBuffer(device, source);
      const convLayers = Array.from({ length: 4 }, (_, layer) => {
        const out = layer === 3 ? 8 : 3;
        return {
          weight: gpu.createStorageBuffer(device, Float32Array.from({ length: out * 3 * 9 }, (_, i) => Math.cos(i * .27 + layer) / 5)),
          bias: gpu.createStorageBuffer(device, Float32Array.from({ length: out }, (_, i) => Math.sin(i + layer) * .1)),
        };
      });
      const reference = [];
      for (let plane = 0; plane < 3; plane++) {
        let current = gpu.createStorageBuffer(device, Float32Array.from({ length: config.channels * pixels },
          (_, i) => source[Math.floor(i / pixels) * 3 * pixels + plane * pixels + i % pixels]));
        encoder = device.createCommandEncoder();
        for (let layer = 0; layer < 4; layer++) {
          const outC = layer === 3 ? 8 : 3;
          current = ops.dispatchConv2d(device, encoder, current, convLayers[layer].weight, convLayers[layer].bias,
            { ...params, outC, inH: 19, inW: 19 }).buffer;
          if (layer < 3) current = ops.dispatchActivation(device, encoder, current, null, outC * pixels, 0);
        }
        current = ops.dispatchPixelShuffle(device, encoder, current, { inC: 8, inH: 19, inW: 19, scaleFactor: 2 }).buffer;
        device.queue.submit([encoder.finish()]);
        reference.push(await gpu.readBuffer(device, current, current.size));
      }
      const regions = [];
      await streamPostProcessor(device, sourceBuffer, { convLayers }, { config, rowsPerRegion: 5,
        onRegion: async ({ buffer, completedRows }) => {
          const actual = await gpu.readBuffer(device, buffer, buffer.size);
          for (let plane = 0; plane < 3; plane++) for (let c = 0; c < 2; c++) {
            for (let y = 0; y < width; y++) for (let x = 0; x < width; x++) {
              const i = (c * width + y) * width + x;
              const expected = plane === 0 || y < completedRows ? reference[plane][i] : 0;
              check(actual[plane * 2 * width * width + i] === expected,
                `spatial network differs at plane ${plane}, row ${y}, frontier ${completedRows}: ${actual[plane * 2 * width * width + i]} vs ${expected}; GPU errors: ${errors.join('; ')}`);
            }
          }
          regions.push(completedRows);
        },
      });
      check(equal(regions, [10, 20, 30, 38]), 'missing spatial observations');
      await device.queue.onSubmittedWorkDone();
      check(errors.length === 0, errors.join('\n'));
      return { rowKernelExact: true, untouchedRowsPreserved: true, fullNetworkExact: true, regions, errors };
    } finally { device.destroy(); }
  });
  }
  report.phase = 'source-postflight'; write();
  assert.equal(git('rev-parse', 'HEAD'), report.sourceCommit, 'source revision changed during parity test');
  assert.equal(git('status', '--porcelain', '--untracked-files=all'), '', 'source changed during parity test');
  report.ok = true; report.phase = 'complete'; write();
  console.log(JSON.stringify(report.result));
} catch (error) { report.error = error.stack; write(); console.error(error); process.exitCode = 1; }
finally { await browser?.close(); await server?.close(); }
