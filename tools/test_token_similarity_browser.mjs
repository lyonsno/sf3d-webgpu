import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
const arg = name => process.argv[process.argv.indexOf(name) + 1];
const outputDir = path.resolve(arg('--output-dir'));
const chrome = arg('--chrome');
fs.mkdirSync(outputDir, { recursive: true });
const report = { ok: false, route: 'native-webgpu-token-similarity', executable: chrome, phase: 'startup' };
const write = () => fs.writeFileSync(path.join(outputDir, 'report.json'), JSON.stringify(report, null, 2));
write();
let server, browser;
try {
  assert.ok(chrome && !chrome.startsWith('/Applications/Google Chrome.app/'), 'an independent testing browser is required');
  server = await createServer({ server: { host: '127.0.0.1', port: 0 }, configFile: false });
  await server.listen();
  browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
  const page = await browser.newPage();
  // A module host, not the model app; no weights or inference are loaded.
  await page.goto(server.resolvedUrls.local[0] + 'src/lib/token_similarity.js');
  report.phase = 'gpu-fixture'; write();
  report.result = await page.evaluate(async () => {
    const { readTokenSimilarity } = await import('/src/lib/token_similarity.js');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No native WebGPU adapter');
    const device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const dim = 67, count = 6;
    const tokens = Float32Array.from({ length: (count + 1) * dim }, (_, i) => Math.sin(i * 0.31));
    tokens.fill(1000, 0, dim); // CLS must be ignored.
    tokens.fill(0, 3 * dim, 4 * dim); // Zero-vector patch.
    for (let i = 0; i < dim; i++) tokens[5 * dim + i] = -tokens[dim + i];
    const input = device.createBuffer({ size: tokens.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
    new Float32Array(input.getMappedRange()).set(tokens); input.unmap();
    try {
      const actual = await readTokenSimilarity(device, input, { dim, width: 3, height: 2, patchIndex: 0 });
      const expected = Array.from({ length: count }, (_, n) => {
        let dot = 0, aa = 0, bb = 0;
        for (let d = 0; d < dim; d++) {
          const a = tokens[(n + 1) * dim + d], b = tokens[dim + d];
          dot += a * b; aa += a * a; bb += b * b;
        }
        return aa && bb ? dot / Math.sqrt(aa * bb) : 0;
      });
      const readback = device.createBuffer({ size: input.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(input, 0, readback, 0, input.size);
        device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
        const after = new Float32Array(readback.getMappedRange());
        return { actual: [...actual], expected, unchanged: after.every((v, i) => v === tokens[i]), errors };
      } finally { readback.destroy(); }
    } finally { input.destroy(); device.destroy(); }
  });
  assert.equal(report.result.actual.length, 6);
  assert.ok(report.result.actual.every((v, i) => Number.isFinite(v) && Math.abs(v - report.result.expected[i]) < 2e-6));
  assert.equal(report.result.unchanged, true);
  assert.deepEqual(report.result.errors, []);
  report.ok = true; report.phase = 'complete'; write();
  console.log(JSON.stringify(report));
} catch (error) { report.error = error.stack; write(); process.exitCode = 1; }
finally { await browser?.close(); await server?.close(); }
