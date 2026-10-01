import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const arg = name => { const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : null; };
const outputDir = arg('--output-dir');
const chrome = arg('--chrome');
assert.ok(outputDir && chrome, '--output-dir and --chrome are required');
fs.mkdirSync(outputDir, { recursive: true });
const report = { ok: false, route: 'native-webgpu-spatial-preview-ab', executable: chrome, phase: 'startup' };
const write = () => fs.writeFileSync(path.join(outputDir, 'report.json'), JSON.stringify(report, null, 2));
write();
let server, browser;
try {
  assert.ok(!chrome.startsWith('/Applications/Google Chrome.app/'), 'independent testing browser required');
  server = await createServer({ server: { host: '127.0.0.1', port: 0 }, configFile: false });
  await server.listen();
  browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
  const page = await browser.newPage();
  await page.goto(server.resolvedUrls.local[0] + 'src/lib/preview_geometry.js');
  report.phase = 'native-decoder'; write();
  report.result = await page.evaluate(async () => {
    const { TriplaneDecoder } = await import('/src/lib/triplane_decoder.js');
    const { decodePreviewMesh } = await import('/src/lib/preview_geometry_gpu.js');
    const { createStorageBuffer, readBuffer } = await import('/src/lib/gpu.js');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('native WebGPU adapter unavailable');
    const device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const buffers = [];
    const upload = data => { const b = createStorageBuffer(device, data); buffers.push(b); return b; };
    const decoder = new TriplaneDecoder(device);
    decoder.init();
    try {
      const size = 16;
      const planes = Float32Array.from({ length: 3 * 40 * size * size }, (_, i) => (i % size) / (size - 1) * 2 - 1);
      const triplanes = upload(planes);
      const layers = out => [120, 64, 64].map((inputDim, index) => {
        const outputDim = index === 2 ? out : 64;
        const weight = new Float32Array(outputDim * inputDim);
        const bias = new Float32Array(outputDim);
        for (let i = 0; i < outputDim; i++) weight[i * inputDim + i % inputDim] = index === 2 ? 2 : 1;
        if (index === 2 && out === 1) bias[0] = 3.3;
        return { weight: upload(weight), bias: upload(bias) };
      });
      const weights = { heads: { density: layers(1), vertex_offset: layers(3) } };
      const whole = await decodePreviewMesh(device, triplanes, decoder, weights, 10, size);
      const samples = [];
      const streamed = await decodePreviewMesh(device, triplanes, decoder, weights, 10, size, {
        layersPerSlab: 3,
        onSlab: sample => samples.push({ layers: sample.completedLayers, samples: sample.completedSamples,
          faces: sample.mesh.numFaces, elapsedMs: sample.elapsedMs }),
      });
      const after = await readBuffer(device, triplanes, planes.byteLength);
      await device.queue.onSubmittedWorkDone();
      return { samples, whole: whole.metrics, streamed: streamed.metrics,
        faces: whole.mesh.numFaces, vertices: whole.mesh.numVertices,
        equalVertices: whole.mesh.vertices.length === streamed.mesh.vertices.length && whole.mesh.vertices.every((v, i) => v === streamed.mesh.vertices[i]),
        equalFaces: whole.mesh.faces.length === streamed.mesh.faces.length && whole.mesh.faces.every((v, i) => v === streamed.mesh.faces[i]),
        inputUnchanged: after.every((v, i) => v === planes[i]), errors };
    } finally { for (const b of buffers) b.destroy(); device.destroy(); }
  });
  assert.ok(report.result.equalVertices && report.result.equalFaces && report.result.inputUnchanged);
  assert.ok(report.result.faces > 0);
  assert.deepEqual(report.result.samples.map(s => s.layers), [3, 6, 9, 11]);
  assert.ok(report.result.samples[0].faces > 0 && report.result.samples[0].faces < report.result.faces);
  assert.deepEqual(report.result.errors, []);
  report.ok = true; report.phase = 'complete'; write();
  console.log(JSON.stringify(report));
} catch (error) { report.error = error.stack; write(); process.exitCode = 1; }
finally { await browser?.close(); await server?.close(); }
