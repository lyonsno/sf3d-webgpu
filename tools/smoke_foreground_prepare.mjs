import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const [executablePath, output] = process.argv.slice(2);
if (!executablePath || !output) throw new Error('Usage: node tools/smoke_foreground_prepare.mjs CHROME REPORT');
const report = { ok: false, phase: 'start', executablePath };
await fs.mkdir(path.dirname(output), { recursive: true });
let server, browser;
try {
  server = await createServer({ server: { host: '127.0.0.1', port: 0 }, plugins: [{ name: 'cpu-preparation-smoke', configureServer(s) {
    s.middlewares.use('/cpu-prep', (_req, res) => { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>CPU image preparation check</title>'); });
  } }] });
  await server.listen();
  report.url = `http://127.0.0.1:${server.httpServer.address().port}/cpu-prep`;
  browser = await puppeteer.launch({ executablePath, headless: true });
  const page = await browser.newPage();
  await page.goto(report.url);
  report.phase = 'preprocessing';
  const fixture = JSON.parse(await fs.readFile(new URL('./fixtures/foreground-reference.json', import.meta.url)));
  report.result = await page.evaluate(async (fixtures) => {
    const { prepareForeground } = await import('/src/lib/foreground_prepare.js');
    const { preprocessImage } = await import('/src/lib/inference.js');
    const { callWorker } = await import('/src/lib/worker_call.js');
    const { validateConditionReply } = await import('/src/lib/condition_image.js');
    const worker = new Worker('/src/lib/preprocess_worker.js', { type: 'module' });
    try {
      const comparisons = [];
      for (const c of fixtures.cases) {
        const result = prepareForeground(Uint8Array.from(c.input), c.width, c.height, c.size);
        if (result.data.some((v, i) => v !== c.output[i])) throw new Error('Pillow pixel mismatch: ' + c.name);
        comparisons.push(c.name);
      }
      const img = new Image(); img.src = '/demo_chair.png'; await img.decode();
      const inline = await preprocessImage(img);
      const offloaded = await preprocessImage(img, undefined, undefined, { preprocessWorker: worker });
      if (inline.length !== 3 * 512 * 512 || inline.some((v, i) => v !== offloaded[i])) throw new Error('Main/worker CHW mismatch');
      const source = new Float32Array(4 * 2 * 2);
      let rejected = false;
      try {
        await callWorker(worker, { id: 'empty', srcBuffer: source.buffer, srcW: 2, srcH: 2, size: 512,
          bg: [.5,.5,.5], imageMean: [0,0,0], imageStd: [1,1,1] }, [source.buffer],
          { onResult: reply => validateConditionReply(reply, 512) });
      } catch (error) { if (!/Empty foreground mask/.test(error.message)) throw error; rejected = true; }
      if (!rejected) throw new Error('Empty source accepted');
      const again = await preprocessImage(img, undefined, undefined, { preprocessWorker: worker });
      if (inline.some((v, i) => v !== again[i])) throw new Error('Worker did not recover after rejected input');
      return { comparisons, elements: inline.length, workerMatches: true, emptyRejected: true, reuseAfterFailure: true };
    } finally { worker.terminate(); }
  }, fixture);
  assert.equal(report.result.workerMatches, true);
  report.ok = true; report.phase = 'complete';
} catch (error) {
  report.error = error.stack; process.exitCode = 1;
} finally {
  await browser?.close(); await server?.close();
  await fs.writeFile(output, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
}
