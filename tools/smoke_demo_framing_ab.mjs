import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const [manifestPath, outputRoot, weightsPath, executablePath] = process.argv.slice(2);
if (![manifestPath, outputRoot, weightsPath, executablePath].every(Boolean)) {
  throw new Error('Usage: node tools/smoke_demo_framing_ab.mjs CASES_JSON OUT WEIGHTS CHROME');
}
const repo = path.resolve(import.meta.dirname, '..');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const report = { ok: false, phase: 'initialization', cases: [],
  sourceCommit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).trim(),
  executablePath, weightsPath, requestedRoute: 'createSf3dProducer.run/product-default',
  modelViewer: 'https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js' };
await fs.mkdir(outputRoot, { recursive: true });
const save = () => fs.writeFile(path.join(outputRoot, 'report.json'), JSON.stringify(report, null, 2));
let server, browser;
try {
  await save();
  const cases = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  if (cases.length !== 4 || new Set(cases.map(c => c.id)).size !== 4) throw new Error('Expected four unique cases');
  for (const c of cases) if (sha(await fs.readFile(c.input)) !== c.inputSha256) throw new Error(`Changed input ${c.id}`);
  report.phase = 'weights-hash'; await save();
  // Stream the large weight file rather than making another resident 2 GB copy.
  const { createReadStream } = await import('node:fs');
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(weightsPath)) hash.update(chunk);
  report.weightsSha256 = hash.digest('hex');
  const viewerResponse = await fetch(report.modelViewer);
  if (!viewerResponse.ok) throw new Error(`model-viewer download ${viewerResponse.status}`);
  const viewerBytes = Buffer.from(await viewerResponse.arrayBuffer());
  report.modelViewerSha256 = sha(viewerBytes);
  await fs.writeFile(path.join(outputRoot, 'model-viewer.min.js'), viewerBytes);
  server = await createServer({ root: repo, server: { host: '127.0.0.1', port: 0,
    fs: { allow: [repo, outputRoot, path.dirname(weightsPath)] } },
    plugins: [{ name: 'framing-assay-shell', configureServer(s) {
      s.middlewares.use('/framing-assay', (_req, res) => {
        res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>SF3D framing comparison</title><body></body>');
      });
    } }] });
  await server.listen();
  const origin = `http://127.0.0.1:${server.httpServer.address().port}`;
  for (const c of cases) {
    const item = { ...c, ok: false, phase: 'browser', errors: [] };
    report.cases.push(item); report.phase = c.id; await save();
    browser = await puppeteer.launch({ executablePath, headless: true, protocolTimeout: 0,
      args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--disable-background-timer-throttling', '--disable-renderer-backgrounding'] });
    const page = await browser.newPage();
    page.setDefaultTimeout(0); page.setDefaultNavigationTimeout(0);
    await page.setViewport({ width: 960, height: 900, deviceScaleFactor: 1 });
    page.on('pageerror', e => item.errors.push(e.message));
    page.on('console', m => { if (m.type() === 'error') item.errors.push(m.text()); });
    await page.goto(`${origin}/framing-assay`);
    item.phase = 'inference'; await save();
    console.log(`Starting ${c.id}`);
    const data = await page.evaluate(async ({ image, weights, commit, runId }) => {
      const { createSf3dProducer } = await import('/src/lib/sf3d_producer.js');
      const loadStart = performance.now();
      const producer = await createSf3dProducer({ weightsUrl: weights, commit });
      const loadMs = performance.now() - loadStart;
      const info = producer.adapterInfo;
      if (/swiftshader|llvmpipe/i.test(JSON.stringify(info)) || info.isFallbackAdapter) throw new Error('Software adapter');
      const img = new Image(); img.src = image; await img.decode();
      const t = performance.now();
      const out = await producer.run(img, { runId });
      const wallMs = performance.now() - t;
      if (!(out.numVertices > 0 && out.numFaces > 0)) throw new Error('Empty mesh');
      const bytes = new Uint8Array(out.glb);
      const header = new DataView(out.glb);
      if (bytes.length < 20 || header.getUint32(0, true) !== 0x46546c67 || header.getUint32(8, true) !== bytes.length) throw new Error('Invalid GLB');
      window.resultBlob = URL.createObjectURL(new Blob([bytes], { type: 'model/gltf-binary' }));
      let binary = '';
      for (let offset = 0; offset < bytes.length; offset += 32768) binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
      const result = { glbBase64: btoa(binary), loadMs, wallMs, numVertices: out.numVertices, numFaces: out.numFaces,
        identity: out.identity, backend: producer.backend, adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description },
        routeOptions: out.routeOptions, receiptValidation: out.receiptValidation, stageTimings: out.stageTimings,
        stageSpans: out.stageSpans, inputDimensions: [img.naturalWidth, img.naturalHeight] };
      await producer.dispose().completion;
      producer.device.destroy();
      return result;
    }, { image: `data:image/png;base64,${(await fs.readFile(c.input)).toString('base64')}`,
      weights: `/@fs${weightsPath}`, commit: report.sourceCommit, runId: c.id });
    const bytes = Buffer.from(data.glbBase64, 'base64'); delete data.glbBase64;
    Object.assign(item, data, { glbSha256: sha(bytes), glbBytes: bytes.length, phase: 'render' });
    await fs.writeFile(path.join(outputRoot, `${c.id}.glb`), bytes); await save();
    await page.addScriptTag({ type: 'module', url: `${origin}/@fs${outputRoot}/model-viewer.min.js` });
    await page.evaluate(async () => {
      await customElements.whenDefined('model-viewer');
      document.body.style.cssText = 'margin:0;background:#30363b';
      const mv = document.createElement('model-viewer'); mv.id = 'model'; mv.style.cssText = 'width:960px;height:900px';
      mv.setAttribute('environment-image', 'neutral'); mv.setAttribute('interaction-prompt', 'none');
      mv.setAttribute('camera-orbit', '0deg 75deg 105%');
      const loaded = new Promise((resolve, reject) => { mv.addEventListener('load', resolve, { once: true }); mv.addEventListener('error', () => reject(new Error('GLB render failed')), { once: true }); });
      mv.src = window.resultBlob; document.body.append(mv); await loaded;
    });
    item.views = [];
    for (const angle of [0, 90, 180, 270]) {
      await page.evaluate(a => { const m = document.querySelector('#model'); m.setAttribute('camera-orbit', `${a}deg 75deg 105%`); m.jumpCameraToGoal(); }, angle);
      await new Promise(resolve => setTimeout(resolve, 1000));
      await page.waitForFunction(() => document.querySelector('#model').loaded && document.querySelector('#model').modelIsVisible);
      const file = `${c.id}-${angle}.png`;
      await page.screenshot({ path: path.join(outputRoot, file) }); item.views.push(file);
    }
    if (item.errors.length) throw new Error(item.errors.join('\n'));
    item.phase = 'complete'; item.ok = true; await save();
    console.log(`Completed ${c.id}: ${(item.wallMs / 1000).toFixed(2)}s, ${item.numVertices} vertices, ${item.numFaces} faces`);
    await browser.close(); browser = null;
  }
  report.ok = report.cases.every(c => c.ok); report.phase = 'complete';
} catch (error) {
  report.error = { message: error.message, stack: error.stack }; process.exitCode = 1;
  console.error(error);
} finally {
  await browser?.close(); await server?.close(); await save();
}
