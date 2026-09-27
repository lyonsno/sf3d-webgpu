#!/usr/bin/env node
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import net from 'node:net';
import { execFileSync } from 'node:child_process';
import { createServer } from 'vite';
import { acceptPreviewAssay } from './preview_assay_acceptance.mjs';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const args = process.argv.slice(2);
const value = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const outputDir = path.resolve(value('--output-dir', path.join(os.tmpdir(), 'sf3d-preview-geometry')));
const resolution = Number(value('--resolution', '40'));
const partial = args.includes('--partial');
const intermediateStage = args.includes('--intermediate-stage')
  ? value('--intermediate-stage', '') : null;
const imagePath = path.resolve(value('--image', path.join(root, 'public/demo_chair.png')));
const chromePath = value('--chrome', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');
const reportPath = path.join(outputDir, 'report.json');
const fallbackReportPath = path.join(os.tmpdir(), `sf3d-preview-preflight-${process.pid}.json`);
const report = {
  schema: 'sf3d.preview-geometry-assay.v0',
  ok: false,
  phase: 'preflight',
  requested: { resolution, partial, intermediateStage, imagePath, outputDir },
  source: { repo: root, commit: null, clean: null },
};
let vite;
let browser;
let page;
const write = () => {
  const body = JSON.stringify(report, null, 2) + '\n';
  try {
    fs.mkdirSync(outputDir, { recursive: true });
    fs.writeFileSync(reportPath, body);
    return true;
  } catch (error) {
    fs.writeFileSync(fallbackReportPath, body);
    console.error(`primary report unavailable (${error.message}); preflight report: ${fallbackReportPath}`);
    return false;
  }
};
const initialReportSaved = write();

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
  });
}

async function closeWithin(operation, label) {
  let timer;
  try {
    await Promise.race([
      operation,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} did not settle within 30 seconds`)), 30000);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

try {
  if (!initialReportSaved) throw new Error(`cannot write requested report: ${reportPath}`);
  report.source.commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();
  report.source.clean = execFileSync('git', ['status', '--porcelain'], { cwd: root, encoding: 'utf8' }).trim() === '';
  write();
  if (!report.source.clean) throw new Error('source worktree is dirty; commit before running a source-bound assay');
  if (!Number.isSafeInteger(resolution) || resolution < 2) throw new Error('invalid preview resolution');
  if (partial && intermediateStage) throw new Error('choose partial or intermediate-stage, not both');
  if (!fs.existsSync(imagePath)) throw new Error(`missing image: ${imagePath}`);
  if (!fs.existsSync(path.join(root, 'public/weights.bin'))) throw new Error('missing public/weights.bin');
  if (!fs.existsSync(chromePath)) throw new Error(`missing Chrome: ${chromePath}`);
  report.phase = 'serve'; write();
  const port = await freePort();
  vite = await createServer({ root, server: { host: '127.0.0.1', port, strictPort: true } });
  await vite.listen();
  report.phase = 'browser'; write();
  browser = await puppeteer.launch({
    executablePath: chromePath, headless: false,
    args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-first-run', '--no-default-browser-check'],
  });
  page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text().slice(0, 500)); });
  const url = `http://127.0.0.1:${port}/`;
  report.effective = { url, chromePath, imagePath, resolution, partial, intermediateStage,
    kitVersion: JSON.parse(fs.readFileSync(path.join(root, 'node_modules/@kaminos/webgpu-inference-kit/package.json'))).version };
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
  report.phase = 'model-load'; write();
  await page.waitForFunction(() => {
    const status = document.querySelector('#status')?.textContent || '';
    if (status.startsWith('Error:')) throw new Error(status);
    return status.includes('Ready.') && window._sf3d_device;
  }, { timeout: 300000 });
  const imageB64 = fs.readFileSync(imagePath).toString('base64');
  await page.evaluate(async (b64) => {
    await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => { window._previewImage = image; resolve(); };
      image.onerror = () => reject(new Error('preview input image failed to load'));
      image.src = `data:image/png;base64,${b64}`;
    });
  }, imageB64);
  report.phase = 'inference'; write();
  const result = await page.evaluate(async ({ res, partial, intermediateStage }) => {
    const { runInference } = await import('/src/lib/inference.js');
    const { decodePreviewMesh } = await import('/src/lib/preview_geometry_gpu.js');
    const device = window._sf3d_device;
    const input = window._previewImage;
    const before = performance.now();
    let preview;
    let previewAvailableMs = null;
    const partialPreviews = [];
    let intermediatePreview = null;
    const final = await runInference(device, window._sf3d_pipelines, window._sf3d_weights,
      input, () => {}, {
        cooperativePostProcessor: partial,
        cooperativeTwoStream: Boolean(intermediateStage),
        twoStreamDutyGranularity: 'stage',
        intermediateStageId: intermediateStage,
        onIntermediateTriplane: intermediateStage ? async ({ stageId, triplanesBuf, decoder,
          decoderWeights, projectionMs }) => {
          const availableMs = performance.now() - before;
          try {
            const candidate = await decodePreviewMesh(
              device, triplanesBuf, decoder, decoderWeights, res);
            intermediatePreview = { stageId, availableMs, projectionMs,
              metrics: candidate.metrics,
              vertices: Array.from(candidate.mesh.vertices),
              faces: Array.from(candidate.mesh.faces) };
          } catch (error) {
            intermediatePreview = { stageId, availableMs, projectionMs, error: error.message };
          }
        } : null,
        postProcessorDutyGranularity: 'plane',
        onPartialTriplane: partial ? async ({ plane, triplanesBuf, decoder, decoderWeights }) => {
          const availableMs = performance.now() - before;
          try {
            const candidate = await decodePreviewMesh(device, triplanesBuf, decoder, decoderWeights, res);
            partialPreviews.push({ plane, availableMs, metrics: candidate.metrics,
              vertices: Array.from(candidate.mesh.vertices), faces: Array.from(candidate.mesh.faces) });
          } catch (error) {
            partialPreviews.push({ plane, availableMs, error: error.message });
          }
        } : null,
        onFinalTriplane: partial || intermediateStage ? null : async ({ triplanesBuf, decoder, decoderWeights }) => {
          preview = await decodePreviewMesh(device, triplanesBuf, decoder, decoderWeights, res);
          previewAvailableMs = performance.now() - before;
        },
      });
    const inferenceMs = performance.now() - before;
    if (!partial && !intermediateStage && (!preview || previewAvailableMs == null)) {
      throw new Error('final triplane preview callback was not invoked');
    }
    if (partial && partialPreviews.length !== 3) throw new Error(`expected 3 partial previews, got ${partialPreviews.length}`);
    return {
      inferenceMs,
      previewAvailableMs,
      final: { vertices: final.numVertices, faces: final.numFaces, stages: final._stageTimings },
      preview: preview?.metrics ?? null,
      vertices: preview ? Array.from(preview.mesh.vertices) : null,
      faces: preview ? Array.from(preview.mesh.faces) : null,
      partialPreviews,
      intermediatePreview,
    };
  }, { res: resolution, partial, intermediateStage });
  report.phase = 'artifact'; write();
  const writeObj = (name, vertices, faces) => {
    const obj = [];
    for (let i = 0; i < vertices.length; i += 3) obj.push(`v ${vertices[i]} ${vertices[i + 1]} ${vertices[i + 2]}`);
    for (let i = 0; i < faces.length; i += 3) obj.push(`f ${faces[i] + 1} ${faces[i + 1] + 1} ${faces[i + 2] + 1}`);
    fs.writeFileSync(path.join(outputDir, name), obj.join('\n') + '\n');
  };
  const { vertices, faces, partialPreviews, intermediatePreview, ...summary } = result;
  if (vertices) writeObj('preview.obj', vertices, faces);
  const partialSummary = partialPreviews.map(({ vertices: pv, faces: pf, ...state }) => {
    if (pv) writeObj(`plane-${state.plane + 1}.obj`, pv, pf);
    return { ...state, mesh: pv ? { vertices: pv.length / 3, faces: pf.length / 3 } : null };
  });
  let intermediateSummary = null;
  if (intermediatePreview) {
    const { vertices: iv, faces: iff, ...state } = intermediatePreview;
    if (iv) writeObj('intermediate.obj', iv, iff);
    intermediateSummary = { ...state, mesh: iv ? { vertices: iv.length / 3, faces: iff.length / 3 } : null };
  }
  report.result = { ...summary, partialPreviews: partialSummary,
    intermediatePreview: intermediateSummary,
    previewMesh: vertices ? { vertices: vertices.length / 3, faces: faces.length / 3 } : null,
    browserErrors: errors };
  try {
    acceptPreviewAssay(report);
    report.integrityOk = true;
  } catch (error) {
    report.integrityOk = false;
    report.acceptanceError = error.message;
  }
  report.phase = 'teardown';
  write();
  console.log(JSON.stringify(report.result, null, 2));
} catch (error) {
  report.error = { message: error.message, stack: error.stack };
  write();
  console.error(`${report.phase}: ${error.stack || error}`);
  process.exitCode = 1;
} finally {
  const teardownErrors = [];
  try {
    if (browser) await closeWithin(browser.close(), 'Chrome close');
  } catch (error) {
    browser?.process()?.kill('SIGTERM');
    teardownErrors.push(error.message);
  }
  try {
    if (vite) await closeWithin(vite.close(), 'Vite close');
  } catch (error) {
    teardownErrors.push(error.message);
  }
  if (teardownErrors.length) {
    report.teardownErrors = teardownErrors;
    report.ok = false;
    report.phase = 'teardown';
    process.exitCode = 1;
  } else if (!report.error) {
    report.ok = report.integrityOk;
    report.phase = report.ok ? 'complete' : 'candidate-failed';
    if (!report.ok) process.exitCode = 1;
  }
  write();
}
