#!/usr/bin/env node

import crypto from 'node:crypto';
import { execFileSync, execSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import puppeteer from 'puppeteer-core';

import {
  DINO_ASSAY_SCHEMA,
  validateDinoWeightRepresentationAssay,
} from './dino_weight_representation_assay_contract.mjs';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const EXPECTED_KIT_VERSION = '0.1.47';
const EXPECTED_KIT_TARBALL_SHA256 = '6d1fd5d778fa5c964393b5f53717c08c118c9a42720ce00aea3d888b2b587535';
const EXPECTED_WEIGHT_SHA256 = '0e5c23c8c502492c0b4432006ac30f2d11b2a4c102a3bf9f21c2fcf094ddbffd';
const REPRESENTATIONS = ['f32-expanded', 'f16-packed-u32'];

function argValue(name, fallback) {
  const index = process.argv.indexOf(name);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

const WEIGHTS = path.resolve(argValue(
  '--weights',
  '/private/tmp/hf-cache-cranial-sf3d-0901/models--BasinShapers--sf3d-webgpu-weights/blobs/'
    + EXPECTED_WEIGHT_SHA256,
));
const INPUT = path.resolve(argValue(
  '--image',
  path.join(os.homedir(), '.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png'),
));
const KIT_TARBALL = path.resolve(argValue(
  '--kit-tarball',
  '/private/tmp/kaminos-weight-representation-0147-final.7x9eMW/'
    + 'kaminos-webgpu-inference-kit-0.1.47.tgz',
));
const REPORT_PATH = path.resolve(argValue(
  '--report',
  '/private/tmp/sf3d-dino-weight-representation-browser-assay/report.json',
));

const report = {
  schema: DINO_ASSAY_SCHEMA,
  ok: false,
  phase: 'init',
  startedAt: new Date().toISOString(),
  source: {},
  arms: {},
  comparison: null,
  failure: null,
};

function writeReport() {
  report.updatedAt = new Date().toISOString();
  fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  const temporary = `${REPORT_PATH}.tmp-${process.pid}`;
  fs.writeFileSync(temporary, `${JSON.stringify(report, null, 2)}\n`);
  fs.renameSync(temporary, REPORT_PATH);
}

function fail(phase, error, details = {}) {
  report.ok = false;
  report.phase = phase;
  report.failure = {
    phase,
    message: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : null,
    ...details,
  };
  writeReport();
  throw error instanceof Error ? error : new Error(String(error));
}

function sha256File(file) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(file);
    stream.on('data', chunk => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
    stream.on('error', reject);
  });
}

function sha256Buffer(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

function allocatePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

async function launchVite(port) {
  const viteProcess = spawn('npx', [
    'vite',
    '--config',
    'tools/vite_dino_assay_config.mjs',
    '--host',
    '127.0.0.1',
    '--port',
    String(port),
    '--strictPort',
  ], {
    cwd: REPO,
    env: { ...process.env, SF3D_ASSAY_WEIGHTS_PATH: WEIGHTS },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let output = '';
  const append = chunk => { output = `${output}${chunk}`.slice(-16_384); };
  viteProcess.stdout.on('data', append);
  viteProcess.stderr.on('data', append);
  await new Promise((resolve, reject) => {
    const onData = chunk => {
      if (/Local:|ready in/i.test(String(chunk))) {
        cleanup();
        resolve();
      }
    };
    const onExit = code => {
      cleanup();
      reject(new Error(`Vite exited before serving (code ${code}): ${output}`));
    };
    const cleanup = () => {
      viteProcess.stdout.off('data', onData);
      viteProcess.stderr.off('data', onData);
      viteProcess.off('exit', onExit);
    };
    viteProcess.stdout.on('data', onData);
    viteProcess.stderr.on('data', onData);
    viteProcess.on('exit', onExit);
  });
  return viteProcess;
}

async function runArm(baseUrl, representation) {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), `sf3d-dino-${representation}-`));
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: false,
    userDataDir: profile,
    args: [
      '--enable-unsafe-webgpu',
      '--use-angle=metal',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
    ],
  });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') pageErrors.push(`console: ${message.text()}`);
  });

  try {
    const loadStartedAt = performance.now();
    const url = `${baseUrl}?imageTokenizerWeightRepresentation=${encodeURIComponent(representation)}`;
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 0 });
    await page.evaluate(() => new Promise((resolve, reject) => {
      const status = document.querySelector('#status');
      if (!status) {
        reject(new Error('status element missing'));
        return;
      }
      const inspect = () => {
        const text = status.textContent || '';
        if (text.includes('Ready')) resolve();
        else if (text.startsWith('Error:')) reject(new Error(text));
      };
      inspect();
      const observer = new MutationObserver(() => {
        inspect();
        if ((status.textContent || '').includes('Ready') || (status.textContent || '').startsWith('Error:')) {
          observer.disconnect();
        }
      });
      observer.observe(status, { childList: true, subtree: true, characterData: true });
    }));
    const loadWallMs = performance.now() - loadStartedAt;

    const imageBase64 = fs.readFileSync(INPUT).toString('base64');
    const result = await page.evaluate(async ({ imageBase64 }) => {
      const { preprocessImage, computeCameraInput } = await import('/src/lib/inference.js');
      const { createStorageBuffer, createEmptyBuffer, readBuffer } = await import('/src/lib/gpu.js');
      const { runCooperativeDino } = await import('/src/lib/cooperative_dino.js');

      const device = window._sf3d_device;
      const weights = window._sf3d_weights;
      const tokenizer = window._sf3d_pipelines?.imageTokenizer;
      if (!device || !weights || !tokenizer) throw new Error('SF3D model globals missing');

      const image = await new Promise((resolve, reject) => {
        const element = new Image();
        element.onload = () => resolve(element);
        element.onerror = () => reject(new Error('input image failed to decode'));
        element.src = `data:image/png;base64,${imageBase64}`;
      });

      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      const adapterInfo = adapter?.info
        || (adapter?.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
      const imageData = await preprocessImage(image, image.naturalWidth, image.naturalHeight);
      const imageBuffer = createStorageBuffer(device, imageData);
      const cameraInputBuffer = createStorageBuffer(device, computeCameraInput());
      const cameraEmbedBuffer = createEmptyBuffer(device, 768 * 4);
      const cameraEncoder = device.createCommandEncoder();
      tokenizer._dispatchLinear(
        cameraEncoder,
        cameraInputBuffer,
        cameraEmbedBuffer,
        weights.cameraEmbedder.weight,
        weights.cameraEmbedder.bias,
        1,
        25,
        768,
      );
      device.queue.submit([cameraEncoder.finish()]);

      const startedAt = performance.now();
      const { result: dino, report: cooperative } = await runCooperativeDino({
        device,
        tokenizer,
        imageBuf: imageBuffer,
        cameraEmbedBuf: cameraEmbedBuffer,
        weights: weights.imageTokenizer,
        numBlocks: 24,
        chunkBlocks: 1,
        schedulingMode: 'cooperative',
      });
      const tokens = await readBuffer(device, dino.tokensBuf, dino.N * 1024 * 4);
      const dinoWallMs = performance.now() - startedAt;
      const bytes = new Uint8Array(tokens.buffer, tokens.byteOffset, tokens.byteLength);
      const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
      const sha256 = [...digest].map(byte => byte.toString(16).padStart(2, '0')).join('');
      let finiteCount = 0;
      for (const value of tokens) if (Number.isFinite(value)) finiteCount++;
      let binary = '';
      for (let offset = 0; offset < bytes.length; offset += 0x8000) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
      }

      return {
        effectiveRepresentation: weights.weightRepresentations.imageTokenizerMatrices.representation,
        matrixStats: weights.weightRepresentations.imageTokenizerMatrices,
        adapter: {
          vendor: adapterInfo.vendor || null,
          architecture: adapterInfo.architecture || null,
          device: adapterInfo.device || null,
          description: adapterInfo.description || null,
        },
        browser: navigator.userAgent,
        dinoWallMs,
        cooperative: {
          status: cooperative.status,
          completedItems: cooperative.progress?.completedItems ?? null,
          totalItems: cooperative.progress?.totalItems ?? null,
          queueCompletionAuthority: cooperative.queueCompletionAuthority,
        },
        output: {
          shape: { N: dino.N, dim: 1024 },
          elementCount: tokens.length,
          byteLength: tokens.byteLength,
          finiteCount,
          sha256,
        },
        tokensBase64: btoa(binary),
      };
    }, { imageBase64 });

    const tokens = Buffer.from(result.tokensBase64, 'base64');
    delete result.tokensBase64;
    if (result.output.sha256 !== sha256Buffer(tokens)) {
      throw new Error(`${representation} output changed while crossing the browser boundary`);
    }
    return {
      receipt: {
        requestedRepresentation: representation,
        ...result,
        loadWallMs,
        pageErrors: [...pageErrors],
      },
      tokens,
    };
  } finally {
    await browser.close().catch(() => {});
    fs.rmSync(profile, { recursive: true, force: true });
  }
}

async function main() {
  writeReport();
  let vite = null;
  try {
    report.phase = 'source-verification';
    writeReport();
    for (const required of [CHROME_PATH, WEIGHTS, INPUT, KIT_TARBALL]) {
      if (!fs.existsSync(required)) fail('source-verification', `required input missing: ${required}`);
    }
    const clean = execSync('git status --porcelain', { cwd: REPO }).toString().trim();
    if (clean) fail('source-verification', 'assay must run from a clean committed checkout', { dirty: clean });
    const kitVersion = JSON.parse(fs.readFileSync(
      path.join(REPO, 'node_modules/@kaminos/webgpu-inference-kit/package.json'),
    )).version;
    const source = {
      checkout: REPO,
      commit: execSync('git rev-parse HEAD', { cwd: REPO }).toString().trim(),
      kitVersion,
      kitTarball: KIT_TARBALL,
      kitTarballSha256: await sha256File(KIT_TARBALL),
      weightsPath: WEIGHTS,
      weightsBytes: fs.statSync(WEIGHTS).size,
      weightsSha256: await sha256File(WEIGHTS),
      inputPath: INPUT,
      inputSha256: await sha256File(INPUT),
      chromeExecutable: CHROME_PATH,
      chromeVersion: execFileSync(CHROME_PATH, ['--version']).toString().trim(),
      greenroomLeaseId: process.env.GPU_GREENROOM_LEASE_ID || null,
    };
    report.source = source;
    if (kitVersion !== EXPECTED_KIT_VERSION) {
      fail('source-verification', `kit ${kitVersion} != ${EXPECTED_KIT_VERSION}`);
    }
    if (source.kitTarballSha256 !== EXPECTED_KIT_TARBALL_SHA256) {
      fail(
        'source-verification',
        `kit tarball hash ${source.kitTarballSha256} != ${EXPECTED_KIT_TARBALL_SHA256}`,
      );
    }
    if (source.weightsSha256 !== EXPECTED_WEIGHT_SHA256) {
      fail('source-verification', `weights hash ${source.weightsSha256} != ${EXPECTED_WEIGHT_SHA256}`);
    }
    if (!source.greenroomLeaseId) {
      fail('source-verification', 'GPU_GREENROOM_LEASE_ID is required for an acceptance run');
    }
    writeReport();

    const port = await allocatePort();
    vite = await launchVite(port);
    report.source.server = { pid: vite.pid, url: `http://127.0.0.1:${port}/` };
    report.phase = 'browser-arms';
    writeReport();

    const tokenBuffers = {};
    for (const representation of REPRESENTATIONS) {
      const { receipt, tokens } = await runArm(report.source.server.url, representation);
      report.arms[representation] = receipt;
      tokenBuffers[representation] = tokens;
      writeReport();
    }

    const expanded = tokenBuffers['f32-expanded'];
    const packed = tokenBuffers['f16-packed-u32'];
    let firstDifferingByte = null;
    const byteLength = Math.max(expanded.length, packed.length);
    for (let index = 0; index < byteLength; index++) {
      if (expanded[index] !== packed[index]) {
        firstDifferingByte = index;
        break;
      }
    }
    let firstDifferingIndex = null;
    let maxAbsDelta = 0;
    if (expanded.length === packed.length) {
      const expandedFloats = new Float32Array(expanded.buffer, expanded.byteOffset, expanded.length / 4);
      const packedFloats = new Float32Array(packed.buffer, packed.byteOffset, packed.length / 4);
      for (let index = 0; index < expandedFloats.length; index++) {
        const delta = Math.abs(expandedFloats[index] - packedFloats[index]);
        if (delta !== 0 && firstDifferingIndex == null) firstDifferingIndex = index;
        if (delta > maxAbsDelta) maxAbsDelta = delta;
      }
    }
    report.comparison = {
      byteIdentical: expanded.equals(packed),
      firstDifferingByte,
      firstDifferingIndex,
      maxAbsDelta,
      packedStorageBytesSaved: (
        report.arms['f16-packed-u32'].matrixStats.savedVsExpandedFp32ByteLength
      ),
    };
    report.ok = true;
    report.phase = 'complete';
    report.completedAt = new Date().toISOString();
    validateDinoWeightRepresentationAssay(report, source);
    writeReport();
    console.log(JSON.stringify({
      ok: true,
      report: REPORT_PATH,
      source: report.source,
      arms: report.arms,
      comparison: report.comparison,
    }, null, 2));
  } catch (error) {
    if (!report.failure) {
      try { fail(report.phase || 'unknown', error); } catch {}
    }
    console.error(`DINO representation assay failed: ${error.stack || error}`);
    process.exitCode = 1;
  } finally {
    if (vite) vite.kill();
  }
}

await main();
