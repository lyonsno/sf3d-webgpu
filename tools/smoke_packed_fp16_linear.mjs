#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import puppeteer from 'puppeteer-core';

import { evaluatePackedFp16LinearSmokeReport } from '../src/lib/packed_fp16_linear_assay.js';

const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const argValue = (flag, fallback = '') => {
  const index = process.argv.indexOf(flag);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
};
const REPORT_PATH = path.resolve(argValue('--report', '/private/tmp/sf3d-packed-fp16-linear-smoke.json'));
const KIT_TARBALL_ARG = argValue('--kit-tarball');
const KIT_TARBALL = KIT_TARBALL_ARG ? path.resolve(KIT_TARBALL_ARG) : null;
const KIT_PRODUCER_REVISION = argValue('--kit-producer-revision');
const rows = Number(argValue('--rows', '3'));
const inDim = Number(argValue('--in-dim', '5'));
const outDim = Number(argValue('--out-dim', '7'));
const children = [];
let browser = null;
let lastTrustworthyEvidence = { phase: 'argument-parse' };

const sha256 = data => crypto.createHash('sha256').update(data).digest('hex');
const sha256File = file => sha256(fs.readFileSync(file));
const git = args => execFileSync('git', args, { cwd: REPO, encoding: 'utf8' }).trim();

function sourceIdentity() {
  const dirtyPaths = git(['status', '--short']).split('\n').filter(Boolean);
  const hash = crypto.createHash('sha256');
  hash.update(execFileSync('git', ['diff', 'HEAD', '--binary'], { cwd: REPO }));
  for (const relativePath of git(['ls-files', '--others', '--exclude-standard'])
    .split('\n').filter(Boolean).sort()) {
    hash.update(relativePath);
    const file = path.join(REPO, relativePath);
    if (fs.statSync(file).isFile()) hash.update(fs.readFileSync(file));
  }
  return {
    revision: git(['rev-parse', 'HEAD']),
    dirtyPaths,
    dirtyDiffSha256: dirtyPaths.length ? hash.digest('hex') : null,
  };
}

function writeReport(report) {
  fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  fs.writeFileSync(REPORT_PATH, `${JSON.stringify(report, null, 2)}\n`);
}

function failureReport(error, phase, context) {
  return {
    schema: 'sf3d.packed-fp16-linear-browser-smoke.v0',
    ok: false,
    failure: { phase, message: error instanceof Error ? error.message : String(error) },
    ...context,
    lastTrustworthyEvidence,
    terminal: { phase, primaryOutputWritten: false },
  };
}

const allocatePort = () => new Promise((resolve, reject) => {
  const server = net.createServer();
  server.unref();
  server.on('error', reject);
  server.listen(0, '127.0.0.1', () => {
    const { port } = server.address();
    server.close(() => resolve(port));
  });
});

async function startVite() {
  const port = await allocatePort();
  const vite = spawn('npx', ['vite', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], {
    cwd: REPO,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  children.push(vite);
  await new Promise((resolve, reject) => {
    let stderr = '';
    vite.stderr.on('data', chunk => { stderr += chunk; });
    vite.stdout.on('data', chunk => {
      if (String(chunk).includes('Local:')) resolve();
    });
    vite.on('error', reject);
    vite.on('exit', code => {
      if (code != null && code !== 0) reject(new Error(`Vite exited ${code}: ${stderr}`));
    });
  });
  return port;
}

let context = {
  source: null,
  kit: {
    packageVersion: null,
    producerRevision: KIT_PRODUCER_REVISION,
    tarballPath: KIT_TARBALL,
    tarballSha256: null,
  },
  requested: { rows, inDim, outDim, representation: 'f16-packed-u32' },
};

try {
  if (!KIT_PRODUCER_REVISION) throw new Error('--kit-producer-revision is required');
  if (!KIT_TARBALL || !fs.statSync(KIT_TARBALL, { throwIfNoEntry: false })?.isFile()) {
    throw new Error('--kit-tarball must name the installed producer artifact');
  }
  const source = sourceIdentity();
  const installedPackage = JSON.parse(fs.readFileSync(
    path.join(REPO, 'node_modules/@kaminos/webgpu-inference-kit/package.json'),
    'utf8',
  ));
  context = {
    source,
    kit: {
      packageVersion: installedPackage.version,
      producerRevision: KIT_PRODUCER_REVISION,
      tarballPath: KIT_TARBALL,
      tarballSha256: sha256File(KIT_TARBALL),
    },
    requested: { rows, inDim, outDim, representation: 'f16-packed-u32' },
  };
  if (![rows, inDim, outDim].every(value => Number.isSafeInteger(value) && value > 0)) {
    throw new Error('rows, inDim, and outDim must be positive integers');
  }
  if (!fs.existsSync(CHROME)) throw new Error(`Chrome not found at ${CHROME}`);
  lastTrustworthyEvidence = { phase: 'source-and-package-identity', ...context };

  const controlShader = fs.readFileSync(path.join(REPO, 'src/shaders/linear.wgsl'), 'utf8');
  const candidateShader = fs.readFileSync(path.join(REPO, 'src/shaders/linear_f16_packed.wgsl'), 'utf8');
  const port = await startVite();
  lastTrustworthyEvidence = { phase: 'vite-live', port };

  browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: false,
    args: [
      '--enable-unsafe-webgpu',
      '--disable-dawn-features=disallow_unsafe_apis',
      '--use-angle=metal',
      '--no-first-run',
      '--no-default-browser-check',
    ],
  });
  const browserVersion = await browser.version();
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${port}/tools/packed_fp16_linear_smoke.html`, { waitUntil: 'load' });
  lastTrustworthyEvidence = { phase: 'browser-page-live', url: page.url() };

  const browserResult = await page.evaluate(async ({ controlShader, candidateShader, rows, inDim, outDim }) => {
    if (!navigator.gpu) throw new Error('navigator.gpu is unavailable');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no WebGPU adapter');
    const device = await adapter.requestDevice();
    const assay = await import('/src/lib/packed_fp16_linear_assay.js');
    const fixture = assay.createPackedFp16LinearFixture({ rows, inDim, outDim });
    const owned = [];

    const createBuffer = (data, usage, label) => {
      const size = Math.ceil(data.byteLength / 4) * 4;
      const buffer = device.createBuffer({ size, usage, mappedAtCreation: true, label });
      new data.constructor(buffer.getMappedRange()).set(data);
      buffer.unmap();
      owned.push(buffer);
      return buffer;
    };
    const createOutput = (size, label) => {
      const buffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        label,
      });
      owned.push(buffer);
      return buffer;
    };
    const readOutput = async (source, size) => {
      const staging = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: 'packed-linear-readback',
      });
      owned.push(staging);
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(source, 0, staging, 0, size);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const output = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return output;
    };
    const compile = async (code, label) => {
      const module = device.createShaderModule({ code, label });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter(message => message.type === 'error');
      if (errors.length) throw new Error(`${label} compilation failed: ${errors.map(error => error.message).join('; ')}`);
      return device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
    };
    const dispatch = async ({ pipeline, weight, output, label }) => {
      const workgroupsX = Math.ceil((rows * outDim) / 256);
      const params = createBuffer(
        new Uint32Array([rows, inDim, outDim, workgroupsX, 0]),
        GPUBufferUsage.UNIFORM,
        `${label}-params`,
      );
      const input = createBuffer(fixture.input, GPUBufferUsage.STORAGE, `${label}-input`);
      const bias = createBuffer(fixture.bias, GPUBufferUsage.STORAGE, `${label}-bias`);
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: input } },
          { binding: 2, resource: { buffer: weight } },
          { binding: 3, resource: { buffer: bias } },
          { binding: 4, resource: { buffer: output } },
        ],
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupsX);
      pass.end();
      const startedAt = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - startedAt;
    };

    try {
      const [controlPipeline, candidatePipeline] = await Promise.all([
        compile(controlShader, 'expanded-f32-linear'),
        compile(candidateShader, 'packed-fp16-linear'),
      ]);
      const outputBytes = rows * outDim * 4;
      const controlWeight = createBuffer(fixture.expandedWeights, GPUBufferUsage.STORAGE, 'expanded-f32-weight');
      const candidateWeight = createBuffer(fixture.packedWeights, GPUBufferUsage.STORAGE, 'packed-fp16-weight');
      const controlOutputBuffer = createOutput(outputBytes, 'expanded-f32-output');
      const candidateOutputBuffer = createOutput(outputBytes, 'packed-fp16-output');
      const controlDurationMs = await dispatch({
        pipeline: controlPipeline,
        weight: controlWeight,
        output: controlOutputBuffer,
        label: 'expanded-f32-linear',
      });
      const candidateDurationMs = await dispatch({
        pipeline: candidatePipeline,
        weight: candidateWeight,
        output: candidateOutputBuffer,
        label: 'packed-fp16-linear',
      });
      const controlOutput = await readOutput(controlOutputBuffer, outputBytes);
      const candidateOutput = await readOutput(candidateOutputBuffer, outputBytes);
      let maxAbsDiff = 0;
      let exact = controlOutput.length === candidateOutput.length;
      for (let index = 0; index < controlOutput.length && exact; index++) {
        const difference = Math.abs(controlOutput[index] - candidateOutput[index]);
        maxAbsDiff = Math.max(maxAbsDiff, difference);
        if (!Object.is(controlOutput[index], candidateOutput[index])) exact = false;
      }
      return {
        browserUserAgent: navigator.userAgent,
        adapter: {
          vendor: adapter.info?.vendor || null,
          architecture: adapter.info?.architecture || null,
          device: adapter.info?.device || null,
          description: adapter.info?.description || null,
          features: [...adapter.features].sort(),
        },
        effectiveBackend: 'webgpu',
        effectiveRepresentation: fixture.plan.effectiveRepresentation,
        control: {
          storageByteLength: controlWeight.size,
          output: [...controlOutput],
          outputFinite: [...controlOutput].every(Number.isFinite),
          durationMs: controlDurationMs,
        },
        candidate: {
          storageByteLength: candidateWeight.size,
          output: [...candidateOutput],
          outputFinite: [...candidateOutput].every(Number.isFinite),
          durationMs: candidateDurationMs,
        },
        comparison: { exact, maxAbsDiff },
      };
    } finally {
      for (const buffer of owned) buffer.destroy();
      device.destroy();
    }
  }, { controlShader, candidateShader, rows, inDim, outDim });

  const report = {
    schema: 'sf3d.packed-fp16-linear-browser-smoke.v0',
    ok: true,
    ...context,
    ...browserResult,
    browser: {
      version: browserVersion,
      userAgent: browserResult.browserUserAgent,
    },
    terminal: { phase: 'complete', primaryOutputWritten: true },
  };
  delete report.browserUserAgent;
  for (const arm of ['control', 'candidate']) {
    const values = new Float32Array(report[arm].output);
    report[arm].outputSha256 = sha256(Buffer.from(values.buffer));
  }
  const admission = evaluatePackedFp16LinearSmokeReport(report);
  report.admission = admission;
  if (!admission.ok) report.ok = false;
  writeReport(report);
  if (!admission.ok) throw new Error(`browser smoke admission failed: ${admission.errors.join('; ')}`);
  console.log(JSON.stringify({
    schema: report.schema,
    ok: report.ok,
    source: report.source,
    kit: report.kit,
    requested: report.requested,
    browser: report.browser,
    adapter: report.adapter,
    effectiveBackend: report.effectiveBackend,
    effectiveRepresentation: report.effectiveRepresentation,
    control: {
      storageByteLength: report.control.storageByteLength,
      durationMs: report.control.durationMs,
      outputCount: report.control.output.length,
      outputSha256: report.control.outputSha256,
    },
    candidate: {
      storageByteLength: report.candidate.storageByteLength,
      durationMs: report.candidate.durationMs,
      outputCount: report.candidate.output.length,
      outputSha256: report.candidate.outputSha256,
    },
    comparison: report.comparison,
    admission: report.admission,
  }, null, 2));
} catch (error) {
  if (!fs.existsSync(REPORT_PATH)) writeReport(failureReport(error, lastTrustworthyEvidence.phase, context));
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
} finally {
  if (browser) await browser.close().catch(() => {});
  for (const child of children) {
    if (child.exitCode == null) child.kill('SIGTERM');
  }
}
