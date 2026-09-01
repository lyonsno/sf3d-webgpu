#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import puppeteer from 'puppeteer-core';

import { verifyPackageArtifactIdentity } from '../src/lib/package_artifact_identity.js';
import { evaluatePackedFp16LinearSmokeReport } from '../src/lib/packed_fp16_linear_report.js';

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
const KIT_PRODUCER_REPO_ARG = argValue('--kit-producer-repo');
const KIT_PRODUCER_REPO = KIT_PRODUCER_REPO_ARG ? path.resolve(KIT_PRODUCER_REPO_ARG) : null;
const KIT_PRODUCER_REMOTE = argValue('--kit-producer-remote');
const EXPECTED_SOURCE_REVISION = argValue('--expected-source-revision');
const EXPECTED_KIT_VERSION = argValue('--expected-kit-version');
const EXPECTED_KIT_TARBALL_SHA256 = argValue('--expected-kit-tarball-sha256');
const EXPECTED_KIT_PRODUCER_TREE = argValue('--expected-kit-producer-tree');
const EXPECTED_KIT_MANIFEST_SHA256 = argValue('--expected-kit-manifest-sha256');
const INSTALLED_PACKAGE_PATH = path.join(REPO, 'node_modules/@kaminos/webgpu-inference-kit');
const rows = Number(argValue('--rows', '3'));
const inDim = Number(argValue('--in-dim', '5'));
const outDim = Number(argValue('--out-dim', '7'));
const children = [];
let browser = null;
let lastTrustworthyEvidence = { phase: 'argument-parse' };

const sha256 = data => crypto.createHash('sha256').update(data).digest('hex');
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
  const temporaryPath = `${REPORT_PATH}.${process.pid}.${crypto.randomUUID()}.tmp`;
  const fd = fs.openSync(temporaryPath, 'wx');
  try {
    fs.writeFileSync(fd, `${JSON.stringify(report, null, 2)}\n`);
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  try {
    fs.renameSync(temporaryPath, REPORT_PATH);
  } catch (error) {
    fs.rmSync(temporaryPath, { force: true });
    throw error;
  }
}

function failureReport(error, phase, context) {
  return {
    schema: 'sf3d.packed-fp16-linear-browser-smoke.v1',
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
  requestedIdentity: {
    sourceRevision: EXPECTED_SOURCE_REVISION,
    kitPackageVersion: EXPECTED_KIT_VERSION,
    kitProducerRevision: KIT_PRODUCER_REVISION,
    kitProducerRemote: KIT_PRODUCER_REMOTE,
    kitProducerTree: EXPECTED_KIT_PRODUCER_TREE,
    kitTarballSha256: EXPECTED_KIT_TARBALL_SHA256,
    kitManifestSha256: EXPECTED_KIT_MANIFEST_SHA256,
  },
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
  if (!EXPECTED_SOURCE_REVISION) throw new Error('--expected-source-revision is required');
  if (!EXPECTED_KIT_VERSION) throw new Error('--expected-kit-version is required');
  if (!KIT_PRODUCER_REVISION) throw new Error('--kit-producer-revision is required');
  if (!KIT_PRODUCER_REPO
      || !fs.statSync(KIT_PRODUCER_REPO, { throwIfNoEntry: false })?.isDirectory()) {
    throw new Error('--kit-producer-repo must name the producer repository');
  }
  if (!KIT_PRODUCER_REMOTE) throw new Error('--kit-producer-remote is required');
  if (!KIT_TARBALL || !fs.statSync(KIT_TARBALL, { throwIfNoEntry: false })?.isFile()) {
    throw new Error('--kit-tarball must name the installed producer artifact');
  }
  if (!EXPECTED_KIT_TARBALL_SHA256) throw new Error('--expected-kit-tarball-sha256 is required');
  if (!EXPECTED_KIT_PRODUCER_TREE) throw new Error('--expected-kit-producer-tree is required');
  if (!EXPECTED_KIT_MANIFEST_SHA256) throw new Error('--expected-kit-manifest-sha256 is required');
  const source = sourceIdentity();
  if (source.revision !== EXPECTED_SOURCE_REVISION) {
    throw new Error(`source revision ${source.revision} does not match expected ${EXPECTED_SOURCE_REVISION}`);
  }
  if (source.dirtyPaths.length !== 0) {
    throw new Error(`source must be clean; dirty paths: ${source.dirtyPaths.join(', ')}`);
  }
  const kitIdentity = verifyPackageArtifactIdentity({
    producerRepo: KIT_PRODUCER_REPO,
    producerRevision: KIT_PRODUCER_REVISION,
    producerRemote: KIT_PRODUCER_REMOTE,
    packageSubdir: 'webgpu-inference-kit',
    packageVersion: EXPECTED_KIT_VERSION,
    tarballPath: KIT_TARBALL,
    tarballSha256: EXPECTED_KIT_TARBALL_SHA256,
    installedPackagePath: INSTALLED_PACKAGE_PATH,
  });
  if (kitIdentity.producerTree !== EXPECTED_KIT_PRODUCER_TREE) {
    throw new Error(`kit producer tree ${kitIdentity.producerTree} does not match expected ${EXPECTED_KIT_PRODUCER_TREE}`);
  }
  if (kitIdentity.tarballManifestSha256 !== EXPECTED_KIT_MANIFEST_SHA256) {
    throw new Error(
      `kit package manifest ${kitIdentity.tarballManifestSha256} does not match expected ${EXPECTED_KIT_MANIFEST_SHA256}`,
    );
  }
  context = {
    requestedIdentity: context.requestedIdentity,
    source,
    kit: {
      ...kitIdentity,
      tarballPath: KIT_TARBALL,
      installedPackagePath: fs.realpathSync(INSTALLED_PACKAGE_PATH),
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
    const dispatch = async ({ pipeline, weight, output, label, fixture: activeFixture, transposed }) => {
      const workgroupsX = Math.ceil((activeFixture.rows * activeFixture.outDim) / 256);
      const params = createBuffer(
        new Uint32Array([
          activeFixture.rows,
          activeFixture.inDim,
          activeFixture.outDim,
          workgroupsX,
          transposed ? 1 : 0,
        ]),
        GPUBufferUsage.UNIFORM,
        `${label}-params`,
      );
      const input = createBuffer(activeFixture.input, GPUBufferUsage.STORAGE, `${label}-input`);
      const bias = createBuffer(activeFixture.bias, GPUBufferUsage.STORAGE, `${label}-bias`);
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
      const compare = (left, right) => {
        let exact = left.length === right.length;
        let maxAbsDiff = 0;
        for (let index = 0; index < left.length; index++) {
          const difference = Math.abs(left[index] - right[index]);
          maxAbsDiff = Math.max(maxAbsDiff, difference);
          if (!Object.is(left[index], right[index])) exact = false;
        }
        return { exact, maxAbsDiff };
      };
      const runPair = async ({ activeFixture, transposed, label }) => {
        const outputBytes = activeFixture.rows * activeFixture.outDim * 4;
        const controlWeight = createBuffer(
          activeFixture.expandedWeights,
          GPUBufferUsage.STORAGE,
          `${label}-expanded-f32-weight`,
        );
        const candidateWeight = createBuffer(
          activeFixture.packedWeights,
          GPUBufferUsage.STORAGE,
          `${label}-packed-fp16-weight`,
        );
        const controlOutputBuffer = createOutput(outputBytes, `${label}-expanded-f32-output`);
        const candidateOutputBuffer = createOutput(outputBytes, `${label}-packed-fp16-output`);
        const controlDurationMs = await dispatch({
          pipeline: controlPipeline,
          weight: controlWeight,
          output: controlOutputBuffer,
          label: `${label}-expanded-f32-linear`,
          fixture: activeFixture,
          transposed,
        });
        const candidateDurationMs = await dispatch({
          pipeline: candidatePipeline,
          weight: candidateWeight,
          output: candidateOutputBuffer,
          label: `${label}-packed-fp16-linear`,
          fixture: activeFixture,
          transposed,
        });
        const controlOutput = await readOutput(controlOutputBuffer, outputBytes);
        const candidateOutput = await readOutput(candidateOutputBuffer, outputBytes);
        return {
          transposed,
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
          comparison: compare(controlOutput, candidateOutput),
        };
      };

      const mainPair = await runPair({ activeFixture: fixture, transposed: false, label: 'real-shape-native' });
      const probeNativeFixture = assay.createPackedFp16LinearFixture({ rows: 3, inDim: 5, outDim: 7 });
      const probeTransposedFixture = assay.createTransposedPackedFp16LinearFixture(probeNativeFixture);
      const referenceOutput = assay.runLinearReference({
        input: probeNativeFixture.input,
        weights: probeNativeFixture.expandedWeights,
        bias: probeNativeFixture.bias,
        rows: probeNativeFixture.rows,
        inDim: probeNativeFixture.inDim,
        outDim: probeNativeFixture.outDim,
      });
      const nativeProbe = await runPair({
        activeFixture: probeNativeFixture,
        transposed: false,
        label: 'probe-native',
      });
      const transposedProbe = await runPair({
        activeFixture: probeTransposedFixture,
        transposed: true,
        label: 'probe-transposed',
      });
      nativeProbe.referenceComparison = compare(referenceOutput, nativeProbe.control.output);
      transposedProbe.referenceComparison = compare(referenceOutput, transposedProbe.control.output);
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
        control: mainPair.control,
        candidate: mainPair.candidate,
        comparison: mainPair.comparison,
        layoutProbe: {
          shape: {
            rows: probeNativeFixture.rows,
            inDim: probeNativeFixture.inDim,
            outDim: probeNativeFixture.outDim,
          },
          reference: {
            output: [...referenceOutput],
            outputFinite: [...referenceOutput].every(Number.isFinite),
          },
          native: nativeProbe,
          transposed: transposedProbe,
          crossLayout: {
            control: compare(nativeProbe.control.output, transposedProbe.control.output),
            candidate: compare(nativeProbe.candidate.output, transposedProbe.candidate.output),
          },
        },
      };
    } finally {
      for (const buffer of owned) buffer.destroy();
      device.destroy();
    }
  }, { controlShader, candidateShader, rows, inDim, outDim });

  const report = {
    schema: 'sf3d.packed-fp16-linear-browser-smoke.v1',
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
  const attachRawOutput = arm => {
    const values = new Float32Array(arm.output);
    const outputBytes = Buffer.from(values.buffer);
    arm.outputF32Base64 = outputBytes.toString('base64');
    arm.outputSha256 = sha256(outputBytes);
  };
  attachRawOutput(report.control);
  attachRawOutput(report.candidate);
  attachRawOutput(report.layoutProbe.reference);
  for (const layout of ['native', 'transposed']) {
    attachRawOutput(report.layoutProbe[layout].control);
    attachRawOutput(report.layoutProbe[layout].candidate);
  }
  const admission = evaluatePackedFp16LinearSmokeReport(report, context.requestedIdentity);
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
    layoutProbe: report.layoutProbe,
    admission: report.admission,
  }, null, 2));
} catch (error) {
  writeReport(failureReport(error, lastTrustworthyEvidence.phase, context));
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
} finally {
  if (browser) await browser.close().catch(() => {});
  for (const child of children) {
    if (child.exitCode == null) child.kill('SIGTERM');
  }
}
