import {
  createWebGpuWeightRepresentationPlan,
  packFp16WeightsToU32,
} from '@kaminos/webgpu-inference-kit';

function fp16BitsToFp32(bits) {
  const sign = (bits >> 15) & 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) {
    if (mantissa === 0) return sign ? -0 : 0;
    const value = (mantissa / 1024) * (2 ** -14);
    return sign ? -value : value;
  }
  if (exponent === 0x1f) {
    if (mantissa !== 0) return NaN;
    return sign ? -Infinity : Infinity;
  }
  const value = (1 + mantissa / 1024) * (2 ** (exponent - 15));
  return sign ? -value : value;
}

function positiveDimensions({ rows, inDim, outDim }) {
  if (![rows, inDim, outDim].every(value => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError('rows, inDim, and outDim must be positive integers');
  }
}

export function assertPackedFp16LinearShaderContract(shader) {
  if (typeof shader !== 'string') throw new TypeError('shader must be a string');
  const requirements = [
    [/var<storage, read> weight: array<u32>;/, 'packed u32 weight storage'],
    [/fn loadWeight\(index: u32\) -> f32/, 'f32 weight load'],
    [/unpack2x16float/, 'unpack2x16float'],
  ];
  for (const [pattern, name] of requirements) {
    if (!pattern.test(shader)) throw new Error(`packed linear shader is missing ${name}`);
  }
  if (/var<storage, read> weight: array<f32>;/.test(shader)) {
    throw new Error('packed linear shader must not bind expanded f32 weights');
  }
  return true;
}

export function createPackedFp16LinearFixture({ rows, inDim, outDim }) {
  positiveDimensions({ rows, inDim, outDim });
  const weightCount = inDim * outDim;
  const finitePatterns = new Uint16Array([
    0x0000, 0x3000, 0xb400, 0x3800, 0xba00, 0x3c00, 0xbc00,
    0x3d00, 0xbd00, 0x3555, 0xb555, 0x3aab, 0xbaab, 0x0400,
  ]);
  const fp16Bits = new Uint16Array(weightCount);
  for (let index = 0; index < fp16Bits.length; index++) {
    fp16Bits[index] = finitePatterns[(index * 5 + 3) % finitePatterns.length];
  }

  const expandedWeights = new Float32Array(weightCount);
  for (let index = 0; index < weightCount; index++) {
    expandedWeights[index] = fp16BitsToFp32(fp16Bits[index]);
  }
  const unpackedWeights = new Float32Array(expandedWeights);
  const packedWeights = packFp16WeightsToU32(fp16Bits);
  const input = new Float32Array(rows * inDim);
  for (let index = 0; index < input.length; index++) {
    input[index] = Math.fround((((index * 17) % 29) - 14) / 11);
  }
  const bias = new Float32Array(outDim);
  for (let index = 0; index < bias.length; index++) {
    bias[index] = Math.fround((index - 3) / 23);
  }
  const plan = createWebGpuWeightRepresentationPlan({
    sourceDtype: 'fp16',
    elementCount: weightCount,
    candidates: ['f16-packed-u32'],
    adapterFeatures: [],
  });

  return Object.freeze({
    rows,
    inDim,
    outDim,
    fp16Bits,
    expandedWeights,
    unpackedWeights,
    packedWeights,
    input,
    bias,
    plan,
  });
}

export function runLinearReference({ input, weights, bias, rows, inDim, outDim }) {
  positiveDimensions({ rows, inDim, outDim });
  if (!(input instanceof Float32Array) || input.length !== rows * inDim) {
    throw new RangeError('input must be a Float32Array with rows * inDim elements');
  }
  if (!(weights instanceof Float32Array) || weights.length !== outDim * inDim) {
    throw new RangeError('weights must be a Float32Array with outDim * inDim elements');
  }
  if (!(bias instanceof Float32Array) || bias.length !== outDim) {
    throw new RangeError('bias must be a Float32Array with outDim elements');
  }

  const output = new Float32Array(rows * outDim);
  const len4 = Math.floor(inDim / 4) * 4;
  for (let row = 0; row < rows; row++) {
    const inputBase = row * inDim;
    for (let col = 0; col < outDim; col++) {
      const weightBase = col * inDim;
      let s0 = 0;
      let s1 = 0;
      let s2 = 0;
      let s3 = 0;
      for (let k = 0; k < len4; k += 4) {
        s0 = Math.fround(s0 + Math.fround(input[inputBase + k] * weights[weightBase + k]));
        s1 = Math.fround(s1 + Math.fround(input[inputBase + k + 1] * weights[weightBase + k + 1]));
        s2 = Math.fround(s2 + Math.fround(input[inputBase + k + 2] * weights[weightBase + k + 2]));
        s3 = Math.fround(s3 + Math.fround(input[inputBase + k + 3] * weights[weightBase + k + 3]));
      }
      for (let k = len4; k < inDim; k++) {
        s0 = Math.fround(s0 + Math.fround(input[inputBase + k] * weights[weightBase + k]));
      }
      output[row * outDim + col] = Math.fround(
        Math.fround(Math.fround(s0 + s1) + Math.fround(s2 + s3)) + bias[col],
      );
    }
  }
  return output;
}

export function evaluatePackedFp16LinearSmokeReport(report) {
  const errors = [];
  if (!report || typeof report !== 'object' || Array.isArray(report)) {
    return { ok: false, errors: ['report must be an object'] };
  }
  if (report.schema !== 'sf3d.packed-fp16-linear-browser-smoke.v0') {
    errors.push('unexpected report schema');
  }
  if (report.ok !== true) errors.push('report must declare terminal success');
  if (report.effectiveBackend !== 'webgpu') errors.push('effectiveBackend must be webgpu');
  if (report.effectiveRepresentation !== 'f16-packed-u32') {
    errors.push('effectiveRepresentation must be f16-packed-u32');
  }
  if (!report.source?.revision) errors.push('source revision is required');
  if (!report.kit?.packageVersion || !report.kit?.producerRevision || !report.kit?.tarballSha256) {
    errors.push('kit package, producer revision, and tarball identity are required');
  }
  if (!report.adapter || typeof report.adapter !== 'object') errors.push('adapter identity is required');

  for (const arm of ['control', 'candidate']) {
    const value = report[arm];
    if (!value || typeof value !== 'object') {
      errors.push(`${arm} arm is required`);
      continue;
    }
    if (!Number.isSafeInteger(value.storageByteLength) || value.storageByteLength <= 0) {
      errors.push(`${arm} storageByteLength must be a positive integer`);
    }
    if (!Array.isArray(value.output) || value.output.length === 0) {
      errors.push(`${arm} output must be a non-empty array`);
    }
    if (value.outputFinite !== true) errors.push(`${arm} output must be finite`);
    if (typeof value.outputSha256 !== 'string' || value.outputSha256.length !== 64) {
      errors.push(`${arm} outputSha256 must be a SHA-256 hex digest`);
    }
  }
  if (Number.isSafeInteger(report.control?.storageByteLength)
      && Number.isSafeInteger(report.candidate?.storageByteLength)
      && report.candidate.storageByteLength >= report.control.storageByteLength) {
    errors.push('candidate storage must be smaller than control storage');
  }
  if (report.comparison?.exact !== true || report.comparison?.maxAbsDiff !== 0) {
    errors.push('exact output parity with zero maxAbsDiff is required');
  }
  if (report.control?.outputSha256 !== report.candidate?.outputSha256) {
    errors.push('control and candidate output hashes must match');
  }
  if (report.terminal?.phase !== 'complete' || report.terminal?.primaryOutputWritten !== true) {
    errors.push('terminal report must preserve complete primary output');
  }
  return { ok: errors.length === 0, errors };
}
