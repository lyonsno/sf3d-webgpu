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
    [/weight\[index >> 1u\]/, 'packed word index'],
    [/select\(pair\.x, pair\.y, \(index & 1u\) == 1u\)/, 'packed lane selection'],
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

export function createTransposedPackedFp16LinearFixture(fixture) {
  const { rows, inDim, outDim, fp16Bits, expandedWeights, input, bias, plan } = fixture;
  positiveDimensions({ rows, inDim, outDim });
  if (!(fp16Bits instanceof Uint16Array) || fp16Bits.length !== inDim * outDim) {
    throw new RangeError('fixture fp16Bits must contain inDim * outDim elements');
  }
  if (!(expandedWeights instanceof Float32Array) || expandedWeights.length !== inDim * outDim) {
    throw new RangeError('fixture expandedWeights must contain inDim * outDim elements');
  }
  const transposedBits = new Uint16Array(fp16Bits.length);
  const transposedWeights = new Float32Array(expandedWeights.length);
  for (let col = 0; col < outDim; col++) {
    for (let k = 0; k < inDim; k++) {
      const nativeIndex = col * inDim + k;
      const transposedIndex = k * outDim + col;
      transposedBits[transposedIndex] = fp16Bits[nativeIndex];
      transposedWeights[transposedIndex] = expandedWeights[nativeIndex];
    }
  }
  return Object.freeze({
    rows,
    inDim,
    outDim,
    fp16Bits: transposedBits,
    expandedWeights: transposedWeights,
    unpackedWeights: new Float32Array(transposedWeights),
    packedWeights: packFp16WeightsToU32(transposedBits),
    input,
    bias,
    plan,
  });
}

export function runLinearReference({
  input,
  weights,
  bias,
  rows,
  inDim,
  outDim,
  transposed = false,
}) {
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
      let s0 = 0;
      let s1 = 0;
      let s2 = 0;
      let s3 = 0;
      const weightIndex = k => (transposed ? k * outDim + col : col * inDim + k);
      for (let k = 0; k < len4; k += 4) {
        s0 = Math.fround(s0 + Math.fround(input[inputBase + k] * weights[weightIndex(k)]));
        s1 = Math.fround(s1 + Math.fround(input[inputBase + k + 1] * weights[weightIndex(k + 1)]));
        s2 = Math.fround(s2 + Math.fround(input[inputBase + k + 2] * weights[weightIndex(k + 2)]));
        s3 = Math.fround(s3 + Math.fround(input[inputBase + k + 3] * weights[weightIndex(k + 3)]));
      }
      for (let k = len4; k < inDim; k++) {
        s0 = Math.fround(s0 + Math.fround(input[inputBase + k] * weights[weightIndex(k)]));
      }
      output[row * outDim + col] = Math.fround(
        Math.fround(Math.fround(s0 + s1) + Math.fround(s2 + s3)) + bias[col],
      );
    }
  }
  return output;
}
