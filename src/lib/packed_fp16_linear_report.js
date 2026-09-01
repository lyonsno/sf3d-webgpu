import crypto from 'node:crypto';

const SHA256_HEX = /^[0-9a-f]{64}$/;
const GIT_SHA_HEX = /^[0-9a-f]{40}$/;

function safeProduct(values, name, errors) {
  let product = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      errors.push(`${name} values must be positive safe integers`);
      return null;
    }
    product *= value;
    if (!Number.isSafeInteger(product)) {
      errors.push(`${name} product must be a safe integer`);
      return null;
    }
  }
  return product;
}

function decodeOutput(armName, arm, expectedCount, errors) {
  if (!arm || typeof arm !== 'object' || Array.isArray(arm)) {
    errors.push(`${armName} arm is required`);
    return null;
  }
  if (!Array.isArray(arm.output) || arm.output.length !== expectedCount) {
    errors.push(`${armName} output length must equal rows * outDim`);
    return null;
  }
  if (typeof arm.outputF32Base64 !== 'string' || arm.outputF32Base64.length === 0) {
    errors.push(`${armName} outputF32Base64 is required`);
    return null;
  }

  let bytes;
  try {
    bytes = Buffer.from(arm.outputF32Base64, 'base64');
  } catch {
    errors.push(`${armName} outputF32Base64 must be valid base64`);
    return null;
  }
  if (bytes.byteLength !== expectedCount * Float32Array.BYTES_PER_ELEMENT) {
    errors.push(`${armName} raw output byte length must equal rows * outDim * 4`);
    return null;
  }
  const aligned = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const decoded = new Float32Array(aligned);
  for (let index = 0; index < expectedCount; index++) {
    if (!Number.isFinite(decoded[index])) {
      errors.push(`${armName} raw output must contain only finite values`);
      break;
    }
    if (!Number.isFinite(arm.output[index]) || !Object.is(decoded[index], arm.output[index])) {
      errors.push(`${armName} output array must match its raw Float32 bytes`);
      break;
    }
  }
  if (arm.outputFinite !== true) errors.push(`${armName} outputFinite must match finite raw output`);
  const digest = crypto.createHash('sha256').update(bytes).digest('hex');
  if (!SHA256_HEX.test(arm.outputSha256 || '') || arm.outputSha256 !== digest) {
    errors.push(`${armName} outputSha256 must match raw Float32 bytes`);
  }
  return { decoded, digest };
}

function meaningfulAdapter(adapter) {
  if (!adapter || typeof adapter !== 'object' || Array.isArray(adapter)) return false;
  return ['vendor', 'architecture', 'device', 'description']
    .some(field => typeof adapter[field] === 'string' && adapter[field].trim().length > 0);
}

function compareDecoded(label, left, right, declaration, errors, { requireExact = true } = {}) {
  if (!left || !right) return;
  let exact = left.decoded.length === right.decoded.length;
  let maxAbsDiff = 0;
  for (let index = 0; index < left.decoded.length; index++) {
    const difference = Math.abs(left.decoded[index] - right.decoded[index]);
    maxAbsDiff = Math.max(maxAbsDiff, difference);
    if (!Object.is(left.decoded[index], right.decoded[index])) exact = false;
  }
  if (requireExact && (!exact || maxAbsDiff !== 0)) {
    errors.push(`${label} raw outputs must be exactly equal`);
  }
  if (declaration?.exact !== exact || declaration?.maxAbsDiff !== maxAbsDiff) {
    errors.push(`${label} comparison declarations must match recomputed raw output parity`);
  }
  if (requireExact && left.digest !== right.digest) {
    errors.push(`${label} recomputed output hashes must match`);
  }
}

function evaluateLayoutProbe(layoutProbe, errors) {
  if (!layoutProbe || typeof layoutProbe !== 'object' || Array.isArray(layoutProbe)) {
    errors.push('live native/transposed layoutProbe is required');
    return;
  }
  const shape = layoutProbe.shape;
  const outputCount = safeProduct([shape?.rows, shape?.outDim], 'layoutProbe output shape', errors);
  const weightCount = safeProduct([shape?.inDim, shape?.outDim], 'layoutProbe weight shape', errors);
  if (outputCount == null || weightCount == null) return;
  const controlBytes = weightCount * Float32Array.BYTES_PER_ELEMENT;
  const candidateBytes = Math.ceil(weightCount / 2) * Uint32Array.BYTES_PER_ELEMENT;
  const reference = decodeOutput('layoutProbe reference', layoutProbe.reference, outputCount, errors);
  const decodedLayouts = {};
  for (const [name, transposed] of [['native', false], ['transposed', true]]) {
    const layout = layoutProbe[name];
    if (!layout || layout.transposed !== transposed) {
      errors.push(`layoutProbe ${name} must declare transposed=${transposed}`);
      continue;
    }
    if (layout.control?.storageByteLength !== controlBytes) {
      errors.push(`layoutProbe ${name} control allocation must match shape`);
    }
    if (layout.candidate?.storageByteLength !== candidateBytes) {
      errors.push(`layoutProbe ${name} candidate allocation must match shape`);
    }
    const control = decodeOutput(`layoutProbe ${name} control`, layout.control, outputCount, errors);
    const candidate = decodeOutput(`layoutProbe ${name} candidate`, layout.candidate, outputCount, errors);
    compareDecoded(`layoutProbe ${name}`, control, candidate, layout.comparison, errors);
    compareDecoded(
      `layoutProbe ${name} reference`,
      reference,
      control,
      layout.referenceComparison,
      errors,
      { requireExact: false },
    );
    decodedLayouts[name] = { control, candidate };
  }
  compareDecoded(
    'layoutProbe native/transposed control',
    decodedLayouts.native?.control,
    decodedLayouts.transposed?.control,
    layoutProbe.crossLayout?.control,
    errors,
  );
  compareDecoded(
    'layoutProbe native/transposed candidate',
    decodedLayouts.native?.candidate,
    decodedLayouts.transposed?.candidate,
    layoutProbe.crossLayout?.candidate,
    errors,
  );
}

export function evaluatePackedFp16LinearSmokeReport(report, expectedIdentity = {}) {
  const errors = [];
  if (!report || typeof report !== 'object' || Array.isArray(report)) {
    return { ok: false, errors: ['report must be an object'] };
  }
  if (report.schema !== 'sf3d.packed-fp16-linear-browser-smoke.v1') {
    errors.push('unexpected report schema');
  }
  if (report.ok !== true) errors.push('report must declare terminal success');
  if (report.effectiveBackend !== 'webgpu') errors.push('effectiveBackend must be webgpu');
  if (report.effectiveRepresentation !== 'f16-packed-u32') {
    errors.push('effectiveRepresentation must be f16-packed-u32');
  }

  if (!GIT_SHA_HEX.test(expectedIdentity.sourceRevision || '')) {
    errors.push('expected source revision must be an exact Git SHA');
  }
  if (!GIT_SHA_HEX.test(expectedIdentity.kitProducerTree || '')) {
    errors.push('expected kit producer tree must be an exact Git SHA');
  }
  if (!SHA256_HEX.test(expectedIdentity.kitManifestSha256 || '')) {
    errors.push('expected kit manifest must be a SHA-256 digest');
  }
  for (const field of [
    'sourceRevision',
    'kitPackageVersion',
    'kitProducerRevision',
    'kitProducerRemote',
    'kitProducerTree',
    'kitTarballSha256',
    'kitManifestSha256',
  ]) {
    if (report.requestedIdentity?.[field] !== expectedIdentity[field]) {
      errors.push(`requested identity ${field} must match the caller-owned expectation`);
    }
  }
  if (!GIT_SHA_HEX.test(report.source?.revision || '')
      || report.source?.revision !== expectedIdentity.sourceRevision) {
    errors.push('source revision must match the expected exact Git SHA');
  }
  if (!Array.isArray(report.source?.dirtyPaths) || report.source.dirtyPaths.length !== 0
      || report.source?.dirtyDiffSha256 != null) {
    errors.push('source must be clean');
  }

  const kit = report.kit;
  if (!kit || typeof kit !== 'object' || Array.isArray(kit)) {
    errors.push('kit identity is required');
  } else {
    if (kit.packageVersion !== expectedIdentity.kitPackageVersion) {
      errors.push('kit package version must match expected identity');
    }
    if (!GIT_SHA_HEX.test(kit.producerRevision || '')
        || kit.producerRevision !== expectedIdentity.kitProducerRevision) {
      errors.push('kit producer revision must match expected exact Git SHA');
    }
    if (!SHA256_HEX.test(kit.tarballSha256 || '')
        || kit.tarballSha256 !== expectedIdentity.kitTarballSha256) {
      errors.push('kit tarball SHA-256 must match expected identity');
    }
    if (typeof kit.producerRemote !== 'string' || kit.producerRemote.trim().length === 0
        || kit.producerRemote !== expectedIdentity.kitProducerRemote) {
      errors.push('kit producer remote must match expected identity');
    }
    if (!GIT_SHA_HEX.test(kit.producerTree || '')
        || kit.producerTree !== expectedIdentity.kitProducerTree) {
      errors.push('kit producer tree must match expected identity');
    }
    for (const field of ['tarballManifestSha256', 'installedManifestSha256', 'sourceManifestSha256']) {
      if (!SHA256_HEX.test(kit[field] || '') || kit[field] !== expectedIdentity.kitManifestSha256) {
        errors.push(`kit ${field} must match the expected package manifest`);
      }
    }
    if (kit.tarballManifestSha256 !== kit.installedManifestSha256
        || kit.tarballManifestSha256 !== kit.sourceManifestSha256) {
      errors.push('tarball, installed package, and producer source manifests must match');
    }
  }

  if (!report.browser?.version || !report.browser?.userAgent) {
    errors.push('browser identity requires version and userAgent');
  }
  if (!meaningfulAdapter(report.adapter)) errors.push('meaningful adapter identity is required');

  const requested = report.requested;
  if (requested?.representation !== 'f16-packed-u32') {
    errors.push('requested representation must be f16-packed-u32');
  }
  const outputCount = safeProduct([requested?.rows, requested?.outDim], 'output shape', errors);
  const weightCount = safeProduct([requested?.inDim, requested?.outDim], 'weight shape', errors);
  let expectedControlBytes = null;
  let expectedCandidateBytes = null;
  if (weightCount != null) {
    expectedControlBytes = weightCount * Float32Array.BYTES_PER_ELEMENT;
    expectedCandidateBytes = Math.ceil(weightCount / 2) * Uint32Array.BYTES_PER_ELEMENT;
    if (!Number.isSafeInteger(expectedControlBytes) || !Number.isSafeInteger(expectedCandidateBytes)) {
      errors.push('weight allocation byte lengths must be safe integers');
    }
  }

  if (expectedControlBytes != null && report.control?.storageByteLength !== expectedControlBytes) {
    errors.push('control storageByteLength must equal inDim * outDim * 4');
  }
  if (expectedCandidateBytes != null && report.candidate?.storageByteLength !== expectedCandidateBytes) {
    errors.push('candidate storageByteLength must equal ceil(inDim * outDim / 2) * 4');
  }

  const control = outputCount == null ? null : decodeOutput('control', report.control, outputCount, errors);
  const candidate = outputCount == null ? null : decodeOutput('candidate', report.candidate, outputCount, errors);
  compareDecoded('control and candidate', control, candidate, report.comparison, errors);
  evaluateLayoutProbe(report.layoutProbe, errors);

  if (report.terminal?.phase !== 'complete' || report.terminal?.primaryOutputWritten !== true) {
    errors.push('terminal report must preserve complete primary output');
  }
  return { ok: errors.length === 0, errors };
}

export function createPackedFp16LinearAdmissionFailureReport(report, errors) {
  if (!report || typeof report !== 'object' || Array.isArray(report)) {
    throw new TypeError('report must be an object');
  }
  if (!Array.isArray(errors) || errors.length === 0 || errors.some(error => typeof error !== 'string')) {
    throw new TypeError('admission errors must be a non-empty string array');
  }
  return {
    ...report,
    ok: false,
    admission: { ok: false, errors: [...errors] },
    failure: { phase: 'admission', message: errors.join('; ') },
    terminal: { phase: 'admission', primaryOutputWritten: true },
  };
}
