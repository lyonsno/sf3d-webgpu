/**
 * SF3D numerical-parity core (pure Node, no browser, no GPU).
 *
 * Adapter around the @kaminos/webgpu-inference-kit parity primitives
 * (compareWebGpuParityArrays / createWebGpuParityCaptureRegistry) plus a
 * minimal .npy reader for the PyTorch reference tensors written by
 * tools/dump_parity_reference.py.
 *
 * Contract (tools/test_parity_compare_contract.mjs):
 *   - floats compare as Float32Array (FP16 storage is decoded first; Float64 rejects)
 *   - integers compare exactly; non-finite values reject; lengths must match
 *   - stages present on only one side are reported under `missing`, never dropped
 *   - a stage flagged `layoutMismatch: true` is never element-compared; the
 *     report keeps only range/mean stats for it and says so explicitly
 */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import {
  WEBGPU_INFERENCE_KIT_VERSION,
  createWebGpuParityCaptureRegistry,
} from '@kaminos/webgpu-inference-kit';

export const SF3D_PARITY_REPORT_SCHEMA = 'sf3d.parity-comparison-report.v0';

// ---------------------------------------------------------------------------
// FP16
// ---------------------------------------------------------------------------

/** Decode one IEEE 754 binary16 bit pattern to a JavaScript number. */
export function decodeFloat16(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24; // zero / subnormal
  if (exponent === 31) return fraction === 0 ? sign * Infinity : NaN;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

// ---------------------------------------------------------------------------
// NPY reader (format spec v1.0; v2.0/v3.0 header-length widening also handled)
// ---------------------------------------------------------------------------

const NPY_MAGIC = '\x93NUMPY';
const INT32_MIN = -2147483648n;
const INT32_MAX = 2147483647n;

const NPY_DTYPES = {
  '<f4': { itemSize: 4, decode: (bytes, n) => new Float32Array(bytes.buffer, 0, n) },
  '<f2': {
    itemSize: 2,
    decode: (bytes, n) => {
      const halves = new Uint16Array(bytes.buffer, 0, n);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = decodeFloat16(halves[i]);
      return out;
    },
  },
  '<i4': { itemSize: 4, decode: (bytes, n) => new Int32Array(bytes.buffer, 0, n) },
  '<u4': { itemSize: 4, decode: (bytes, n) => new Uint32Array(bytes.buffer, 0, n) },
  '<i2': { itemSize: 2, decode: (bytes, n) => new Int16Array(bytes.buffer, 0, n) },
  '<u2': { itemSize: 2, decode: (bytes, n) => new Uint16Array(bytes.buffer, 0, n) },
  '|i1': { itemSize: 1, decode: (bytes, n) => new Int8Array(bytes.buffer, 0, n) },
  '|u1': { itemSize: 1, decode: (bytes, n) => new Uint8Array(bytes.buffer, 0, n) },
  // int64 (torch long indices) narrows to Int32Array; the kit has no BigInt64 domain.
  '<i8': {
    itemSize: 8,
    decode: (bytes, n, label) => {
      const wide = new BigInt64Array(bytes.buffer, 0, n);
      const out = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        const value = wide[i];
        if (value < INT32_MIN || value > INT32_MAX) {
          throw new RangeError(`${label}: <i8 value ${value} at index ${i} does not fit Int32Array`);
        }
        out[i] = Number(value);
      }
      return out;
    },
  },
};

function parseNpyHeader(headerText, label) {
  const descr = /['"]descr['"]\s*:\s*['"]([^'"]+)['"]/.exec(headerText)?.[1];
  const fortran = /['"]fortran_order['"]\s*:\s*(True|False)/.exec(headerText)?.[1];
  const shapeText = /['"]shape['"]\s*:\s*\(([^)]*)\)/.exec(headerText)?.[1];
  if (descr == null || fortran == null || shapeText == null) {
    throw new Error(`${label}: unparseable NPY header ${JSON.stringify(headerText)}`);
  }
  const shape = shapeText.split(',').map(s => s.trim()).filter(s => s.length > 0).map((s, i) => {
    const dimension = Number(s);
    if (!Number.isSafeInteger(dimension) || dimension < 0) {
      throw new Error(`${label}: NPY shape[${i}] ${JSON.stringify(s)} is not a nonnegative integer`);
    }
    return dimension;
  });
  return { descr, fortranOrder: fortran === 'True', shape };
}

/**
 * Parse an in-memory .npy file.
 * @returns {{ dtype: string, shape: number[], fortranOrder: boolean, elementCount: number, data: TypedArray }}
 */
export function parseNpy(input, label = 'npy') {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (bytes.length < 10 || Buffer.from(bytes.buffer, bytes.byteOffset, 6).toString('latin1') !== NPY_MAGIC) {
    throw new Error(`${label}: not an NPY file (bad magic)`);
  }
  const major = bytes[6];
  const minor = bytes[7];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let headerLength;
  let headerStart;
  if (major === 1) {
    headerLength = view.getUint16(8, true);
    headerStart = 10;
  } else if (major === 2 || major === 3) {
    headerLength = view.getUint32(8, true);
    headerStart = 12;
  } else {
    throw new Error(`${label}: unsupported NPY version ${major}.${minor}`);
  }
  if (bytes.length < headerStart + headerLength) {
    throw new Error(`${label}: NPY header length ${headerLength} exceeds file size ${bytes.length}`);
  }
  const headerText = Buffer.from(bytes.buffer, bytes.byteOffset + headerStart, headerLength).toString('latin1');
  const { descr, fortranOrder, shape } = parseNpyHeader(headerText, label);
  const codec = NPY_DTYPES[descr];
  if (!codec) {
    throw new Error(
      `${label}: unsupported NPY dtype ${descr}; supported: ${Object.keys(NPY_DTYPES).join(', ')}`
      + ' (dump float references as float32/float16 and integers as <=32-bit or int64 within Int32 range)',
    );
  }
  const elementCount = shape.reduce((product, dimension) => product * dimension, 1);
  const dataOffset = headerStart + headerLength;
  const expectedBytes = elementCount * codec.itemSize;
  const actualBytes = bytes.length - dataOffset;
  if (actualBytes !== expectedBytes) {
    throw new Error(
      `${label}: NPY payload is ${actualBytes} bytes, expected ${expectedBytes} for dtype ${descr} shape [${shape.join(', ')}]`,
    );
  }
  // Explicit copy into a fresh, zero-offset ArrayBuffer so typed views are aligned.
  // (Buffer.prototype.slice is a view, not a copy, so it must not be used here.)
  const payload = new Uint8Array(expectedBytes);
  payload.set(bytes.subarray(dataOffset, dataOffset + expectedBytes));
  const data = codec.decode(payload, elementCount, label);
  return { dtype: descr, shape, fortranOrder, elementCount, data };
}

/** Read a .npy file from disk. See parseNpy for the returned record. */
export function readNpy(filePath) {
  return parseNpy(readFileSync(filePath), filePath);
}

/**
 * Load `<dir>/<stageId>.npy` for each stage id that exists on disk.
 * Absent files are skipped (they surface as `missing.reference` in compareStages).
 */
export function loadReferenceStages(referenceDir, stageIds) {
  const stages = {};
  for (const stageId of stageIds) {
    const file = path.join(referenceDir, `${stageId}.npy`);
    if (!existsSync(file)) continue;
    const record = readNpy(file);
    stages[stageId] = {
      values: record.data,
      shape: record.shape,
      dtype: record.dtype,
      fortranOrder: record.fortranOrder,
      file,
    };
  }
  return stages;
}

// ---------------------------------------------------------------------------
// Transfer helper: browser page -> Node (base64 of raw little-endian f32 bytes)
// ---------------------------------------------------------------------------

export function decodeBase64Float32(base64) {
  const bytes = Buffer.from(base64, 'base64');
  if (bytes.length % 4 !== 0) {
    throw new RangeError(`base64 payload is ${bytes.length} bytes, not a multiple of 4 (Float32)`);
  }
  // Copy out of the Buffer pool into a zero-offset ArrayBuffer of exact length.
  const aligned = new Uint8Array(bytes);
  return new Float32Array(aligned.buffer);
}

// ---------------------------------------------------------------------------
// Stats and comparison
// ---------------------------------------------------------------------------

/** Range/mean summary. Non-finite values are counted and excluded from min/max/mean. */
export function summarizeValues(values) {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let compensation = 0;
  let finite = 0;
  let nonFiniteCount = 0;
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (!Number.isFinite(value)) { nonFiniteCount += 1; continue; }
    finite += 1;
    if (value < min) min = value;
    if (value > max) max = value;
    const y = value - compensation;
    const t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
  }
  const summary = {
    min: finite === 0 ? null : min,
    max: finite === 0 ? null : max,
    mean: finite === 0 ? null : sum / finite,
    count: values.length,
  };
  if (nonFiniteCount > 0) summary.nonFiniteCount = nonFiniteCount;
  return summary;
}

function stageStats(stageId, side, descriptor) {
  if (descriptor.stats) {
    const { min, max, mean, count } = descriptor.stats;
    return { min, max, mean, count: count ?? null };
  }
  if (descriptor.values) return summarizeValues(descriptor.values);
  throw new Error(`stage ${stageId}: ${side} descriptor needs values or stats for a stats-only comparison`);
}

function requireValues(stageId, side, descriptor) {
  if (!descriptor.values) {
    throw new Error(`stage ${stageId}: ${side} descriptor has no values for element-wise comparison`);
  }
  if (descriptor.fortranOrder && Array.isArray(descriptor.shape) && descriptor.shape.length > 1) {
    throw new Error(
      `stage ${stageId}: ${side} tensor is column-major (fortran_order) with shape [${descriptor.shape.join(', ')}];`
      + ' element-wise comparison against a row-major WebGPU buffer would compare the wrong elements',
    );
  }
  return descriptor.values;
}

/**
 * Compare per-stage WebGPU captures against PyTorch reference arrays.
 *
 * Stage descriptor: { values?: TypedArray, shape?: number[], stats?: {min,max,mean,count},
 *                     layoutMismatch?: boolean, layoutNote?: string, fortranOrder?: boolean }
 *
 * A stage flagged layoutMismatch on either side is reported stats-only.
 */
export function compareStages(webgpuStages, referenceStages, { runId } = {}) {
  if (typeof runId !== 'string' || runId.trim().length === 0) {
    throw new TypeError('compareStages requires a non-empty runId');
  }
  const registry = createWebGpuParityCaptureRegistry({ runId });
  const webgpuIds = Object.keys(webgpuStages ?? {});
  const referenceIds = Object.keys(referenceStages ?? {});
  const referenceSet = new Set(referenceIds);
  const webgpuSet = new Set(webgpuIds);

  const stages = {};
  const summary = {
    compared: 0,
    exactMatches: 0,
    statsOnly: 0,
    worstRelativeL2: { stageId: null, value: null, status: null },
    worstMaxAbsoluteError: { stageId: null, value: null, sourceIndex: null },
  };
  let worstRelativeL2Rank = -1;

  for (const stageId of webgpuIds) {
    if (!referenceSet.has(stageId)) continue;
    const actual = webgpuStages[stageId];
    const reference = referenceStages[stageId];
    const shape = {
      webgpu: actual.shape ?? null,
      reference: reference.shape ?? null,
    };

    if (actual.layoutMismatch || reference.layoutMismatch) {
      stages[stageId] = {
        stageId,
        mode: 'stats-only',
        layoutMismatch: true,
        layoutNote: actual.layoutNote ?? reference.layoutNote ?? null,
        shape,
        actual: stageStats(stageId, 'webgpu', actual),
        reference: stageStats(stageId, 'reference', reference),
        capture: null,
        comparison: null,
      };
      summary.statsOnly += 1;
      continue;
    }

    const actualValues = requireValues(stageId, 'webgpu', actual);
    const referenceValues = requireValues(stageId, 'reference', reference);
    let capture;
    let comparison;
    try {
      capture = registry.capture(stageId, actualValues, actual.shape ? { shape: actual.shape } : {});
      comparison = registry.compare(stageId, referenceValues);
    } catch (err) {
      throw new Error(`stage ${stageId}: ${err.message}`, { cause: err });
    } finally {
      registry.release(stageId);
    }

    stages[stageId] = {
      stageId,
      mode: 'element-wise',
      layoutMismatch: false,
      layoutNote: null,
      shape,
      capture,
      comparison,
    };
    summary.compared += 1;
    if (comparison.metrics.exactMatch) summary.exactMatches += 1;

    const { metrics } = comparison;
    // A null relative L2 means nonzero error against an all-zero reference: rank it worst.
    const relativeRank = metrics.relativeL2Error == null ? Infinity : metrics.relativeL2Error;
    if (relativeRank > worstRelativeL2Rank) {
      worstRelativeL2Rank = relativeRank;
      summary.worstRelativeL2 = {
        stageId,
        value: metrics.relativeL2Error,
        status: metrics.relativeL2Status,
      };
    }
    if (summary.worstMaxAbsoluteError.value == null || metrics.maxAbsoluteError > summary.worstMaxAbsoluteError.value) {
      summary.worstMaxAbsoluteError = {
        stageId,
        value: metrics.maxAbsoluteError,
        sourceIndex: metrics.worstSourceIndex,
      };
    }
  }

  registry.clear();

  return {
    schema: SF3D_PARITY_REPORT_SCHEMA,
    runId,
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    stages,
    summary,
    missing: {
      webgpu: referenceIds.filter(stageId => !webgpuSet.has(stageId)),
      reference: webgpuIds.filter(stageId => !referenceSet.has(stageId)),
    },
  };
}


// ---------------------------------------------------------------------------
// Reference provenance (review 2026-09-16, MEDIUM). A reference directory is
// parity evidence only with a manifest written by dump_parity_reference.py
// that binds every artifact hash, the input image, and the source/model
// identities. Any mismatch is named; a missing manifest is refused.
// ---------------------------------------------------------------------------
import { createHash } from 'node:crypto';
import { existsSync as _existsSync, readFileSync as _readFileSync, openSync as _openSync, readSync as _readSync, closeSync as _closeSync } from 'node:fs';
import { join as _join } from 'node:path';

export const PARITY_REFERENCE_MANIFEST_SCHEMA = 'sf3d.parity-reference-manifest.v0';

/** Streaming (chunked, synchronous) SHA-256 — weights.bin is larger than Node's 2 GiB readFileSync limit. */
export function sha256File(filePath, chunkBytes = 8 * 1024 * 1024) {
  const hash = createHash('sha256');
  const fd = _openSync(filePath, 'r');
  try {
    const buf = Buffer.allocUnsafe(chunkBytes);
    let n;
    while ((n = _readSync(fd, buf, 0, chunkBytes, null)) > 0) hash.update(buf.subarray(0, n));
  } finally {
    _closeSync(fd);
  }
  return hash.digest('hex');
}

/** Load and structurally validate <dir>/manifest.json; throws when absent or wrong. */
export function loadReferenceManifest(referenceDir) {
  const manifestPath = _join(referenceDir, 'manifest.json');
  if (!_existsSync(manifestPath)) {
    throw new Error(`no manifest.json in ${referenceDir}: the reference is not provenance-bound (regenerate with tools/dump_parity_reference.py)`);
  }
  const manifest = JSON.parse(_readFileSync(manifestPath, 'utf8'));
  if (manifest?.schema !== PARITY_REFERENCE_MANIFEST_SCHEMA) {
    throw new Error(`manifest schema ${manifest?.schema ?? 'missing'} != ${PARITY_REFERENCE_MANIFEST_SCHEMA}`);
  }
  if (!manifest.artifacts || typeof manifest.artifacts !== 'object' || Object.keys(manifest.artifacts).length === 0) {
    throw new Error('manifest lists no artifacts');
  }
  return manifest;
}

/**
 * Verify every manifest artifact hash against the files on disk, the input
 * image hash against the manifest, and the presence of source/model identity.
 * Returns { ok, errors, identities } — never throws on a mismatch.
 */
export function verifyReferenceProvenance(referenceDir, manifest, { inputSha256 = null } = {}) {
  const errors = [];
  const artifacts = manifest?.artifacts || {};
  for (const [name, expected] of Object.entries(artifacts)) {
    const full = _join(referenceDir, name);
    if (!_existsSync(full)) { errors.push(`${name} missing from ${referenceDir}`); continue; }
    const actual = sha256File(full);
    if (actual !== expected?.sha256) errors.push(`${name} sha256 ${actual.slice(0, 12)}… != manifest ${String(expected?.sha256 ?? 'missing').slice(0, 12)}…`);
  }
  if (inputSha256 != null) {
    if (manifest?.input?.sha256 !== inputSha256) {
      errors.push(`input image sha256 ${inputSha256.slice(0, 12)}… != manifest ${String(manifest?.input?.sha256 ?? 'missing').slice(0, 12)}…`);
    }
  } else {
    errors.push('input image sha256 not supplied for provenance check');
  }
  if (!manifest?.sf3d?.commit) errors.push('sf3d source commit missing from manifest');
  if (!manifest?.model?.repo_id) errors.push('model identity missing from manifest');
  const identities = Object.freeze({
    generatedAt: manifest?.generated_at ?? null,
    inputSha256: manifest?.input?.sha256 ?? null,
    sf3dCommit: manifest?.sf3d?.commit ?? null,
    sf3dDirty: manifest?.sf3d?.dirty ?? null,
    generatorCommit: manifest?.generator?.sf3d_webgpu?.commit ?? null,
    generatorScriptSha256: manifest?.generator?.script_sha256 ?? null,
    modelRepoId: manifest?.model?.repo_id ?? null,
    modelSnapshotCommit: manifest?.model?.snapshot_commit ?? null,
    modelWeightsSha256: manifest?.model?.weights_sha256 ?? null,
    torchVersion: manifest?.torch?.version ?? null,
    artifactCount: Object.keys(artifacts).length,
  });
  return Object.freeze({ ok: errors.length === 0, errors: Object.freeze(errors), identities });
}
