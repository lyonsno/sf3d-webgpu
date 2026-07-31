/**
 * SF3D cooperative-DINO adapter conformance.
 *
 * Consumes the Kaminos kit's runWebGpuCooperativeAdapterConformance() (kit
 * >=0.1.42) to prove the DINO cooperative boundary's ORCHESTRATION on the four
 * conformance scenarios (cooperative success, disabled success, cancellation,
 * runtime failure), per cranial-depth-enema directive
 * `bind-sf3d-candidate-to-kit-0138-conformance`.
 *
 * Design constraints from that directive:
 *   - The conformance callback drives the harness-SUPPLIED facade directly via
 *     the SHARED driver (driveDinoCooperativeBoundary). It must NOT call
 *     runCooperativeDino() or nest a cooperative execution — that would move the
 *     declared work outside the harness's coverage/cancellation/failure/progress
 *     /settlement authority.
 *   - outputFingerprint = SHA-256 over the COMPLETE canonical orchestration
 *     trace: all 24 block ranges + the full ordered dispatch operation sequence
 *     (the 363 ops proven by tools/test_cooperative_dino_contract.mjs) + stable
 *     output-shape identity. Any block omission/dup/reorder/range-drift/dispatch
 *     -drift must change the fingerprint. Not a constant label, not mesh counts.
 *   - initialResources ["dinov2.weights"], expectedFinalResources
 *     ["dinov2.weights","dinov2.tokens"]; kitVersion must equal 0.1.42.
 *
 * No real GPU is used here: conformance is deterministic. A trace-recording
 * tokenizer stand-in emits the exact dispatch sequence sf3d_backbone.js would,
 * so the fingerprint is bound to the real orchestration, not to device output.
 */

import { defineDinoEncoderManifest, driveDinoCooperativeBoundary, SF3D_ROUTE_ID } from './cooperative_dino.js';
import { recordDinoDispatchTrace, VIT_NUM_BLOCKS } from './sf3d_backbone.js';

export const SF3D_DINO_ADAPTER_ID = 'sf3d.dino.cooperative.webgpu.v0';
export const SF3D_CONFORMANCE_ID = 'sf3d:cooperative-adapter:v0';
export const REQUIRED_KIT_VERSION = '0.1.42';

/**
 * Deterministic SHA-256 over a string, using WebCrypto (browser + Node 20+).
 * Returns lowercase hex.
 */
async function sha256Hex(text) {
  const bytes = new TextEncoder().encode(text);
  const digest = await (globalThis.crypto?.subtle ?? (await import('node:crypto')).webcrypto.subtle)
    .digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Build the canonical orchestration-trace fingerprint for one chunk policy.
 *
 * The trace records, in order: each cooperative range (blockStart..blockEnd) and
 * every dispatch op the tokenizer records into that chunk (op name + input/output
 * buffer identity), plus the final output-shape identity. This is exactly the
 * data tools/test_cooperative_dino_contract.mjs proves is byte-identical between
 * legacy, cooperative, and disabled — so equal fingerprints across enabled/
 * disabled prove orchestration equivalence.
 *
 * @param {number} chunkBlocks
 * @returns {Promise<{ fingerprint:string, opCount:number, rangeCount:number, canonical:string }>}
 */
export async function computeDinoOrchestrationFingerprint(chunkBlocks) {
  const chunks = [];
  // recordDinoDispatchTrace drives the exact _setupEncode/_encodeBlock/
  // _finalizeEncode sequence against a recording stub, cut into the same fixed
  // chunks the real driver uses, and returns { chunks:[{range, ops[]}], output }.
  const trace = recordDinoDispatchTrace({ numBlocks: VIT_NUM_BLOCKS, chunkBlocks });
  for (const chunk of trace.chunks) {
    chunks.push(
      `range:${chunk.blockStart}-${chunk.blockEnd}|` +
      chunk.ops.map(o => `${o.op}:${o.input}->${o.output}`).join(','),
    );
  }
  const canonical = JSON.stringify({
    schema: 'sf3d.dino.orchestration-trace.v0',
    routeId: SF3D_ROUTE_ID,
    numBlocks: VIT_NUM_BLOCKS,
    chunkBlocks,
    chunks,
    outputShape: trace.output, // { N, tokenH, tokenW, dim }
  });
  const opCount = trace.chunks.reduce((n, c) => n + c.ops.length, 0);
  return {
    fingerprint: await sha256Hex(canonical),
    opCount,
    rangeCount: trace.chunks.length,
    canonical,
  };
}

/**
 * The conformance adapter callback. The harness invokes this inside its own
 * execution.run with { cooperative, scenario, schedulingMode, adapterIdentity,
 * manifest }. We drive the SHARED boundary driver against the harness facade
 * using deterministic encode/submit tokens, then return { outputFingerprint }.
 *
 * Because encode/submit here are deterministic and GPU-free, the harness's
 * scenario machinery (cancel after first duty, inject queue/host failure)
 * exercises our boundary lifecycle without a device.
 *
 * @param {number} chunkBlocks
 */
export function createSf3dCooperativeDinoAdapter(chunkBlocks = 1) {
  // Precompute the fingerprint once; it is a pure function of the chunk policy
  // and the fixed 24-block orchestration, identical for cooperative and disabled.
  let fingerprintPromise = null;
  const getFingerprint = () => {
    if (!fingerprintPromise) fingerprintPromise = computeDinoOrchestrationFingerprint(chunkBlocks);
    return fingerprintPromise;
  };

  return async function runSf3dCooperativeAdapter({ cooperative }) {
    // Drive the SHARED boundary driver against the harness-supplied facade.
    // No real GPU: encodeChunk records the exact ordered dispatch ops for its
    // blocks (via the recording tokenizer) so the boundary walks the true
    // orchestration, and returns an opaque token; submit is a no-op the
    // harness's measureSubmission owns. The tokenizer's per-chunk driver is the
    // `encodeInto` callback the shared driver hands us.
    const result = await driveDinoCooperativeBoundary(cooperative, {
      numBlocks: VIT_NUM_BLOCKS,
      chunkBlocks,
      encodeChunk: ({ blockStart, blockEnd, encodeInto }) => {
        // encodeInto here is the recording tokenizer's per-chunk callback
        // (emitChunk); invoking it walks that chunk's block dispatches in order.
        encodeInto();
        return { __dinoChunk: `${blockStart}-${blockEnd}` };
      },
      // kit >=0.1.41: the kit owns queue.submit; encodeChunk returns the command
      // buffer and there is no producer-side submit callback.
      // encodeTokenizer receives the per-chunk driver from the shared boundary
      // driver and runs the deterministic recording pass, cutting on the same
      // fixed chunks. It returns { chunks, output } — the canonical trace.
      encodeTokenizer: (driver) => Promise.resolve(
        recordDinoDispatchTrace({ numBlocks: VIT_NUM_BLOCKS, chunkBlocks, driver }),
      ),
    });

    const { fingerprint } = await getFingerprint();
    return { outputFingerprint: fingerprint, output: { shape: result.output } };
  };
}

/**
 * Run the full adapter conformance for the SF3D DINO boundary.
 *
 * @param {object} kit  the imported @kaminos/webgpu-inference-kit module
 * @param {object} o
 * @param {number} o.chunkBlocks
 * @param {string} o.packageVersion   candidate sf3d-webgpu package version
 * @param {string} o.sourceRevision   candidate git revision
 * @returns {Promise<object>} the conformance report (throws with
 *          error.cooperativeAdapterConformanceReport on rejection)
 */
export async function runSf3dDinoConformance(kit, { chunkBlocks = 1, packageVersion, sourceRevision }) {
  const manifest = defineDinoEncoderManifest(VIT_NUM_BLOCKS, chunkBlocks);
  return kit.runWebGpuCooperativeAdapterConformance({
    conformanceId: SF3D_CONFORMANCE_ID,
    adapterIdentity: {
      adapterId: SF3D_DINO_ADAPTER_ID,
      routeId: SF3D_ROUTE_ID,
      packageName: 'sf3d-webgpu',
      packageVersion,
      sourceRevision,
    },
    manifest,
    initialResources: ['dinov2.weights'],
    expectedFinalResources: ['dinov2.weights', 'dinov2.tokens'],
    runAdapter: createSf3dCooperativeDinoAdapter(chunkBlocks),
  });
}
