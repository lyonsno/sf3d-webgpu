/**
 * Cooperative DINO encoder boundary — first uptake of the Kaminos WebGPU
 * Inference Kit cooperative porting spine (@kaminos/webgpu-inference-kit@^0.1.36)
 * in the sf3d.image-to-mesh.webgpu-local.v0 route.
 *
 * Scope (session 7, cranial-depth-enema directive
 * `adopt-cooperative-porting-spine-0136-in-sf3d`): wire ONLY the DINOv2 ViT
 * encoder — a strictly sequential 24-block loop in sf3d_backbone.js — through
 * the kit's cooperative execution facade as one gpu-command boundary
 * (unit: vit-block, totalItems: 24). This is a scheduling-granularity change,
 * NOT a numerical change: the exact per-block dispatch sequence and tokenBufA/
 * tokenBufB ping-pong are preserved; the only difference from the legacy path
 * is that each block (or fixed chunk of blocks) is submitted as its own command
 * buffer with a browser yield between duties, instead of all 24 blocks landing
 * in one command buffer with one submit.
 *
 * The runtime here is adapter-owned and thin: it wraps the real GPUDevice.queue
 * and exposes exactly the surface createWebGpuCooperativeExecution() consumes
 * (routeId, queue.submit/onSubmittedWorkDone, runInvocation, commandDuties
 * .measureSubmission, hostPhases, yieldToBrowser). No kit inference-runtime,
 * session, or resource-residency machinery is pulled in for this first slice —
 * those are separate pending directives, deliberately kept out to avoid
 * ambiguous parity/perf regression attribution.
 */

import {
  createWebGpuCooperativeExecution,
  defineWebGpuCooperativeBoundaryManifest,
} from '@kaminos/webgpu-inference-kit';

export const SF3D_ROUTE_ID = 'sf3d.image-to-mesh.webgpu-local.v0';
export const DINO_COOPERATIVE_MANIFEST_ID = 'sf3d.dino-encoder-cooperative-boundaries.v0';
export const DINO_BOUNDARY_ID = 'dino-vit-blocks';

/**
 * Declare the DINO encoder cooperative boundary manifest.
 *
 * One phase, one gpu-command boundary over the 24 ViT blocks. Fixed chunking
 * by default (chunkBlocks blocks per submitted duty). progressWeight equals the
 * block count so denominator-bearing progress reads as fraction-of-blocks; no
 * hidden cap — totalItems is exactly the declared block count.
 *
 * @param {number} numBlocks   total ViT blocks (24 for SF3D DINOv2-large)
 * @param {number} chunkBlocks blocks per submitted GPU duty (>=1)
 */
export function defineDinoEncoderManifest(numBlocks, chunkBlocks = 1) {
  if (!Number.isSafeInteger(numBlocks) || numBlocks <= 0) {
    throw new TypeError('numBlocks must be a positive safe integer');
  }
  if (!Number.isSafeInteger(chunkBlocks) || chunkBlocks <= 0) {
    throw new TypeError('chunkBlocks must be a positive safe integer');
  }
  return defineWebGpuCooperativeBoundaryManifest({
    manifestId: DINO_COOPERATIVE_MANIFEST_ID,
    routeId: SF3D_ROUTE_ID,
    phases: [
      {
        phaseId: 'dinov2-tokenizer',
        boundaries: [
          {
            boundaryId: DINO_BOUNDARY_ID,
            kind: 'gpu-command',
            unit: 'vit-block',
            totalItems: numBlocks,
            progressWeight: numBlocks,
            commandDutyKind: 'compute',
            chunking: { mode: 'fixed', chunkItems: chunkBlocks },
            yieldPolicy: 'after-duty',
            resources: {
              retain: ['dinov2.weights'],
              produce: ['dinov2.tokens'],
              release: [],
            },
          },
        ],
      },
    ],
    metadata: { source: 'sf3d-webgpu-cooperative-dino-uptake' },
  });
}

/**
 * Adapter-owned thin runtime over the real GPUDevice.queue.
 *
 * `yieldToBrowser` is a real macrotask yield (MessageChannel) so the browser
 * event loop can service a frame between submitted block duties — this is the
 * cooperative behavior the operator smoke measures. In scheduling-disabled A/B
 * the facade never calls yieldToBrowser, so both arms declare identical work
 * but only the cooperative arm actually cedes the main thread per duty.
 */
export function createSf3dCooperativeRuntime(device) {
  const queue = {
    submit(commandBuffers) {
      device.queue.submit(commandBuffers);
    },
    onSubmittedWorkDone() {
      return device.queue.onSubmittedWorkDone();
    },
  };

  return {
    routeId: SF3D_ROUTE_ID,
    runtimeLabel: 'sf3d-webgpu-cooperative-dino',
    queue,
    hostPhases: {
      snapshot() {
        return { status: 'not-instrumented', phase: 'dinov2-tokenizer' };
      },
    },
    commandDuties: {
      // Observe/route the submission only — do NOT impose our own queue fence
      // here. The facade owns fence policy: in cooperative mode it fences the
      // queue prefix per duty (per-gpu-duty-prefix-fence); in disabled mode it
      // takes exactly one terminal fence. Fencing here would double-fence the
      // cooperative arm and, worse, force per-duty fences onto the disabled arm,
      // destroying the scheduling A/B over identical declared work.
      async measureSubmission(descriptor, submit) {
        return submit();
      },
    },
    // The facade requires this hook before each cooperative GPU encode. In this
    // first facade-only slice there is no separate foreground governor to
    // consult at the boundary — cooperation is expressed by the post-duty
    // browser yield and per-duty queue fence — so preparation is an honest
    // pass-through that returns the descriptor unchanged. When SF3D later adopts
    // the foreground-opportunity interlock, this is where that service goes.
    async prepareCommandDutyAtBoundary(descriptor) {
      return descriptor;
    },
    settleCommandDuty() {
      // No scheduler-owned duty ledger in this slice; settlement is a no-op.
    },
    async runInvocation({ invocationId }, fn) {
      return fn({
        invocationId,
        schedulerRevision: null,
        scheduler: {
          mode: 'cooperative',
          yieldMs: 0,
          waitForSubmittedWorkDone: true,
          phaseChunkSize: {},
        },
        async yieldToBrowser() {
          const start = globalThis.performance?.now?.() ?? Date.now();
          await macrotaskYield();
          const end = globalThis.performance?.now?.() ?? Date.now();
          return { reason: 'cooperative-boundary-duty-complete', elapsedMs: end - start };
        },
      });
    },
  };
}

/**
 * Yield to the browser macrotask queue so a foreground frame can run.
 * MessageChannel gives a true macrotask (unlike microtask await), which is
 * what lets the compositor paint between block duties.
 */
function macrotaskYield() {
  if (typeof MessageChannel === 'function') {
    return new Promise(resolve => {
      const channel = new MessageChannel();
      channel.port1.onmessage = () => {
        channel.port1.close();
        resolve();
      };
      channel.port2.postMessage(null);
    });
  }
  return new Promise(resolve => setTimeout(resolve, 0));
}

/**
 * Run the DINO encoder cooperatively.
 *
 * @param {object}   opts
 * @param {GPUDevice} opts.device
 * @param {object}   opts.tokenizer            SF3DImageTokenizer instance
 * @param {GPUBuffer} opts.imageBuf
 * @param {GPUBuffer} opts.cameraEmbedBuf
 * @param {object}   opts.weights              weights.imageTokenizer
 * @param {number}   opts.numBlocks            24 for SF3D
 * @param {number}  [opts.chunkBlocks=1]       blocks per submitted duty
 * @param {'cooperative'|'disabled'} [opts.schedulingMode='cooperative']
 * @param {(progress:object)=>void} [opts.onProgress]
 * @param {AbortSignal} [opts.signal]
 * @param {string}   [opts.invocationId]
 * @returns {Promise<{ result: object, report: object }>}
 *          result is the tokenizer.encode() output ({ tokensBuf, N, tokenH, tokenW });
 *          report is the kit cooperative execution report.
 */
export async function runCooperativeDino(opts) {
  const {
    device,
    tokenizer,
    imageBuf,
    cameraEmbedBuf,
    weights,
    numBlocks,
    chunkBlocks = 1,
    schedulingMode = 'cooperative',
    onProgress,
    signal,
    invocationId = `sf3d:dino:${schedulingMode}`,
  } = opts;

  const manifest = defineDinoEncoderManifest(numBlocks, chunkBlocks);
  const runtime = createSf3dCooperativeRuntime(device);
  const execution = createWebGpuCooperativeExecution({
    runtime,
    manifest,
    invocationId,
    schedulingMode,
    onProgress,
    signal,
  });

  let result = null;

  await execution.run(async cooperative => {
    const gpu = cooperative.startBoundary(DINO_BOUNDARY_ID);

    // Drive the tokenizer's block loop through the cooperative boundary. The
    // tokenizer owns all encoding, bindings, buffers, dispatch geometry, and
    // ping-pong; the driver only decides WHERE the command buffer is cut and
    // WHEN to submit + yield. When schedulingMode is 'disabled' the facade
    // still requires the same coverage but skips per-duty yields and takes one
    // terminal fence, giving the A/B comparison over identical declared work.
    result = await tokenizer.encodeCooperative({
      imageBuf,
      cameraEmbedBuf,
      weights,
      numBlocks,
      chunkBlocks,
      // driver.runChunk(blockStart, blockEnd, encodeChunk) is called by the
      // tokenizer once per fixed chunk of blocks, in order. encodeChunk(encoder)
      // records that chunk's dispatches into the supplied command encoder.
      driver: async (blockStart, blockEnd, encodeChunk) => {
        const range = gpu.nextRange();
        if (!range) {
          throw new Error(
            `cooperative DINO boundary exhausted ranges before block ${blockStart}`,
          );
        }
        await gpu.runGpuDuty(range, {
          encode() {
            const encoder = device.createCommandEncoder({
              label: `dino-blocks-${blockStart}-${blockEnd}`,
            });
            encodeChunk(encoder);
            return encoder.finish();
          },
          submit(commandBuffer) {
            runtime.queue.submit([commandBuffer]);
          },
        });
      },
    });

    // Boundary must be fully consumed for the facade to accept completion.
    if (gpu.nextRange() != null) {
      throw new Error('cooperative DINO boundary left ranges unconsumed');
    }
  });

  return { result, report: execution.finish() };
}
