/**
 * Cooperative texture-bake boundary — second SF3D cooperative uptake of the
 * Kaminos cooperative porting spine.
 *
 * Profiling (foreground-tail-stage-attribution_2026-07-24) showed the texture
 * bake's single monolithic triplane-decode dispatch + readback owns a ~231ms
 * contiguous foreground gap — the largest GPU gap in the pipeline. Per-texel
 * decode is fully independent, so we split the occupied texels into fixed
 * batches and decode each as its own GPU command duty through the cooperative
 * facade (submit + queue-prefix fence + browser yield between batches),
 * collapsing the one big gap into yieldable pieces while preserving byte-
 * identical texture output.
 *
 * Reuses the exact command-duty pattern proven at the DINO boundary: the
 * adapter-owned thin runtime (createSf3dCooperativeRuntime) and the boundary/
 * duty facade. This is the GPU cooperative mechanism, not a CPU worker offload.
 */

import {
  createWebGpuCooperativeExecution,
  defineWebGpuCooperativeBoundaryManifest,
} from '@kaminos/webgpu-inference-kit';
import { createSf3dCooperativeRuntime, SF3D_ROUTE_ID } from './cooperative_dino.js';

export const TEXTURE_BAKE_MANIFEST_ID = 'sf3d.texture-bake-cooperative-boundaries.v0';
export const TEXTURE_BAKE_BOUNDARY_ID = 'texture-bake-texel-batches';

/**
 * Declare the texture-bake cooperative boundary manifest: one gpu-command
 * boundary over the occupied texels, fixed chunking by batchTexels.
 */
export function defineTextureBakeManifest(numOccupied, batchTexels) {
  if (!Number.isSafeInteger(numOccupied) || numOccupied <= 0) {
    throw new TypeError('numOccupied must be a positive safe integer');
  }
  if (!Number.isSafeInteger(batchTexels) || batchTexels <= 0) {
    throw new TypeError('batchTexels must be a positive safe integer');
  }
  return defineWebGpuCooperativeBoundaryManifest({
    manifestId: TEXTURE_BAKE_MANIFEST_ID,
    routeId: SF3D_ROUTE_ID,
    phases: [
      {
        phaseId: 'texture-bake',
        boundaries: [
          {
            boundaryId: TEXTURE_BAKE_BOUNDARY_ID,
            kind: 'gpu-command',
            unit: 'texel-batch',
            totalItems: numOccupied,
            progressWeight: numOccupied,
            commandDutyKind: 'compute',
            chunking: { mode: 'fixed', chunkItems: batchTexels },
            yieldPolicy: 'after-duty',
            resources: {
              retain: ['triplane.features', 'decoder.weights'],
              produce: ['texture.albedo', 'texture.normal'],
              release: [],
            },
          },
        ],
      },
    ],
    metadata: { source: 'sf3d-webgpu-cooperative-texture-bake' },
  });
}

/**
 * Build the `cooperativeBatch` callback that bakeTexture/decodeTexelFeatures
 * consumes. It runs a real cooperative execution over the texel boundary and,
 * for each batch, encodes+submits the decode duty (facade fences the queue
 * prefix) and then reads back that batch's outputs.
 *
 * @param {GPUDevice} device
 * @param {object} opts { batchTexels, schedulingMode, onProgress, signal, invocationId }
 * @returns {(numOccupied:number, makeBatch:(start,end)=>Promise<{encode,submit,readback}>)=>Promise<object>}
 *   The returned function resolves to the cooperative execution report.
 */
export function makeCooperativeTextureBake(device, opts = {}) {
  const {
    batchTexels = 16384,
    schedulingMode = 'cooperative',
    onProgress,
    signal,
    invocationId = `sf3d:texture-bake:${schedulingMode}`,
  } = opts;

  return async function cooperativeBatch(numOccupied, makeBatch) {
    const batch = Math.min(batchTexels, numOccupied);
    const manifest = defineTextureBakeManifest(numOccupied, batch);
    const runtime = createSf3dCooperativeRuntime(device);
    const execution = createWebGpuCooperativeExecution({
      runtime, manifest, invocationId, schedulingMode, onProgress, signal,
    });

    await execution.run(async (cooperative) => {
      const gpu = cooperative.startBoundary(TEXTURE_BAKE_BOUNDARY_ID);
      let range;
      while ((range = gpu.nextRange()) != null) {
        const start = range.itemStart, end = range.itemEnd;
        const b = await makeBatch(start, end);
        await gpu.runGpuDuty(range, {
          encode: () => b.encode(),
          submit: (cb) => b.submit(cb),
        });
        // The facade has fenced this batch's queue prefix (cooperative mode) or
        // will take one terminal fence (disabled); read back the batch outputs.
        await b.readback();
      }
    });

    return execution.finish();
  };
}
