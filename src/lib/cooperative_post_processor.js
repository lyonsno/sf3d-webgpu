/**
 * Cooperative SF3D PixelShuffle postprocessor.
 *
 * Each of the three triplane planes is independent. This module cuts the legacy
 * one-command-buffer postprocessor at those dependency-safe boundaries and
 * drives the resulting duties through the Kaminos cooperative execution facade.
 * The convolutions within each plane stay ordered inside one command buffer.
 */

import {
  createWebGpuCooperativeExecution,
  defineWebGpuCooperativeBoundaryManifest,
} from '@kaminos/webgpu-inference-kit';
import { createSf3dCooperativeRuntime, SF3D_ROUTE_ID } from './cooperative_dino.js';
import {
  createPostProcessorOutput,
  dispatchPostProcessorPlane,
} from './post_processor.js';

export const POST_PROCESSOR_MANIFEST_ID = 'sf3d.post-processor-cooperative-boundaries.v0';
export const POST_PROCESSOR_BOUNDARY_ID = 'post-processor-triplane-planes';
export const POST_PROCESSOR_PLANE_COUNT = 3;

export function definePostProcessorManifest(numPlanes = POST_PROCESSOR_PLANE_COUNT) {
  if (!Number.isSafeInteger(numPlanes) || numPlanes <= 0) {
    throw new TypeError('numPlanes must be a positive safe integer');
  }
  if (numPlanes !== POST_PROCESSOR_PLANE_COUNT) {
    throw new RangeError(`SF3D postprocessor must declare exactly ${POST_PROCESSOR_PLANE_COUNT} planes`);
  }
  return defineWebGpuCooperativeBoundaryManifest({
    manifestId: POST_PROCESSOR_MANIFEST_ID,
    routeId: SF3D_ROUTE_ID,
    phases: [
      {
        phaseId: 'triplane-post-processor',
        boundaries: [
          {
            boundaryId: POST_PROCESSOR_BOUNDARY_ID,
            kind: 'gpu-command',
            unit: 'triplane-plane',
            totalItems: numPlanes,
            progressWeight: numPlanes,
            commandDutyKind: 'compute',
            chunking: { mode: 'fixed', chunkItems: 1 },
            yieldPolicy: 'after-duty',
            resources: {
              retain: ['triplane.low-resolution', 'post-processor.weights'],
              produce: ['triplane.high-resolution'],
              release: [],
            },
          },
        ],
      },
    ],
    metadata: { source: 'sf3d-webgpu-cooperative-post-processor' },
  });
}

/**
 * Shared boundary driver used by production and deterministic tests.
 */
export async function drivePostProcessorCooperativeBoundary(cooperative, options) {
  const {
    numPlanes = POST_PROCESSOR_PLANE_COUNT,
    encodePlane,
    submitPlane,
  } = options;
  const gpu = cooperative.startBoundary(POST_PROCESSOR_BOUNDARY_ID);
  let completedPlanes = 0;

  for (let plane = 0; plane < numPlanes; plane++) {
    const range = gpu.nextRange();
    if (!range) {
      throw new Error(`cooperative postprocessor exhausted ranges before plane ${plane}`);
    }
    if (range.itemStart !== plane || range.itemEnd !== plane + 1) {
      throw new Error(
        `cooperative postprocessor range ${range.itemStart}-${range.itemEnd} does not match plane ${plane}`,
      );
    }
    await gpu.runGpuDuty(range, {
      encode: () => encodePlane({ plane, range }),
      submit: (commandBuffer) => submitPlane(commandBuffer, { plane, range }),
    });
    completedPlanes++;
  }

  if (gpu.nextRange() != null) {
    throw new Error('cooperative postprocessor left ranges unconsumed');
  }
  return { completedPlanes, totalPlanes: numPlanes };
}

export async function runCooperativePostProcessor(options) {
  const {
    device,
    triplanesBuf,
    weights,
    schedulingMode = 'cooperative',
    onProgress,
    signal,
    invocationId = `sf3d:post-processor:${schedulingMode}`,
  } = options;
  const manifest = definePostProcessorManifest();
  const runtime = createSf3dCooperativeRuntime(device);
  const execution = createWebGpuCooperativeExecution({
    runtime,
    manifest,
    invocationId,
    schedulingMode,
    onProgress,
    signal,
  });
  const output = createPostProcessorOutput(device);

  await execution.run(async (cooperative) => {
    await drivePostProcessorCooperativeBoundary(cooperative, {
      numPlanes: output.numPlanes,
      encodePlane({ plane }) {
        const encoder = device.createCommandEncoder({
          label: `post-processor-plane-${plane}`,
        });
        dispatchPostProcessorPlane(
          device,
          encoder,
          triplanesBuf,
          weights,
          output,
          plane,
        );
        return encoder.finish();
      },
      submitPlane(commandBuffer) {
        runtime.queue.submit([commandBuffer]);
      },
    });
  });

  return { result: output, report: execution.finish() };
}
