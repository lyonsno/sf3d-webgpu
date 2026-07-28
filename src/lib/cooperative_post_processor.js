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
  createPostProcessorPlaneState,
  dispatchPostProcessorPlane,
  dispatchPostProcessorPlaneStage,
  POST_PROCESSOR_PLANE_STAGE_IDS,
} from './post_processor.js';

export const POST_PROCESSOR_MANIFEST_ID = 'sf3d.post-processor-cooperative-boundaries.v0';
export const POST_PROCESSOR_BOUNDARY_ID = 'post-processor-triplane-planes';
export const POST_PROCESSOR_PLANE_COUNT = 3;
export const POST_PROCESSOR_LAYER_MANIFEST_ID =
  'sf3d.post-processor-layer-cooperative-boundaries.v0';
export const POST_PROCESSOR_LAYER_BOUNDARY_ID = 'post-processor-triplane-stages';
export { POST_PROCESSOR_PLANE_STAGE_IDS };
export const POST_PROCESSOR_LAYER_DUTY_COUNT =
  POST_PROCESSOR_PLANE_COUNT * POST_PROCESSOR_PLANE_STAGE_IDS.length;

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

export function definePostProcessorLayerManifest() {
  return defineWebGpuCooperativeBoundaryManifest({
    manifestId: POST_PROCESSOR_LAYER_MANIFEST_ID,
    routeId: SF3D_ROUTE_ID,
    phases: [
      {
        phaseId: 'triplane-post-processor',
        boundaries: [
          {
            boundaryId: POST_PROCESSOR_LAYER_BOUNDARY_ID,
            kind: 'gpu-command',
            unit: 'post-processor-stage',
            totalItems: POST_PROCESSOR_LAYER_DUTY_COUNT,
            progressWeight: POST_PROCESSOR_LAYER_DUTY_COUNT,
            commandDutyKind: 'compute',
            chunking: { mode: 'fixed', chunkItems: 1 },
            yieldPolicy: 'after-duty',
            resources: {
              retain: [
                'triplane.low-resolution',
                'post-processor.weights',
                'post-processor.intermediates',
              ],
              produce: ['triplane.high-resolution'],
              release: [],
            },
          },
        ],
      },
    ],
    metadata: { source: 'sf3d-webgpu-cooperative-post-processor-layers' },
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

export async function drivePostProcessorLayerBoundary(cooperative, options) {
  const {
    encodeStage,
    submitStage,
    now = () => globalThis.performance?.now?.() ?? Date.now(),
  } = options;
  const gpu = cooperative.startBoundary(POST_PROCESSOR_LAYER_BOUNDARY_ID);
  const telemetry = [];

  for (let dutyIndex = 0; dutyIndex < POST_PROCESSOR_LAYER_DUTY_COUNT; dutyIndex++) {
    const range = gpu.nextRange();
    if (!range) {
      throw new Error(`cooperative postprocessor exhausted ranges before duty ${dutyIndex}`);
    }
    if (range.itemStart !== dutyIndex || range.itemEnd !== dutyIndex + 1) {
      throw new Error(
        `cooperative postprocessor range ${range.itemStart}-${range.itemEnd} `
        + `does not match duty ${dutyIndex}`,
      );
    }
    const plane = Math.floor(dutyIndex / POST_PROCESSOR_PLANE_STAGE_IDS.length);
    const stageIndex = dutyIndex % POST_PROCESSOR_PLANE_STAGE_IDS.length;
    const stageId = POST_PROCESSOR_PLANE_STAGE_IDS[stageIndex];
    const timing = {
      dutyIndex,
      plane,
      stageIndex,
      stageId,
      dutyStartedAtMs: now(),
      encodeStartedAtMs: null,
      encodeCompletedAtMs: null,
      submitStartedAtMs: null,
      submitCompletedAtMs: null,
      dutyCompletedAtMs: null,
      encodeMs: null,
      submitMs: null,
      dutyMs: null,
    };

    await gpu.runGpuDuty(range, {
      encode: () => {
        timing.encodeStartedAtMs = now();
        const commandBuffer = encodeStage({
          dutyIndex,
          plane,
          stageIndex,
          stageId,
          range,
        });
        timing.encodeCompletedAtMs = now();
        timing.encodeMs = timing.encodeCompletedAtMs - timing.encodeStartedAtMs;
        return commandBuffer;
      },
      submit: commandBuffer => {
        timing.submitStartedAtMs = now();
        const result = submitStage(commandBuffer, {
          dutyIndex,
          plane,
          stageIndex,
          stageId,
          range,
        });
        timing.submitCompletedAtMs = now();
        timing.submitMs = timing.submitCompletedAtMs - timing.submitStartedAtMs;
        return result;
      },
    });
    timing.dutyCompletedAtMs = now();
    timing.dutyMs = timing.dutyCompletedAtMs - timing.dutyStartedAtMs;
    telemetry.push(Object.freeze(timing));
  }

  if (gpu.nextRange() != null) {
    throw new Error('cooperative postprocessor left ranges unconsumed');
  }
  return {
    completedDuties: telemetry.length,
    totalDuties: POST_PROCESSOR_LAYER_DUTY_COUNT,
    telemetry: Object.freeze(telemetry),
  };
}

export async function runCooperativePostProcessor(options) {
  const {
    device,
    triplanesBuf,
    weights,
    schedulingMode = 'cooperative',
    dutyGranularity = 'plane',
    onProgress,
    signal,
    invocationId = `sf3d:post-processor:${schedulingMode}`,
  } = options;
  if (!['plane', 'layer'].includes(dutyGranularity)) {
    throw new RangeError(`unsupported postprocessor duty granularity: ${dutyGranularity}`);
  }
  const manifest = dutyGranularity === 'layer'
    ? definePostProcessorLayerManifest()
    : definePostProcessorManifest();
  const queueFences = [];
  const browserYields = [];
  const runtime = createSf3dCooperativeRuntime(device, {
    onQueueFenceResolved: event => queueFences.push(Object.freeze({ ...event })),
    onBrowserYield: event => browserYields.push(Object.freeze({ ...event })),
  });
  const execution = createWebGpuCooperativeExecution({
    runtime,
    manifest,
    invocationId,
    schedulingMode,
    onProgress,
    signal,
  });
  const output = createPostProcessorOutput(device);
  let dutyTelemetry = [];

  await execution.run(async (cooperative) => {
    if (dutyGranularity === 'layer') {
      const planeStates = Array(output.numPlanes).fill(null);
      const driven = await drivePostProcessorLayerBoundary(cooperative, {
        encodeStage({ plane, stageIndex, stageId }) {
          if (!planeStates[plane]) {
            planeStates[plane] = createPostProcessorPlaneState(
              device,
              triplanesBuf,
              weights,
              output,
              plane,
            );
          }
          const encoder = device.createCommandEncoder({
            label: `post-processor-plane-${plane}-${stageId}`,
          });
          dispatchPostProcessorPlaneStage(
            device,
            encoder,
            planeStates[plane],
            stageIndex,
          );
          return encoder.finish();
        },
        submitStage(commandBuffer) {
          runtime.queue.submit([commandBuffer]);
        },
      });
      dutyTelemetry = driven.telemetry;
      return;
    }

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

  const report = execution.finish();
  return {
    result: output,
    report: Object.freeze({
      ...report,
      adapterTelemetry: Object.freeze({
        dutyGranularity,
        stageDuties: Object.freeze(dutyTelemetry),
        queueFences: Object.freeze(queueFences),
        browserYields: Object.freeze(browserYields),
      }),
    }),
  };
}
