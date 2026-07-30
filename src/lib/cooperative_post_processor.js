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
  dispatchPostProcessorChannelDuty,
  dispatchPostProcessorPlane,
  dispatchPostProcessorPlaneStage,
  POST_PROCESSOR_CONV_OUTPUT_CHANNELS,
  POST_PROCESSOR_PLANE_STAGE_IDS,
} from './post_processor.js';

export const POST_PROCESSOR_MANIFEST_ID = 'sf3d.post-processor-cooperative-boundaries.v0';
export const POST_PROCESSOR_BOUNDARY_ID = 'post-processor-triplane-planes';
export const POST_PROCESSOR_PLANE_COUNT = 3;
export const POST_PROCESSOR_LAYER_MANIFEST_ID =
  'sf3d.post-processor-layer-cooperative-boundaries.v0';
export const POST_PROCESSOR_LAYER_BOUNDARY_ID = 'post-processor-triplane-stages';
export const POST_PROCESSOR_CHANNEL_MANIFEST_ID =
  'sf3d.post-processor-channel-cooperative-boundaries.v0';
export const POST_PROCESSOR_CHANNEL_BOUNDARY_ID =
  'post-processor-triplane-channel-ranges';
export { POST_PROCESSOR_PLANE_STAGE_IDS };
export const POST_PROCESSOR_LAYER_DUTY_COUNT =
  POST_PROCESSOR_PLANE_COUNT * POST_PROCESSOR_PLANE_STAGE_IDS.length;

function buildPostProcessorChannelDutyPlan(channelsPerDuty) {
  const duties = [];
  for (let plane = 0; plane < POST_PROCESSOR_PLANE_COUNT; plane++) {
    duties.push({
      plane,
      kind: 'gather',
      stageIndex: 0,
      stageId: POST_PROCESSOR_PLANE_STAGE_IDS[0],
    });
    for (let layerIndex = 0; layerIndex < POST_PROCESSOR_CONV_OUTPUT_CHANNELS.length; layerIndex++) {
      const totalChannels = POST_PROCESSOR_CONV_OUTPUT_CHANNELS[layerIndex];
      const rangeCount = Math.ceil(totalChannels / channelsPerDuty);
      for (let rangeIndex = 0; rangeIndex < rangeCount; rangeIndex++) {
        const channelStart = rangeIndex * channelsPerDuty;
        const channelEnd = Math.min(totalChannels, channelStart + channelsPerDuty);
        duties.push({
          plane,
          kind: 'conv-range',
          stageIndex: layerIndex + 1,
          stageId: POST_PROCESSOR_PLANE_STAGE_IDS[layerIndex + 1],
          layerIndex,
          rangeIndex,
          rangeCount,
          channelStart,
          channelCount: channelEnd - channelStart,
          channelEnd,
          totalChannels,
        });
      }
    }
    duties.push({
      plane,
      kind: 'pixel-shuffle-copy',
      stageIndex: POST_PROCESSOR_PLANE_STAGE_IDS.length - 1,
      stageId: POST_PROCESSOR_PLANE_STAGE_IDS.at(-1),
    });
  }
  const totalDuties = duties.length;
  return Object.freeze({
    channelsPerDuty,
    duties: Object.freeze(duties.map((duty, dutyIndex) => Object.freeze({
      ...duty,
      dutyIndex,
      totalDuties,
      channelsPerDuty,
    }))),
  });
}

export function createPostProcessorChannelDutyPlan(channelsPerDuty) {
  if (!Number.isSafeInteger(channelsPerDuty) || channelsPerDuty <= 0) {
    throw new TypeError('channelsPerDuty must be a positive safe integer');
  }
  return buildPostProcessorChannelDutyPlan(channelsPerDuty);
}

function requireExactPostProcessorChannelDutyPlan(plan) {
  if (!plan || !Number.isSafeInteger(plan.channelsPerDuty) || plan.channelsPerDuty <= 0
      || !Array.isArray(plan.duties)) {
    throw new TypeError('invalid postprocessor channel duty plan');
  }
  const expected = buildPostProcessorChannelDutyPlan(plan.channelsPerDuty);
  if (plan.duties.length !== expected.duties.length) {
    throw new Error('postprocessor channel duty plan has the wrong duty count');
  }
  for (let index = 0; index < expected.duties.length; index++) {
    const actualDuty = plan.duties[index];
    const expectedDuty = expected.duties[index];
    for (const key of Object.keys(expectedDuty)) {
      if (actualDuty?.[key] !== expectedDuty[key]) {
        throw new Error(
          `postprocessor channel duty plan mismatch at duty ${index} field ${key}`,
        );
      }
    }
  }
  return expected;
}

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

export function definePostProcessorChannelManifest(channelsPerDuty) {
  const plan = createPostProcessorChannelDutyPlan(channelsPerDuty);
  return defineWebGpuCooperativeBoundaryManifest({
    manifestId: POST_PROCESSOR_CHANNEL_MANIFEST_ID,
    routeId: SF3D_ROUTE_ID,
    phases: [
      {
        phaseId: 'triplane-post-processor',
        boundaries: [
          {
            boundaryId: POST_PROCESSOR_CHANNEL_BOUNDARY_ID,
            kind: 'gpu-command',
            unit: 'post-processor-channel-duty',
            totalItems: plan.duties.length,
            progressWeight: plan.duties.length,
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
            metadata: { channelsPerDuty },
          },
        ],
      },
    ],
    metadata: {
      source: 'sf3d-webgpu-cooperative-post-processor-channels',
      channelsPerDuty,
    },
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

export async function drivePostProcessorChannelBoundary(cooperative, options) {
  const {
    plan: suppliedPlan,
    encodeDuty,
    submitDuty,
    now = () => globalThis.performance?.now?.() ?? Date.now(),
  } = options;
  const plan = requireExactPostProcessorChannelDutyPlan(suppliedPlan);
  const gpu = cooperative.startBoundary(POST_PROCESSOR_CHANNEL_BOUNDARY_ID);
  const telemetry = [];

  for (const duty of plan.duties) {
    const range = gpu.nextRange();
    if (!range) {
      throw new Error(
        `cooperative postprocessor exhausted ranges before channel duty ${duty.dutyIndex}`,
      );
    }
    if (range.itemStart !== duty.dutyIndex || range.itemEnd !== duty.dutyIndex + 1) {
      throw new Error(
        `cooperative postprocessor range ${range.itemStart}-${range.itemEnd} `
        + `does not match channel duty ${duty.dutyIndex}`,
      );
    }
    const timing = {
      ...duty,
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
        const commandBuffer = encodeDuty(duty, range);
        timing.encodeCompletedAtMs = now();
        timing.encodeMs = timing.encodeCompletedAtMs - timing.encodeStartedAtMs;
        return commandBuffer;
      },
      submit: commandBuffer => {
        timing.submitStartedAtMs = now();
        const result = submitDuty(commandBuffer, duty, range);
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
    throw new Error('cooperative postprocessor left channel ranges unconsumed');
  }
  return {
    completedDuties: telemetry.length,
    totalDuties: plan.duties.length,
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
    channelsPerDuty = 16,
    onProgress,
    signal,
    invocationId = `sf3d:post-processor:${schedulingMode}`,
    // Bounded-prefix completion (kit >=0.1.41): allow up to maxInFlightGpuDuties
    // GPU duties to be in flight before the facade fences a prefix, instead of
    // strict per-duty prefix fencing. Default null → strict-prefix (unchanged).
    // Per Cranial's composition contract, bounded-prefix is opt-in ONLY on the
    // fixed postprocessor channel-range boundary; plane/layer/adaptive stay
    // strict. The kit itself also rejects bounded-prefix on adaptive boundaries
    // and outside cooperative scheduling.
    completionPolicy = 'strict-prefix',
    maxInFlightGpuDuties = null,
  } = options;
  if (!['plane', 'layer', 'channel-range'].includes(dutyGranularity)) {
    throw new RangeError(`unsupported postprocessor duty granularity: ${dutyGranularity}`);
  }
  if (!Number.isSafeInteger(channelsPerDuty) || channelsPerDuty <= 0) {
    throw new TypeError('channelsPerDuty must be a positive safe integer');
  }
  if (!['strict-prefix', 'bounded-prefix'].includes(completionPolicy)) {
    throw new RangeError(`completionPolicy must be strict-prefix or bounded-prefix; got ${completionPolicy}`);
  }
  if (completionPolicy === 'bounded-prefix') {
    // Bounded-prefix is only lawful on the fixed channel-range boundary under
    // cooperative scheduling (Cranial contract + kit constraint). Fail loud
    // rather than silently downgrade.
    if (dutyGranularity !== 'channel-range') {
      throw new RangeError(`bounded-prefix completion is only supported for channel-range duties; got ${dutyGranularity}`);
    }
    if (schedulingMode !== 'cooperative') {
      throw new RangeError('bounded-prefix completion requires cooperative scheduling');
    }
    if (!Number.isSafeInteger(maxInFlightGpuDuties) || maxInFlightGpuDuties <= 0) {
      throw new TypeError('bounded-prefix completion requires a positive maxInFlightGpuDuties');
    }
  } else if (maxInFlightGpuDuties != null) {
    throw new TypeError('maxInFlightGpuDuties is available only with bounded-prefix completion');
  }
  const channelPlan = dutyGranularity === 'channel-range'
    ? createPostProcessorChannelDutyPlan(channelsPerDuty)
    : null;
  const manifest = dutyGranularity === 'channel-range'
    ? definePostProcessorChannelManifest(channelsPerDuty)
    : dutyGranularity === 'layer'
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
    // Only the fixed channel-range boundary carries bounded-prefix; all other
    // granularities and the default keep strict-prefix (the kit rejects
    // bounded-prefix on adaptive boundaries regardless).
    completionPolicy,
    ...(completionPolicy === 'bounded-prefix' ? { maxInFlightGpuDuties } : {}),
  });
  const output = createPostProcessorOutput(device);
  let dutyTelemetry = [];

  await execution.run(async (cooperative) => {
    if (dutyGranularity === 'channel-range') {
      const planeStates = Array(output.numPlanes).fill(null);
      const driven = await drivePostProcessorChannelBoundary(cooperative, {
        plan: channelPlan,
        encodeDuty(duty) {
          if (!planeStates[duty.plane]) {
            planeStates[duty.plane] = createPostProcessorPlaneState(
              device,
              triplanesBuf,
              weights,
              output,
              duty.plane,
            );
          }
          const encoder = device.createCommandEncoder({
            label: `post-processor-plane-${duty.plane}-${duty.stageId}`
              + (duty.kind === 'conv-range' ? `-${duty.rangeIndex}` : ''),
          });
          dispatchPostProcessorChannelDuty(
            device,
            encoder,
            planeStates[duty.plane],
            duty,
          );
          return encoder.finish();
        },
        submitDuty(commandBuffer) {
          runtime.queue.submit([commandBuffer]);
        },
      });
      dutyTelemetry = driven.telemetry;
      return;
    }
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
        channelsPerDuty: dutyGranularity === 'channel-range' ? channelsPerDuty : null,
        stageDuties: Object.freeze(dutyTelemetry),
        queueFences: Object.freeze(queueFences),
        browserYields: Object.freeze(browserYields),
      }),
    }),
  };
}
