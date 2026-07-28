/**
 * Cooperative execution for the SF3D two-stream interleave transformer.
 *
 * The graph is cut only at dependency-safe stage boundaries. Legacy forward()
 * drives these same stages into one command encoder, so scheduling changes do
 * not create a second numerical implementation.
 */

import {
  createWebGpuCooperativeExecution,
  defineWebGpuCooperativeBoundaryManifest,
} from '@kaminos/webgpu-inference-kit';
import { createSf3dCooperativeRuntime, SF3D_ROUTE_ID } from './cooperative_dino.js';
import {
  TWO_STREAM_STAGE_IDS,
  createTwoStreamAttentionDutyPlan,
} from './two_stream.js';

export const TWO_STREAM_MANIFEST_ID = 'sf3d.two-stream-cooperative-boundaries.v0';
export const TWO_STREAM_BOUNDARY_ID = 'two-stream-stages';
export const TWO_STREAM_ATTENTION_MANIFEST_ID =
  'sf3d.two-stream-attention-cooperative-boundaries.v0';
export const TWO_STREAM_ATTENTION_BOUNDARY_ID = 'two-stream-attention-duties';
export const TWO_STREAM_DUTY_COUNT = TWO_STREAM_STAGE_IDS.length;
export { TWO_STREAM_STAGE_IDS };

export function defineTwoStreamManifest(options = {}) {
  const {
    dutyGranularity = 'stage',
    N_img,
  } = options;
  if (!['stage', 'attention-tile'].includes(dutyGranularity)) {
    throw new RangeError(`unknown two-stream duty granularity ${dutyGranularity}`);
  }
  const attentionPlan = dutyGranularity === 'attention-tile'
    ? createTwoStreamAttentionDutyPlan(N_img)
    : null;
  const boundaryId = attentionPlan ? TWO_STREAM_ATTENTION_BOUNDARY_ID : TWO_STREAM_BOUNDARY_ID;
  const totalItems = attentionPlan?.length ?? TWO_STREAM_DUTY_COUNT;
  return defineWebGpuCooperativeBoundaryManifest({
    manifestId: attentionPlan ? TWO_STREAM_ATTENTION_MANIFEST_ID : TWO_STREAM_MANIFEST_ID,
    routeId: SF3D_ROUTE_ID,
    phases: [
      {
        phaseId: 'two-stream-backbone',
        boundaries: [
          {
            boundaryId,
            kind: 'gpu-command',
            unit: attentionPlan ? 'two-stream-attention-duty' : 'two-stream-stage',
            totalItems,
            progressWeight: totalItems,
            commandDutyKind: 'compute',
            chunking: { mode: 'fixed', chunkItems: 1 },
            yieldPolicy: 'after-duty',
            resources: {
              retain: [
                'dinov2.tokens',
                'triplane.low-resolution',
                'two-stream.weights',
                'two-stream.intermediates',
              ],
              produce: ['two-stream.triplane-features'],
              release: [],
            },
          },
        ],
      },
    ],
    metadata: {
      source: 'sf3d-webgpu-cooperative-two-stream',
      dutyGranularity,
    },
  });
}

export async function driveTwoStreamBoundary(cooperative, options) {
  const {
    encodeStage,
    submitStage,
    now = () => globalThis.performance?.now?.() ?? Date.now(),
  } = options;
  const gpu = cooperative.startBoundary(TWO_STREAM_BOUNDARY_ID);
  const telemetry = [];

  for (let stageIndex = 0; stageIndex < TWO_STREAM_DUTY_COUNT; stageIndex++) {
    const range = gpu.nextRange();
    if (!range) {
      throw new Error(`cooperative two-stream exhausted ranges before stage ${stageIndex}`);
    }
    if (range.itemStart !== stageIndex || range.itemEnd !== stageIndex + 1) {
      throw new Error(
        `cooperative two-stream range ${range.itemStart}-${range.itemEnd} `
        + `does not match stage ${stageIndex}`,
      );
    }

    const stageId = TWO_STREAM_STAGE_IDS[stageIndex];
    const timing = {
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
      encode() {
        timing.encodeStartedAtMs = now();
        const commandBuffer = encodeStage({ stageIndex, stageId, range });
        timing.encodeCompletedAtMs = now();
        timing.encodeMs = timing.encodeCompletedAtMs - timing.encodeStartedAtMs;
        return commandBuffer;
      },
      submit(commandBuffer) {
        timing.submitStartedAtMs = now();
        const result = submitStage(commandBuffer, { stageIndex, stageId, range });
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
    throw new Error('cooperative two-stream left ranges unconsumed');
  }
  return {
    completedDuties: telemetry.length,
    totalDuties: TWO_STREAM_DUTY_COUNT,
    telemetry: Object.freeze(telemetry),
  };
}

export async function driveTwoStreamAttentionBoundary(cooperative, options) {
  const {
    plan,
    encodeDuty,
    submitDuty,
    now = () => globalThis.performance?.now?.() ?? Date.now(),
  } = options;
  const gpu = cooperative.startBoundary(TWO_STREAM_ATTENTION_BOUNDARY_ID);
  const telemetry = [];

  for (const duty of plan) {
    const range = gpu.nextRange();
    if (!range) {
      throw new Error(
        `cooperative two-stream attention exhausted ranges before duty ${duty.dutyIndex}`,
      );
    }
    if (range.itemStart !== duty.dutyIndex || range.itemEnd !== duty.dutyIndex + 1) {
      throw new Error(
        `cooperative two-stream attention range ${range.itemStart}-${range.itemEnd} `
        + `does not match duty ${duty.dutyIndex}`,
      );
    }
    const timing = {
      dutyIndex: duty.dutyIndex,
      dutyId: duty.dutyId,
      kind: duty.kind,
      ownerId: duty.ownerId ?? null,
      tileIndex: duty.tileIndex ?? null,
      tileCount: duty.tileCount ?? null,
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
      encode() {
        timing.encodeStartedAtMs = now();
        const commandBuffer = encodeDuty({ duty, range });
        timing.encodeCompletedAtMs = now();
        timing.encodeMs = timing.encodeCompletedAtMs - timing.encodeStartedAtMs;
        return commandBuffer;
      },
      submit(commandBuffer) {
        timing.submitStartedAtMs = now();
        const result = submitDuty(commandBuffer, { duty, range });
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
    throw new Error('cooperative two-stream attention left ranges unconsumed');
  }
  return {
    completedDuties: telemetry.length,
    totalDuties: plan.length,
    telemetry: Object.freeze(telemetry),
  };
}

export async function runCooperativeTwoStream(options) {
  const {
    device,
    backbone,
    imageTokensBuf,
    N_img,
    weights,
    schedulingMode = 'cooperative',
    dutyGranularity = 'stage',
    onProgress,
    signal,
    invocationId = `sf3d:two-stream:${schedulingMode}`,
  } = options;
  if (!['stage', 'attention-tile'].includes(dutyGranularity)) {
    throw new RangeError(`unknown two-stream duty granularity ${dutyGranularity}`);
  }
  const attentionPlan = dutyGranularity === 'attention-tile'
    ? createTwoStreamAttentionDutyPlan(N_img)
    : null;
  const now = () => globalThis.performance?.now?.() ?? Date.now();
  const queueFences = [];
  const browserYields = [];
  let activeStage = null;
  const runtime = createSf3dCooperativeRuntime(device, {
    onQueueFenceResolved(event) {
      queueFences.push(Object.freeze({ ...activeStage, ...event }));
    },
    onBrowserYield(event) {
      browserYields.push(Object.freeze({ ...activeStage, ...event }));
    },
  });
  const execution = createWebGpuCooperativeExecution({
    runtime,
    manifest: defineTwoStreamManifest({ dutyGranularity, N_img }),
    invocationId,
    schedulingMode,
    onProgress,
    signal,
  });
  const state = attentionPlan
    ? backbone.createAttentionForwardState(imageTokensBuf, N_img, weights)
    : backbone.createForwardState(imageTokensBuf, N_img, weights);
  let stageTelemetry = [];

  await execution.run(async cooperative => {
    const driven = attentionPlan
      ? await driveTwoStreamAttentionBoundary(cooperative, {
        plan: attentionPlan,
        now,
        encodeDuty({ duty }) {
          activeStage = {
            dutyIndex: duty.dutyIndex,
            dutyId: duty.dutyId,
            kind: duty.kind,
            ownerId: duty.ownerId ?? null,
            tileIndex: duty.tileIndex ?? null,
            tileCount: duty.tileCount ?? null,
          };
          const encoder = device.createCommandEncoder({
            label: `two-stream-${duty.dutyIndex}-${duty.dutyId}`,
          });
          backbone.dispatchAttentionForwardDuty(encoder, state, duty.dutyIndex);
          return encoder.finish();
        },
        submitDuty(commandBuffer) {
          runtime.queue.submit([commandBuffer]);
        },
      })
      : await driveTwoStreamBoundary(cooperative, {
        now,
        encodeStage({ stageIndex, stageId }) {
          activeStage = { stageIndex, stageId };
          const encoder = device.createCommandEncoder({
            label: `two-stream-${stageIndex}-${stageId}`,
          });
          backbone.dispatchForwardStage(encoder, state, stageIndex);
          return encoder.finish();
        },
        submitStage(commandBuffer) {
          runtime.queue.submit([commandBuffer]);
        },
      });
    stageTelemetry = driven.telemetry;
  });

  activeStage = null;
  const result = backbone.getForwardResult(state);
  return {
    result,
    report: Object.freeze({
      ...execution.finish(),
      adapterTelemetry: Object.freeze({
        dutyGranularity,
        declaredDutyCount: attentionPlan?.length ?? TWO_STREAM_DUTY_COUNT,
        stageDuties: Object.freeze(stageTelemetry),
        queueFences: Object.freeze(queueFences),
        browserYields: Object.freeze(browserYields),
      }),
    }),
  };
}
