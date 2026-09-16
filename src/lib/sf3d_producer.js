/**
 * SF3D producer — the device/adapter-injected callable a Kaminos host composes
 * into its own route (the kiln composition: the host owns the GPUDevice, SF3D
 * runs on it, host frames are submitted through the kit's foreground-
 * opportunity interlock before each SF3D GPU duty).
 *
 *   const producer = await createSf3dProducer({ device, adapter, weightsUrl });
 *   const handle = producer.requestForegroundOpportunity({ requestId, run(ctx) { ... ctx.submit([cb]) } });
 *   const { glb, receipt, foregroundOpportunityReport } = await producer.run(image, { runId });
 *   producer.dispose();
 *
 * Without an injected device the producer requests its own (the standalone
 * app path). The route is the product route (every proven foreground-liveness
 * mechanism on by default: cooperative GPU duties, decoder arena, five CPU
 * offload workers); routeOverrides adjust it per run.
 *
 * Honest boundaries: one run at a time per producer (the pipelines and decoder
 * arena are not reentrant); `signal` is checked at run start only (no mid-run
 * cancellation of SF3D GPU work); the route receipt's artifact hashes are
 * 'not-computed' as in the app (the witness harness hashes the GLB).
 * `dispose()` terminates producer-created workers and releases producer-loaded
 * weight buffers; it never destroys an injected device or injected weights;
 * called during a run it defers the release until the run ends and refuses
 * new runs and requests meanwhile (returns the status).
 * A failed run throws with `error.sf3dRun` = { runId, lastProgress,
 * foregroundOpportunityReport, identity } so the host keeps the phase and the
 * last trustworthy evidence.
 */
import { initGPU } from './gpu.js';
import { loadWeights } from './weights.js';
import { initPipelines } from './inference.js';
import { runFullPipelineToGlb } from './full_pipeline.js';
import {
  createProductRouteOptions,
  createProductRouteWorkers,
  describeProductRouteOptions,
  productRouteWorkerModuleUrls,
  terminateProductRouteWorkers,
} from './product_route.js';
import { createForegroundOpportunityBridge } from './foreground_opportunity_bridge.js';
import { createProducerLifecycle } from './producer_lifecycle.js';
import {
  createSf3dImageToMeshRouteReceipt,
  createStagedSubmitProfile,
  addStagedSubmitStage,
  createWebGpuBackendIdentity,
  SF3D_IMAGE_TO_MESH_ROUTE_ID,
  WEBGPU_INFERENCE_KIT_VERSION,
  validateRouteReceipt,
} from '@kaminos/webgpu-inference-kit';

export const SF3D_PRODUCER_SCHEMA = 'sf3d.producer.v0';
export const SF3D_PRODUCER_RUN_IDENTITY_SCHEMA = 'sf3d.producer-run-identity.v0';

/** Resolve the weights URL the way loadWeights will fetch it (absolute when a document location exists). */
export function resolveWeightsUrl(weightsUrl, base = globalThis.location?.href ?? null) {
  if (typeof weightsUrl !== 'string' || !weightsUrl) throw new Error('weightsUrl must be a non-empty string');
  try { return new URL(weightsUrl, base ?? undefined).href; } catch { return weightsUrl; }
}

/** Producer-created GPU weight buffers are released on dispose; injected weights and the borrowed device never are. */
export function releaseLoadedWeights(weights) {
  let released = 0;
  for (const value of Object.values(weights || {})) {
    if (value && typeof value.destroy === 'function') { value.destroy(); released += 1; }
  }
  return released;
}

/** Run/clock identity bound to a producer run (Wake answer 5: run/clock identity + effective topology). */
export function buildRunIdentity({ runId, routeId, startedAtMs, finishedAtMs, deviceInjected, commit, kitVersion }) {
  return Object.freeze({
    schema: SF3D_PRODUCER_RUN_IDENTITY_SCHEMA,
    runId,
    routeId,
    timeOrigin: globalThis.performance?.timeOrigin ?? null,
    clock: 'performance.now',
    startedAtMs,
    finishedAtMs,
    durationMs: finishedAtMs - startedAtMs,
    deviceTopology: deviceInjected ? 'host-injected-device' : 'producer-owned-device',
    producerCommit: commit,
    kitVersion,
  });
}
export const SF3D_REQUIRED_RECEIPT_STAGES = Object.freeze([
  'image-preprocess', 'dinov2-tokenizer', 'two-stream-backbone',
  'triplane-decode', 'marching-tet', 'texture-bake', 'glb-export',
]);
const TEX_RESOLUTION = 1024;

async function describeBackend(adapter, device) {
  let info = adapter?.info ?? device.adapterInfo ?? null;
  if (!info && adapter?.requestAdapterInfo) info = await adapter.requestAdapterInfo();
  info = info || {};
  const limits = adapter?.limits ?? device.limits;
  const features = [...(adapter?.features ?? device.features ?? [])];
  const identity = createWebGpuBackendIdentity({
    adapterName: info.description || info.device || 'unknown',
    browser: globalThis.navigator?.userAgent ?? 'unknown',
    requestedFeatures: features,
    effectiveFeatures: features,
    limits: {
      maxBufferSize: limits.maxBufferSize,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
    },
    timestampQuery: features.includes('timestamp-query') ? 'available' : 'unavailable',
  });
  return { identity, info, limits, features };
}

function buildRouteReceipt({ backend, image, result, commit }) {
  const profile = createStagedSubmitProfile({
    route: SF3D_IMAGE_TO_MESH_ROUTE_ID,
    timingSource: 'performance-now-wall-clock',
    requiredStages: [...SF3D_REQUIRED_RECEIPT_STAGES],
  });
  const stageTimings = result.stageTimings || {};
  for (const name of ['image-preprocess', 'dinov2-tokenizer', 'two-stream-backbone', 'triplane-decode', 'marching-tet']) {
    addStagedSubmitStage(profile, { name, ms: stageTimings[name] || 0 });
  }
  const spanMs = (name) => {
    const span = (result.stageSpans || []).find(s => s.name === name);
    return span ? span.end - span.start : 0;
  };
  addStagedSubmitStage(profile, { name: 'texture-bake', ms: spanMs('texture-bake') });
  addStagedSubmitStage(profile, { name: 'glb-export', ms: spanMs('glb-export') });

  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  return createSf3dImageToMeshRouteReceipt({
    input: { artifactId: `source-image:${width}x${height}`, sha256: 'not-computed', shape: [height, width, 4] },
    outputs: {
      meshGlb: { artifactId: `mesh-glb:${result.numVertices}v-${result.numFaces}f`, sha256: 'not-computed', shape: [result.glb.byteLength] },
      albedoTexture: { artifactId: `albedo-texture:${TEX_RESOLUTION}`, sha256: 'not-computed', shape: [TEX_RESOLUTION, TEX_RESOLUTION, 4] },
      normalMap: { artifactId: `normal-map:${TEX_RESOLUTION}`, sha256: 'not-computed', shape: [TEX_RESOLUTION, TEX_RESOLUTION, 4] },
    },
    backend: backend.identity,
    model: { revision: 'v1.0.0-webgpu', weightsHash: 'not-computed' },
    kernel: {
      kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
      profile: 'dinov2-two-stream-triplane-marching-tet-texture-bake',
      commit,
    },
    profile,
  });
}

export async function createSf3dProducer({
  device = null,
  adapter = null,
  weights = null,
  weightsUrl = 'weights.bin',
  workers = null,
  onWeightsProgress = null,
  commit = (typeof __COMMIT_HASH__ !== 'undefined' ? __COMMIT_HASH__ : 'dev'),
} = {}) {
  const gpu = await initGPU(device ? { device, adapter } : {});
  const dev = gpu.device;
  const backend = await describeBackend(gpu.adapter, dev);
  const ownsWeights = weights == null;
  const modelWeights = weights ?? await loadWeights(dev, weightsUrl, onWeightsProgress || undefined);
  // Explicit resource identity for a mounting host: where the weights came
  // from and which worker module URLs must be reachable from the artifact.
  const resources = Object.freeze({
    weightsSource: ownsWeights ? 'loaded-by-producer' : 'injected-by-host',
    weightsUrl: ownsWeights ? resolveWeightsUrl(weightsUrl) : null,
    weightsSha256: 'not-computed',
    workerModuleUrls: productRouteWorkerModuleUrls(),
    workersSource: workers == null ? 'created-by-producer' : 'injected-by-host',
  });
  const pipelines = initPipelines(dev);
  const ownsWorkers = workers == null;
  const routeWorkers = workers ?? createProductRouteWorkers();
  const bridge = createForegroundOpportunityBridge({ routeId: SF3D_IMAGE_TO_MESH_ROUTE_ID, device: dev, queue: dev.queue });

  let runSequence = 0;
  // One run at a time; dispose() during a run defers the release until the
  // run ends and refuses everything new meanwhile (producer_lifecycle.js).
  const lifecycle = createProducerLifecycle({
    release() {
      if (ownsWorkers) terminateProductRouteWorkers(routeWorkers);
      // Release what the producer created; never destroy an injected weight set
      // or the (possibly borrowed) device.
      if (ownsWeights) releaseLoadedWeights(modelWeights);
    },
  });

  return Object.freeze({
    schema: SF3D_PRODUCER_SCHEMA,
    routeId: SF3D_IMAGE_TO_MESH_ROUTE_ID,
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    device: dev,
    adapter: gpu.adapter,
    deviceInjected: gpu.injected,
    weights: modelWeights,
    pipelines,
    workers: routeWorkers,
    backend: backend.identity,
    resources,
    adapterInfo: backend.info,
    adapterLimits: backend.limits,
    adapterFeatures: Object.freeze([...backend.features]),
    get activeRunId() { return lifecycle.activeRunId; },
    get disposed() { return lifecycle.disposed; },

    /** Host (kiln) frames: kit-shaped { requestId, run(ctx), metadata } → { requestId, completion, cancel }. */
    requestForegroundOpportunity(request) {
      lifecycle.assertAcceptingRequests();
      return bridge.request(request);
    },
    foregroundSnapshot() { return bridge.snapshot(); },

    async run(image, { runId = null, onProgress = null, routeOverrides = {}, signal = null } = {}) {
      if (image == null) throw new Error('sf3d run requires an image (HTMLImageElement/ImageBitmap-like)');
      if (signal?.aborted) throw new Error('sf3d run aborted before start');
      runSequence += 1;
      const id = runId ?? `sf3d-run-${runSequence}`;
      lifecycle.beginRun(id);                       // refuses when disposed or a run is active
      const foregroundRun = bridge.beginRun(id);
      const options = Object.freeze({
        ...createProductRouteOptions({ workers: routeWorkers, overrides: routeOverrides }),
        foregroundOpportunities: foregroundRun.foregroundOpportunities,
      });
      let result;
      let foregroundOpportunityReport;
      let lastProgress = null;
      const startedAtMs = performance.now();
      const progress = (message) => { lastProgress = String(message); if (onProgress) onProgress(message); };
      try {
        result = await runFullPipelineToGlb(dev, pipelines, modelWeights, image, options, progress);
      } catch (error) {
        // Preserve the phase and the last trustworthy evidence on the error
        // (Wake answer 5): the host keeps it with its own episode receipts.
        foregroundOpportunityReport = await foregroundRun.finish();
        lifecycle.endRun(id);                       // runs a deferred dispose if one was requested
        try {
          error.sf3dRun = Object.freeze({
            runId: id,
            lastProgress,
            foregroundOpportunityReport,
            identity: buildRunIdentity({ runId: id, routeId: SF3D_IMAGE_TO_MESH_ROUTE_ID, startedAtMs, finishedAtMs: performance.now(), deviceInjected: gpu.injected, commit, kitVersion: WEBGPU_INFERENCE_KIT_VERSION }),
          });
        } catch { /* error object not extensible; the throw still carries the message */ }
        throw error;
      }
      foregroundOpportunityReport = await foregroundRun.finish();
      lifecycle.endRun(id);                         // runs a deferred dispose if one was requested
      const finishedAtMs = performance.now();
      const receipt = buildRouteReceipt({ backend, image, result, commit });
      const receiptValidation = validateRouteReceipt(receipt);
      return Object.freeze({
        schema: 'sf3d.producer-run-result.v0',
        runId: id,
        identity: buildRunIdentity({ runId: id, routeId: SF3D_IMAGE_TO_MESH_ROUTE_ID, startedAtMs, finishedAtMs, deviceInjected: gpu.injected, commit, kitVersion: WEBGPU_INFERENCE_KIT_VERSION }),
        resources,
        glb: result.glb,
        receipt,
        receiptValidation,
        routeOptions: describeProductRouteOptions(options),
        offloads: result.offloads,
        cooperativeReports: result.cooperativeReports,
        stageSpans: result.stageSpans,
        stageTimings: result.stageTimings,
        totalMs: result.totalMs,
        foregroundOpportunityReport,
        numVertices: result.numVertices,
        numFaces: result.numFaces,
        vertices: result.vertices,
        faces: result.faces,
        roughness: result.roughness,
        metallic: result.metallic,
        sdf: result.sdf,
        isosurfaceThreshold: result.isosurfaceThreshold,
        arenaSnapshot: result.arenaSnapshot,
        dinoPayload: result.dinoPayload,
      });
    },

    /**
     * Safe at any time. With no run active: releases producer-created workers
     * and weight buffers now. During a run: refuses new runs/requests at once
     * and releases when the run ends (the run's own finish and receipts still
     * happen). Never destroys an injected device or injected weights.
     * Returns { status: 'released' | 'deferred-until-run-ends' | 'already-disposed' }.
     */
    dispose() {
      return lifecycle.dispose();
    },
  });
}
