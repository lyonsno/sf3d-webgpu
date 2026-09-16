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
 */
import { initGPU } from './gpu.js';
import { loadWeights } from './weights.js';
import { initPipelines } from './inference.js';
import { runFullPipelineToGlb } from './full_pipeline.js';
import {
  createProductRouteOptions,
  createProductRouteWorkers,
  describeProductRouteOptions,
  terminateProductRouteWorkers,
} from './product_route.js';
import { createForegroundOpportunityBridge } from './foreground_opportunity_bridge.js';
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
  const modelWeights = weights ?? await loadWeights(dev, weightsUrl, onWeightsProgress || undefined);
  const pipelines = initPipelines(dev);
  const ownsWorkers = workers == null;
  const routeWorkers = workers ?? createProductRouteWorkers();
  const bridge = createForegroundOpportunityBridge({ routeId: SF3D_IMAGE_TO_MESH_ROUTE_ID, device: dev, queue: dev.queue });

  let runSequence = 0;
  let activeRunId = null;
  let disposed = false;
  const assertLive = () => { if (disposed) throw new Error('sf3d producer is disposed'); };

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
    adapterInfo: backend.info,
    adapterLimits: backend.limits,
    adapterFeatures: Object.freeze([...backend.features]),
    get activeRunId() { return activeRunId; },

    /** Host (kiln) frames: kit-shaped { requestId, run(ctx), metadata } → { requestId, completion, cancel }. */
    requestForegroundOpportunity(request) {
      assertLive();
      return bridge.request(request);
    },
    foregroundSnapshot() { return bridge.snapshot(); },

    async run(image, { runId = null, onProgress = null, routeOverrides = {}, signal = null } = {}) {
      assertLive();
      if (image == null) throw new Error('sf3d run requires an image (HTMLImageElement/ImageBitmap-like)');
      if (signal?.aborted) throw new Error('sf3d run aborted before start');
      runSequence += 1;
      const id = runId ?? `sf3d-run-${runSequence}`;
      const foregroundRun = bridge.beginRun(id);   // refuses a second concurrent run
      activeRunId = id;
      const options = Object.freeze({
        ...createProductRouteOptions({ workers: routeWorkers, overrides: routeOverrides }),
        foregroundOpportunities: foregroundRun.foregroundOpportunities,
      });
      let result;
      let foregroundOpportunityReport;
      try {
        result = await runFullPipelineToGlb(dev, pipelines, modelWeights, image, options, onProgress || undefined);
      } finally {
        foregroundOpportunityReport = await foregroundRun.finish();
        activeRunId = null;
      }
      const receipt = buildRouteReceipt({ backend, image, result, commit });
      const receiptValidation = validateRouteReceipt(receipt);
      return Object.freeze({
        schema: 'sf3d.producer-run-result.v0',
        runId: id,
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

    dispose() {
      if (disposed) return;
      disposed = true;
      if (ownsWorkers) terminateProductRouteWorkers(routeWorkers);
    },
  });
}
