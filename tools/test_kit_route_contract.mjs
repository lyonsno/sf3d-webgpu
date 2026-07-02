import assert from 'node:assert/strict';
import {
  addStagedSubmitStage,
  createSf3dImageToMeshRouteDefinition,
  createSf3dImageToMeshRouteReceipt,
  createStagedSubmitProfile,
  createWebGpuBackendIdentity,
  SF3D_IMAGE_TO_MESH_ROUTE_ID,
  WEBGPU_INFERENCE_KIT_VERSION,
  WEBGPU_ROUTE_BACKPRESSURE_SCHEMA,
  WEBGPU_ROUTE_SCHEDULER_SCHEMA,
  validateRouteReceipt,
} from '@kaminos/webgpu-inference-kit';

const [kitMajor, kitMinor, kitPatch] = WEBGPU_INFERENCE_KIT_VERSION.split('.').map(Number);
assert.deepEqual([kitMajor, kitMinor], [0, 1]);
assert.ok(kitPatch >= 4, `breathability contract requires kit >=0.1.4, got ${WEBGPU_INFERENCE_KIT_VERSION}`);

const requiredStages = [
  'image-preprocess',
  'dinov2-tokenizer',
  'two-stream-backbone',
  'triplane-decode',
  'marching-tet',
  'texture-bake',
  'glb-export',
];

const definition = createSf3dImageToMeshRouteDefinition({
  kernel: {
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    profile: 'dinov2-two-stream-triplane-marching-tet-texture-bake',
    commit: 'sf3d-webgpu-kit-contract-smoke',
  },
});

assert.equal(SF3D_IMAGE_TO_MESH_ROUTE_ID, 'sf3d.image-to-mesh.webgpu-local.v0');
assert.equal(definition.routeId, SF3D_IMAGE_TO_MESH_ROUTE_ID);
assert.deepEqual(definition.requiredStages, requiredStages);
assert.equal(definition.scheduler.schema, WEBGPU_ROUTE_SCHEDULER_SCHEMA);
assert.equal(definition.scheduler.requestedScheduler.mode, 'throughput');
assert.equal(definition.backpressure.schema, WEBGPU_ROUTE_BACKPRESSURE_SCHEMA);
assert.equal(definition.backpressure.effectiveBudget, 'furnace');
assert.deepEqual(
  definition.scheduler.breathability.spans.map(span => [span.stage, span.kind, span.interruptible]),
  [
    ['image-preprocess', 'gpu-submit-bound', false],
    ['dinov2-tokenizer', 'gpu-submit-bound', false],
    ['two-stream-backbone', 'gpu-submit-bound', false],
    ['triplane-decode', 'gpu-submit-bound', false],
    ['marching-tet', 'gpu-submit-bound', false],
    ['texture-bake', 'gpu-submit-bound', false],
    ['glb-export', 'cpu-bound', false],
  ],
);
assert.deepEqual(
  definition.scheduler.breathability.checkpoints.map(checkpoint => [checkpoint.afterStage, checkpoint.yieldable]),
  requiredStages.map(stage => [stage, true]),
);
assert.deepEqual(
  definition.outputRoles.filter(output => output.required).map(output => output.role),
  ['mesh-glb', 'albedo-texture', 'normal-map'],
);

const backend = createWebGpuBackendIdentity({
  adapterName: 'contract-test-webgpu-adapter',
  browser: 'node-contract-smoke',
  requestedFeatures: ['timestamp-query'],
  effectiveFeatures: ['timestamp-query'],
  limits: {
    maxBufferSize: 1024,
    maxStorageBufferBindingSize: 1024,
    maxComputeInvocationsPerWorkgroup: 256,
  },
  timestampQuery: 'requested',
});

const profile = createStagedSubmitProfile({
  route: SF3D_IMAGE_TO_MESH_ROUTE_ID,
  timingSource: 'adapter-phase-wall-clock',
  requiredStages,
});
for (const [index, name] of requiredStages.entries()) {
  addStagedSubmitStage(profile, { name, ms: index + 1 });
}

const receipt = createSf3dImageToMeshRouteReceipt({
  input: {
    artifactId: 'source-image:test',
    sha256: 'sha256-source-image',
    shape: [1024, 1024, 4],
  },
  outputs: {
    meshGlb: {
      artifactId: 'mesh-glb:test',
      sha256: 'sha256-glb',
      shape: [1],
    },
    albedoTexture: {
      artifactId: 'albedo-texture:test',
      sha256: 'sha256-albedo',
      shape: [1024, 1024, 4],
    },
    normalMap: {
      artifactId: 'normal-map:test',
      sha256: 'sha256-normal',
      shape: [1024, 1024, 4],
    },
    meshObj: {
      artifactId: 'mesh-obj:test',
      sha256: 'sha256-obj',
      shape: [1],
    },
  },
  backend,
  model: {
    revision: 'stable-fast-3d',
    weightsHash: 'sha256-weights',
  },
  kernel: {
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    profile: 'dinov2-two-stream-triplane-marching-tet-texture-bake',
    commit: 'sf3d-webgpu-kit-contract-smoke',
  },
  profile,
});

const result = validateRouteReceipt(receipt);
assert.equal(result.ok, true, result.errors.join('; '));
assert.equal(receipt.requestedRouteId, SF3D_IMAGE_TO_MESH_ROUTE_ID);
assert.deepEqual(receipt.outputs.map(output => output.role), [
  'mesh-glb',
  'albedo-texture',
  'normal-map',
  'mesh-obj',
]);

console.log('SF3D kit route contract passed');
