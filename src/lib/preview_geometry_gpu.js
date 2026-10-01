import { captureGpuBufferAllocations, createStorageBuffer, readBuffer } from './gpu.js';
import { scaleTensor } from './marching_tet.js';
import { createPreviewTetGrid, extractPreviewMesh } from './preview_geometry.js';

/** Decode a final SF3D triplane on a coarse, independent tetrahedral grid. */
export async function decodePreviewMesh(device, triplanesBuf, decoder, decoderWeights, resolution = 40, planeSize = 384, options = {}) {
  const started = performance.now();
  const grid = createPreviewTetGrid(resolution);
  const positions = scaleTensor(grid.gridVertices, [0, 1], [-0.87, 0.87]);
  if (options.onSlab != null) {
    if (typeof options.onSlab !== 'function') throw new TypeError('onSlab must be a function');
    return decodeConstruction(device, triplanesBuf, decoder, decoderWeights, grid, positions, planeSize, options, started);
  }
  const { value, allocations } = captureGpuBufferAllocations(() => {
    const positionsBuf = createStorageBuffer(device, positions, 0, 'preview:positions');
    const encoder = device.createCommandEncoder();
    const decoded = decoder.decode(encoder, positionsBuf, triplanesBuf, grid.numVertices,
      decoderWeights, ['density', 'vertex_offset'], planeSize);
    device.queue.submit([encoder.finish()]);
    return decoded;
  });
  const submitted = performance.now();
  try {
    const density = await readBuffer(device, value.density, grid.numVertices * 4);
    const vertexOffsets = await readBuffer(device, value.vertex_offset, grid.numVertices * 3 * 4);
    const readback = performance.now();
    const mesh = extractPreviewMesh(grid, density, vertexOffsets);
    const finished = performance.now();
    if (!mesh.numVertices || !mesh.numFaces) throw new Error('preview mesh is empty');
    return {
      mesh,
      metrics: {
        resolution,
        planeSize,
        queryVertices: grid.numVertices,
        tetrahedra: grid.numTets,
        submitMs: submitted - started,
        readbackMs: readback - submitted,
        marchingMs: finished - readback,
        totalMs: finished - started,
        transientBufferBytes: allocations.reduce((total, item) => total + item.size, 0),
      },
    };
  } finally {
    for (const { buffer } of allocations) buffer.destroy();
  }
}

async function decodeConstruction(device, triplanesBuf, decoder, weights, grid, positions, planeSize, {
  onSlab, layersPerSlab = 4,
}, started) {
  if (!Number.isSafeInteger(layersPerSlab) || layersPerSlab < 1) throw new RangeError('layersPerSlab must be a positive integer');
  const side = grid.resolution + 1;
  const planeVertices = side ** 2;
  const density = new Float32Array(grid.numVertices);
  const offsets = new Float32Array(grid.numVertices * 3);
  let mesh;
  let submitMs = 0, readbackMs = 0, marchingMs = 0, observationMs = 0, peakBufferBytes = 0;
  for (let startLayer = 0; startLayer < side; startLayer += layersPerSlab) {
    const endLayer = Math.min(side, startLayer + layersPerSlab);
    const first = startLayer * planeVertices;
    const count = (endLayer - startLayer) * planeVertices;
    const begin = performance.now();
    const { value, allocations } = captureGpuBufferAllocations(() => {
      const input = createStorageBuffer(device, positions.subarray(first * 3, (first + count) * 3), 0, 'preview:slab');
      const encoder = device.createCommandEncoder();
      const decoded = decoder.decode(encoder, input, triplanesBuf, count, weights, ['density', 'vertex_offset'], planeSize);
      device.queue.submit([encoder.finish()]);
      return decoded;
    });
    const submitted = performance.now();
    submitMs += submitted - begin;
    peakBufferBytes = Math.max(peakBufferBytes, allocations.reduce((total, item) => total + item.size, 0));
    try {
      density.set(await readBuffer(device, value.density, count * 4), first);
      offsets.set(await readBuffer(device, value.vertex_offset, count * 12), first * 3);
    } finally {
      for (const { buffer } of allocations) buffer.destroy();
    }
    const readback = performance.now();
    readbackMs += readback - submitted;
    mesh = extractPreviewMesh(grid, density, offsets, { completedLayers: endLayer });
    const extracted = performance.now();
    marchingMs += extracted - readback;
    // Observation owns its snapshot; mutation cannot poison the completed preview.
    await onSlab({
      mesh: { ...mesh, vertices: mesh.vertices.slice(), faces: mesh.faces.slice() },
      completedLayers: endLayer, totalLayers: side,
      completedSamples: endLayer * planeVertices, totalSamples: grid.numVertices,
      maxZ: positions[((endLayer - 1) * planeVertices) * 3 + 2],
      elapsedMs: extracted - started,
    });
    observationMs += performance.now() - extracted;
  }
  if (!mesh.numVertices || !mesh.numFaces) throw new Error('preview mesh is empty');
  return { mesh, metrics: { resolution: grid.resolution, planeSize, queryVertices: grid.numVertices,
    tetrahedra: grid.numTets, submitMs, readbackMs, marchingMs, observationMs,
    totalMs: performance.now() - started, transientBufferBytes: peakBufferBytes,
    slabs: Math.ceil(side / layersPerSlab), layersPerSlab } };
}
