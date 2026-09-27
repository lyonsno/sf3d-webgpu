import { captureGpuBufferAllocations, createStorageBuffer, readBuffer } from './gpu.js';
import { scaleTensor } from './marching_tet.js';
import { createPreviewTetGrid, extractPreviewMesh } from './preview_geometry.js';

/** Decode a final SF3D triplane on a coarse, independent tetrahedral grid. */
export async function decodePreviewMesh(device, triplanesBuf, decoder, decoderWeights, resolution = 40) {
  const started = performance.now();
  const grid = createPreviewTetGrid(resolution);
  const positions = scaleTensor(grid.gridVertices, [0, 1], [-0.87, 0.87]);
  const { value, allocations } = captureGpuBufferAllocations(() => {
    const positionsBuf = createStorageBuffer(device, positions, 0, 'preview:positions');
    const encoder = device.createCommandEncoder();
    const decoded = decoder.decode(encoder, positionsBuf, triplanesBuf, grid.numVertices,
      decoderWeights, ['density', 'vertex_offset']);
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
