/**
 * marching_tet_worker.js — Web Worker running marching tetrahedra off the main
 * thread. The tet grid (535,882 vertices / 2.97M tets, ~47MB of indices) is
 * loaded ONCE by the worker at startup and stays resident, so a run only ships
 * the per-run SDF (2MB) and vertex offsets (6.4MB) in and the mesh out.
 *
 * Protocol (request/response by id, driven through worker_call.js):
 *   { id, sdf, vertexOffsets|null, bbox: [lo, hi], resolution }
 *     sdf / vertexOffsets: ArrayBuffers (transferred)
 *   → { id, ok: true, vertices, faces, numVertices, numFaces }  (transferred)
 * Uses the SAME marching_tet math and the same scaleTensor grid scaling as the
 * main-thread path, so the mesh is byte-identical.
 */
import { loadTetData, marchingTetrahedra, scaleTensor } from './marching_tet.js';

// Start loading the grid immediately; requests await it. A load failure is
// reported on every request rather than swallowed.
const tetReady = loadTetData();

self.onmessage = async (e) => {
  const { id, sdf, vertexOffsets, bbox, resolution } = e.data;
  try {
    const tet = await tetReady;
    if (!Array.isArray(bbox) || bbox.length !== 2 || !bbox.every(Number.isFinite)) {
      throw new Error('marching tet request requires a finite [lo, hi] bbox');
    }
    const gridPositions = scaleTensor(tet.gridVertices, [0, 1], bbox);
    const sdfArr = new Float32Array(sdf);
    if (sdfArr.length !== tet.numVertices) {
      throw new Error(`sdf length ${sdfArr.length} != tet grid vertices ${tet.numVertices}`);
    }
    const offsets = vertexOffsets ? new Float32Array(vertexOffsets) : null;
    if (offsets && offsets.length !== tet.numVertices * 3) {
      throw new Error(`vertexOffsets length ${offsets.length} != ${tet.numVertices * 3}`);
    }
    const mesh = marchingTetrahedra(gridPositions, sdfArr, tet.indices, offsets, resolution);
    self.postMessage({
      id, ok: true,
      vertices: mesh.vertices.buffer,
      faces: mesh.faces.buffer,
      numVertices: mesh.numVertices,
      numFaces: mesh.numFaces,
    }, [mesh.vertices.buffer, mesh.faces.buffer]);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.stack || err) });
  }
};
