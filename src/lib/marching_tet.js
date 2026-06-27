/**
 * marching_tet.js — CPU-side marching tetrahedra mesh extraction.
 *
 * Port of sf3d/models/isosurface.py MarchingTetrahedraHelper._forward.
 *
 * Input: SDF values at grid vertices + vertex offsets
 * Output: { vertices: Float32Array([x,y,z,...]), faces: Uint32Array([i,j,k,...]) }
 */

// Lookup tables (matching the buffers in system.py)
const TRIANGLE_TABLE = [
  [-1, -1, -1, -1, -1, -1],
  [1, 0, 2, -1, -1, -1],
  [4, 0, 3, -1, -1, -1],
  [1, 4, 2, 1, 3, 4],
  [3, 1, 5, -1, -1, -1],
  [2, 3, 0, 2, 5, 3],
  [1, 4, 0, 1, 5, 4],
  [4, 2, 5, -1, -1, -1],
  [4, 5, 2, -1, -1, -1],
  [4, 1, 0, 4, 5, 1],
  [3, 2, 0, 3, 5, 2],
  [1, 3, 5, -1, -1, -1],
  [4, 1, 2, 4, 3, 1],
  [3, 0, 4, -1, -1, -1],
  [2, 0, 1, -1, -1, -1],
  [-1, -1, -1, -1, -1, -1],
];

const NUM_TRIANGLES_TABLE = [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0];

const BASE_TET_EDGES = [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3];

/**
 * Load tetrahedra grid data from binary files.
 *
 * @param {string} basePath - path to tets directory (e.g., 'tets/')
 * @returns {Object} - { gridVertices, indices }
 */
export async function loadTetData(basePath = 'tets/') {
  const [vertsBuf, indicesBuf] = await Promise.all([
    fetch(`${basePath}_grid_vertices.bin`).then(r => r.arrayBuffer()),
    fetch(`${basePath}indices.bin`).then(r => r.arrayBuffer()),
  ]);

  const gridVertices = new Float32Array(vertsBuf);  // [N_v, 3]
  const indices = new Int32Array(indicesBuf);        // [N_t, 4]

  return {
    gridVertices,
    numVertices: gridVertices.length / 3,
    indices,
    numTets: indices.length / 4,
  };
}

/**
 * Run marching tetrahedra to extract a mesh from SDF values.
 *
 * @param {Float32Array} gridVertices - [N_v, 3] grid vertex positions
 * @param {Float32Array} sdf - [N_v] signed distance values (positive = inside)
 * @param {Int32Array} tetIndices - [N_t, 4] tetrahedra vertex indices
 * @param {Float32Array|null} vertexOffsets - [N_v, 3] optional vertex deformations
 * @param {number} resolution - grid resolution (for normalizing deformation)
 * @returns {{ vertices: Float32Array, faces: Uint32Array, numVertices: number, numFaces: number }}
 */
export function marchingTetrahedra(gridVertices, sdf, tetIndices, vertexOffsets = null, resolution = 160) {
  const N_v = gridVertices.length / 3;
  const N_t = tetIndices.length / 4;

  // Apply vertex deformation if provided
  let positions;
  if (vertexOffsets) {
    // normalize_grid_deformation: (1-0)/resolution * tanh(offsets)
    const scale = 1.0 / resolution;
    positions = new Float32Array(N_v * 3);
    for (let i = 0; i < N_v * 3; i++) {
      positions[i] = gridVertices[i] + scale * Math.tanh(vertexOffsets[i]);
    }
  } else {
    positions = gridVertices;
  }

  // Determine occupancy: sdf > 0 means inside
  const occ = new Uint8Array(N_v);
  for (let i = 0; i < N_v; i++) {
    occ[i] = sdf[i] > 0 ? 1 : 0;
  }

  // Find valid tetrahedra (partially occupied: 0 < occ_sum < 4)
  const validTets = [];
  for (let t = 0; t < N_t; t++) {
    const base = t * 4;
    const sum = occ[tetIndices[base]] + occ[tetIndices[base + 1]] +
                occ[tetIndices[base + 2]] + occ[tetIndices[base + 3]];
    if (sum > 0 && sum < 4) {
      validTets.push(t);
    }
  }

  // Collect all edges from valid tetrahedra
  // Each tet has 6 edges (from BASE_TET_EDGES: pairs of vertex indices within tet)
  const edgeMap = new Map(); // "v0,v1" → edge index
  const edgeList = [];       // [[v0, v1], ...]
  const tetEdgeIndices = new Int32Array(validTets.length * 6); // per-valid-tet edge mapping

  for (let vi = 0; vi < validTets.length; vi++) {
    const t = validTets[vi];
    const tetBase = t * 4;
    for (let e = 0; e < 6; e++) {
      let v0 = tetIndices[tetBase + BASE_TET_EDGES[e * 2]];
      let v1 = tetIndices[tetBase + BASE_TET_EDGES[e * 2 + 1]];
      // Sort edge vertices
      if (v0 > v1) { const tmp = v0; v0 = v1; v1 = tmp; }
      const key = `${v0},${v1}`;
      let edgeIdx;
      if (edgeMap.has(key)) {
        edgeIdx = edgeMap.get(key);
      } else {
        edgeIdx = edgeList.length;
        edgeMap.set(key, edgeIdx);
        edgeList.push([v0, v1]);
      }
      tetEdgeIndices[vi * 6 + e] = edgeIdx;
    }
  }

  // Find edges that cross the isosurface (one vertex inside, one outside)
  const crossingEdges = [];
  const edgeToVertex = new Int32Array(edgeList.length).fill(-1);
  let vertexCount = 0;

  for (let i = 0; i < edgeList.length; i++) {
    const [v0, v1] = edgeList[i];
    if (occ[v0] !== occ[v1]) {
      edgeToVertex[i] = vertexCount++;
      crossingEdges.push(i);
    }
  }

  // Interpolate vertex positions along crossing edges
  const vertices = new Float32Array(vertexCount * 3);
  for (const edgeIdx of crossingEdges) {
    const [v0, v1] = edgeList[edgeIdx];
    const s0 = sdf[v0];
    const s1 = sdf[v1];
    // Linear interpolation: find zero crossing
    // s0 + t*(s1-s0) = 0 → t = -s0/(s1-s0) = s0/(s0-s1)
    const denom = s0 - s1;
    const t = denom !== 0 ? s0 / denom : 0.5;

    const outIdx = edgeToVertex[edgeIdx] * 3;
    for (let d = 0; d < 3; d++) {
      vertices[outIdx + d] = positions[v0 * 3 + d] * (1 - t) + positions[v1 * 3 + d] * t;
    }
  }

  // Generate triangle faces using lookup table
  const faceList = [];
  for (let vi = 0; vi < validTets.length; vi++) {
    const t = validTets[vi];
    const tetBase = t * 4;

    // Compute tet index for lookup table
    let tetindex = 0;
    for (let j = 0; j < 4; j++) {
      if (occ[tetIndices[tetBase + j]]) {
        tetindex |= (1 << j);
      }
    }

    const numTri = NUM_TRIANGLES_TABLE[tetindex];
    const triRow = TRIANGLE_TABLE[tetindex];

    for (let tri = 0; tri < numTri; tri++) {
      const i0 = edgeToVertex[tetEdgeIndices[vi * 6 + triRow[tri * 3]]];
      const i1 = edgeToVertex[tetEdgeIndices[vi * 6 + triRow[tri * 3 + 1]]];
      const i2 = edgeToVertex[tetEdgeIndices[vi * 6 + triRow[tri * 3 + 2]]];

      if (i0 >= 0 && i1 >= 0 && i2 >= 0) {
        faceList.push(i0, i1, i2);
      }
    }
  }

  const faces = new Uint32Array(faceList);

  return {
    vertices,
    faces,
    numVertices: vertexCount,
    numFaces: faces.length / 3,
  };
}

/**
 * Scale tensor from one range to another.
 * Matches sf3d.models.utils.scale_tensor.
 */
export function scaleTensor(data, fromRange, toRange) {
  const [fromMin, fromMax] = fromRange;
  const [toMin, toMax] = toRange;
  const scale = (toRange[1] - toRange[0]) / (fromMax - fromMin);
  const offset = toRange[0] - fromMin * scale;
  const result = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = data[i] * scale + offset;
  }
  return result;
}
