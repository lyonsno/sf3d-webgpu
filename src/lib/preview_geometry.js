import { marchingTetrahedra, scaleTensor } from './marching_tet.js';

export function createReducedTriplanePlan(factor) {
  const sourceSize = 96;
  if (!Number.isSafeInteger(factor) || factor < 1 || sourceSize % factor !== 0) {
    throw new RangeError('reduced triplane factor must divide 96');
  }
  return { sourceSize, inputSize: sourceSize / factor, outputSize: sourceSize * 4 / factor, factor };
}

export function createPreviewTetGrid(resolution) {
  if (!Number.isSafeInteger(resolution) || resolution < 2) {
    throw new RangeError('preview grid resolution must be an integer of at least 2');
  }
  const side = resolution + 1;
  const numVertices = side ** 3;
  const numTets = 6 * resolution ** 3;
  if (!Number.isSafeInteger(numVertices * 3) || !Number.isSafeInteger(numTets * 4)) {
    throw new RangeError('preview grid resolution exceeds safe array capacity');
  }
  const gridVertices = new Float32Array(numVertices * 3);
  const indices = new Int32Array(numTets * 4);
  const index = (x, y, z) => (z * side + y) * side + x;
  let vertexOffset = 0;
  for (let z = 0; z < side; z += 1) {
    for (let y = 0; y < side; y += 1) {
      for (let x = 0; x < side; x += 1) {
        gridVertices[vertexOffset++] = x / resolution;
        gridVertices[vertexOffset++] = y / resolution;
        gridVertices[vertexOffset++] = z / resolution;
      }
    }
  }
  let tetOffset = 0;
  for (let z = 0; z < resolution; z += 1) {
    for (let y = 0; y < resolution; y += 1) {
      for (let x = 0; x < resolution; x += 1) {
        const a = index(x, y, z);
        const b = index(x + 1, y, z);
        const c = index(x, y + 1, z);
        const d = index(x + 1, y + 1, z);
        const e = index(x, y, z + 1);
        const f = index(x + 1, y, z + 1);
        const g = index(x, y + 1, z + 1);
        const h = index(x + 1, y + 1, z + 1);
        for (const tet of [[a, b, d, h], [a, d, c, h], [a, c, g, h],
          [a, g, e, h], [a, e, f, h], [a, f, b, h]]) {
          indices.set(tet, tetOffset);
          tetOffset += 4;
        }
      }
    }
  }
  return { gridVertices, indices, numVertices, numTets, resolution };
}

export function extractPreviewMesh(grid, density, vertexOffsets, {
  threshold = 10,
  sourceResolution = 160,
  radius = 0.87,
  completedLayers = grid.resolution + 1,
} = {}) {
  if (density.length !== grid.numVertices) throw new RangeError('preview density length mismatch');
  if (vertexOffsets && vertexOffsets.length !== grid.numVertices * 3) {
    throw new RangeError('preview offset length mismatch');
  }
  if (!Number.isFinite(threshold) || !Number.isFinite(radius) || radius <= 0) {
    throw new RangeError('preview threshold and radius must be finite');
  }
  if (!Number.isSafeInteger(completedLayers) || completedLayers < 1 || completedLayers > grid.resolution + 1) {
    throw new RangeError('preview completedLayers outside grid');
  }
  // A cell is eligible only after both of its z planes have been decoded.
  // Unknown samples are never interpreted as outside/zero density.
  const knownVertices = completedLayers * (grid.resolution + 1) ** 2;
  const knownTets = (completedLayers - 1) * grid.resolution ** 2 * 6;
  density = density.subarray(0, knownVertices);
  vertexOffsets = vertexOffsets?.subarray(0, knownVertices * 3);
  for (const value of density) if (!Number.isFinite(value)) throw new Error('preview density contains non-finite values');
  if (vertexOffsets) {
    for (const value of vertexOffsets) if (!Number.isFinite(value)) throw new Error('preview offsets contain non-finite values');
  }
  const positions = scaleTensor(grid.gridVertices.subarray(0, knownVertices * 3), [0, 1], [-radius, radius]);
  const sdf = Float32Array.from(density, (value) => value - threshold);
  return marchingTetrahedra(positions, sdf, grid.indices.subarray(0, knownTets * 4), vertexOffsets, sourceResolution);
}
