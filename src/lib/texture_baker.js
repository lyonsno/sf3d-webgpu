/**
 * texture_baker.js — UV unwrapping + texture baking for SF3D WebGPU.
 *
 * Pipeline:
 *   1. Cube-projection UV unwrapping (matching SF3D's box-projection approach)
 *   2. CPU rasterization of UV space → 3D positions
 *   3. GPU triplane query + features decoder → RGB
 *   4. Texture dilation to fill seams
 *
 * The triplane query and decoder reuse triplane_decoder.js.
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';

/**
 * UV unwrap a mesh using cube projection.
 *
 * Each triangle is assigned to one of 6 cube faces based on its face normal,
 * then projected onto that face's 2D plane. The 6 projections are packed
 * into a 3×2 grid in UV space.
 *
 * Quality note: cube projection is simpler than xatlas but produces higher
 * distortion for triangles at 45° to principal axes. These get foreshortened
 * on their assigned face, causing stretched textures. The PyTorch reference
 * uses a more sophisticated PCA-aligned box projection with atlas packing.
 * This is a deliberate trade-off: pure JS, no WASM dependency, matching
 * SF3D's box-projection concept. UV utilization is ~5% of texture area.
 *
 * Vertices are duplicated per-face (no sharing across faces) to allow
 * per-face UVs without seam issues.
 *
 * @param {Float32Array} vertices - [N_v * 3] vertex positions
 * @param {Uint32Array} faces - [N_f * 3] triangle indices
 * @param {number} numVertices
 * @param {number} numFaces
 * @param {number} radius - model space radius for UV normalization
 * @returns {{ uvs: Float32Array, newVertices: Float32Array, newNormals: Float32Array, newFaces: Uint32Array, newNumVertices: number, newNumFaces: number }}
 */
export function unwrapUV(vertices, faces, numVertices, numFaces, radius = 0.87) {
  // Compute smooth vertex normals from the original shared-vertex topology.
  // Area-weighted: each face contributes its (unnormalized) cross product
  // to all 3 vertices. Larger faces contribute more. Then normalize.
  const smoothNormals = new Float32Array(numVertices * 3);
  const faceAssignment = new Uint8Array(numFaces);

  for (let f = 0; f < numFaces; f++) {
    const i0 = faces[f * 3], i1 = faces[f * 3 + 1], i2 = faces[f * 3 + 2];
    const v0x = vertices[i0*3], v0y = vertices[i0*3+1], v0z = vertices[i0*3+2];
    const v1x = vertices[i1*3], v1y = vertices[i1*3+1], v1z = vertices[i1*3+2];
    const v2x = vertices[i2*3], v2y = vertices[i2*3+1], v2z = vertices[i2*3+2];

    const e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    const e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;

    // Accumulate unnormalized face normal to each vertex (area-weighted)
    for (const idx of [i0, i1, i2]) {
      smoothNormals[idx * 3] += nx;
      smoothNormals[idx * 3 + 1] += ny;
      smoothNormals[idx * 3 + 2] += nz;
    }

    // Assign face to dominant cube axis
    const ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz);
    if (ax >= ay && ax >= az) {
      faceAssignment[f] = nx > 0 ? 0 : 1;
    } else if (ay >= ax && ay >= az) {
      faceAssignment[f] = ny > 0 ? 2 : 3;
    } else {
      faceAssignment[f] = nz > 0 ? 4 : 5;
    }
  }

  // Normalize accumulated normals
  for (let i = 0; i < numVertices; i++) {
    const nx = smoothNormals[i*3], ny = smoothNormals[i*3+1], nz = smoothNormals[i*3+2];
    const len = Math.sqrt(nx*nx + ny*ny + nz*nz) || 1;
    smoothNormals[i*3] /= len;
    smoothNormals[i*3+1] /= len;
    smoothNormals[i*3+2] /= len;
  }

  // Build per-face-vertex UVs (each face vertex gets its own UV)
  const newNumVertices = numFaces * 3;
  const newNumFaces = numFaces;
  const newVertices = new Float32Array(newNumVertices * 3);
  const newNormals = new Float32Array(newNumVertices * 3);
  const newFaces = new Uint32Array(newNumFaces * 3);
  const uvs = new Float32Array(newNumVertices * 2);

  // Grid layout: 3 columns × 2 rows
  const pad = 0.005;
  const cellW = 1 / 3;
  const cellH = 1 / 2;

  // Cube face → grid position
  const gridPos = [
    [0, 0], [1, 0], [2, 0],
    [0, 1], [1, 1], [2, 1],
  ];

  // UV projection axes per cube face:
  //   +X: (z, y)   -X: (-z, y)
  //   +Y: (x, z)   -Y: (x, -z)
  //   +Z: (x, y)   -Z: (-x, y)
  const projAxes = [
    [2, 1, false], [2, 1, true],
    [0, 2, false], [0, 2, true],
    [0, 1, false], [0, 1, true],
  ];

  for (let f = 0; f < numFaces; f++) {
    const cubeF = faceAssignment[f];
    const [col, row] = gridPos[cubeF];
    const [ax0, ax1, flip] = projAxes[cubeF];

    for (let v = 0; v < 3; v++) {
      const origIdx = faces[f * 3 + v];
      const newIdx = f * 3 + v;

      // Copy position and smooth normal from original topology
      newVertices[newIdx * 3] = vertices[origIdx * 3];
      newVertices[newIdx * 3 + 1] = vertices[origIdx * 3 + 1];
      newVertices[newIdx * 3 + 2] = vertices[origIdx * 3 + 2];
      newNormals[newIdx * 3] = smoothNormals[origIdx * 3];
      newNormals[newIdx * 3 + 1] = smoothNormals[origIdx * 3 + 1];
      newNormals[newIdx * 3 + 2] = smoothNormals[origIdx * 3 + 2];

      // Project onto cube face axes
      let u = vertices[origIdx * 3 + ax0];
      let vv = vertices[origIdx * 3 + ax1];
      if (flip) u = -u;

      // Normalize from [-radius, radius] to [0, 1]
      u = (u / radius + 1) * 0.5;
      vv = (vv / radius + 1) * 0.5;

      // Map to grid cell with padding
      uvs[newIdx * 2] = col * cellW + pad + u * (cellW - 2 * pad);
      uvs[newIdx * 2 + 1] = row * cellH + pad + vv * (cellH - 2 * pad);

      newFaces[f * 3 + v] = newIdx;
    }
  }

  return { uvs, newVertices, newNormals, newFaces, newNumVertices, newNumFaces };
}

/**
 * Rasterize UV space to get 3D positions and TBN basis at each texel.
 *
 * For each triangle in UV space, rasterize its bounding box and compute
 * barycentric interpolation of 3D positions. Also computes per-face
 * tangent/bitangent/normal basis from UV edges and position edges.
 *
 * @param {Float32Array} uvs - [N_v * 2] UV coordinates
 * @param {Float32Array} positions - [N_v * 3] vertex positions
 * @param {Uint32Array} faces - [N_f * 3] face indices
 * @param {number} numFaces
 * @param {number} resolution - texture resolution (default 1024)
 * @returns {{ positions3D: Float32Array, mask: Uint8Array, tbnData: Float32Array }}
 */
export function rasterizeUV(uvs, positions, faces, numFaces, resolution = 1024) {
  const positions3D = new Float32Array(resolution * resolution * 3);
  const mask = new Uint8Array(resolution * resolution);
  // TBN: 9 floats per texel (tangent[3], bitangent[3], normal[3])
  const tbnData = new Float32Array(resolution * resolution * 9);

  for (let f = 0; f < numFaces; f++) {
    const i0 = faces[f * 3], i1 = faces[f * 3 + 1], i2 = faces[f * 3 + 2];

    const u0 = uvs[i0 * 2], v0 = uvs[i0 * 2 + 1];
    const u1 = uvs[i1 * 2], v1 = uvs[i1 * 2 + 1];
    const u2 = uvs[i2 * 2], v2 = uvs[i2 * 2 + 1];

    const p0x = positions[i0*3], p0y = positions[i0*3+1], p0z = positions[i0*3+2];
    const p1x = positions[i1*3], p1y = positions[i1*3+1], p1z = positions[i1*3+2];
    const p2x = positions[i2*3], p2y = positions[i2*3+1], p2z = positions[i2*3+2];

    // Compute face TBN from edges and UV deltas
    const e1x = p1x-p0x, e1y = p1y-p0y, e1z = p1z-p0z;
    const e2x = p2x-p0x, e2y = p2y-p0y, e2z = p2z-p0z;
    const duv1u = u1-u0, duv1v = v1-v0;
    const duv2u = u2-u0, duv2v = v2-v0;

    // Face normal
    let fnx = e1y*e2z - e1z*e2y;
    let fny = e1z*e2x - e1x*e2z;
    let fnz = e1x*e2y - e1y*e2x;
    let fnLen = Math.sqrt(fnx*fnx + fny*fny + fnz*fnz) || 1;
    fnx /= fnLen; fny /= fnLen; fnz /= fnLen;

    // Tangent from UV gradients: T = (e1 * duv2v - e2 * duv1v) / det
    const det = duv1u * duv2v - duv1v * duv2u;
    let tx, ty, tz, bx, by, bz;
    if (Math.abs(det) > 1e-10) {
      const invDet = 1.0 / det;
      tx = (e1x * duv2v - e2x * duv1v) * invDet;
      ty = (e1y * duv2v - e2y * duv1v) * invDet;
      tz = (e1z * duv2v - e2z * duv1v) * invDet;
    } else {
      // Degenerate UV: pick arbitrary tangent perpendicular to normal
      tx = 1; ty = 0; tz = 0;
      if (Math.abs(fnx) > 0.9) { tx = 0; ty = 1; }
    }

    // Orthogonalize tangent against normal (Gram-Schmidt)
    const tDotN = tx*fnx + ty*fny + tz*fnz;
    tx -= tDotN * fnx; ty -= tDotN * fny; tz -= tDotN * fnz;
    let tLen = Math.sqrt(tx*tx + ty*ty + tz*tz) || 1;
    tx /= tLen; ty /= tLen; tz /= tLen;

    // Bitangent = cross(normal, tangent)
    bx = fny*tz - fnz*ty;
    by = fnz*tx - fnx*tz;
    bz = fnx*ty - fny*tx;
    let bLen = Math.sqrt(bx*bx + by*by + bz*bz) || 1;
    bx /= bLen; by /= bLen; bz /= bLen;

    // Bounding box in pixel coords
    const minPx = Math.max(0, Math.floor(Math.min(u0, u1, u2) * resolution));
    const maxPx = Math.min(resolution - 1, Math.ceil(Math.max(u0, u1, u2) * resolution));
    const minPy = Math.max(0, Math.floor(Math.min(v0, v1, v2) * resolution));
    const maxPy = Math.min(resolution - 1, Math.ceil(Math.max(v0, v1, v2) * resolution));

    const denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2);
    if (Math.abs(denom) < 1e-10) continue;
    const invDenom = 1.0 / denom;

    for (let py = minPy; py <= maxPy; py++) {
      for (let px = minPx; px <= maxPx; px++) {
        const u = (px + 0.5) / resolution;
        const v = (py + 0.5) / resolution;

        const w0 = ((v1 - v2) * (u - u2) + (u2 - u1) * (v - v2)) * invDenom;
        const w1 = ((v2 - v0) * (u - u2) + (u0 - u2) * (v - v2)) * invDenom;
        const w2 = 1 - w0 - w1;

        if (w0 >= -0.001 && w1 >= -0.001 && w2 >= -0.001) {
          const pixIdx = py * resolution + px;
          mask[pixIdx] = 1;
          positions3D[pixIdx * 3] = w0 * p0x + w1 * p1x + w2 * p2x;
          positions3D[pixIdx * 3 + 1] = w0 * p0y + w1 * p1y + w2 * p2y;
          positions3D[pixIdx * 3 + 2] = w0 * p0z + w1 * p1z + w2 * p2z;
          // TBN is constant per face (vertices are duplicated per face)
          const tbnBase = pixIdx * 9;
          tbnData[tbnBase]   = tx; tbnData[tbnBase+1] = ty; tbnData[tbnBase+2] = tz;
          tbnData[tbnBase+3] = bx; tbnData[tbnBase+4] = by; tbnData[tbnBase+5] = bz;
          tbnData[tbnBase+6] = fnx; tbnData[tbnBase+7] = fny; tbnData[tbnBase+8] = fnz;
        }
      }
    }
  }

  return { positions3D, mask, tbnData };
}

/**
 * Bake albedo texture and normal map by querying the triplane decoder.
 *
 * @param {GPUDevice} device
 * @param {Object} triplaneDecoder - TriplaneDecoder instance
 * @param {GPUBuffer} triplanesBuf - [3, 40, 384, 384] triplane features
 * @param {Object} decoderWeights - decoder weights
 * @param {Float32Array} positions3D - [res*res, 3] from rasterizeUV
 * @param {Uint8Array} mask - [res*res] from rasterizeUV
 * @param {Float32Array} tbnData - [res*res, 9] TBN basis from rasterizeUV
 * @param {number} resolution
 * @returns {{ albedo: Uint8Array, normalMap: Uint8Array }} - [res, res, 4] RGBA textures
 */
export async function bakeTexture(device, triplaneDecoder, triplanesBuf, decoderWeights,
                                   positions3D, mask, tbnData, resolution = 1024) {
  // Collect occupied texel positions
  const occupiedIndices = [];
  for (let i = 0; i < resolution * resolution; i++) {
    if (mask[i]) occupiedIndices.push(i);
  }

  const numOccupied = occupiedIndices.length;
  console.log(`Texture bake: ${numOccupied} occupied texels out of ${resolution * resolution}`);

  const emptyTex = new Uint8Array(resolution * resolution * 4);
  if (numOccupied === 0) {
    return { albedo: emptyTex, normalMap: new Uint8Array(emptyTex) };
  }

  // Pack occupied positions into a dense array
  const queryPositions = new Float32Array(numOccupied * 3);
  for (let i = 0; i < numOccupied; i++) {
    const idx = occupiedIndices[i];
    queryPositions[i * 3] = positions3D[idx * 3];
    queryPositions[i * 3 + 1] = positions3D[idx * 3 + 1];
    queryPositions[i * 3 + 2] = positions3D[idx * 3 + 2];
  }

  // Upload to GPU and decode features + perturb_normal
  const queryPosBuf = createStorageBuffer(device, queryPositions);

  const encoder = device.createCommandEncoder();
  const decoded = triplaneDecoder.decode(
    encoder, queryPosBuf, triplanesBuf, numOccupied,
    decoderWeights, ['features', 'perturb_normal']);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  // Read back both outputs
  const featuresCPU = await readBuffer(device, decoded.features, numOccupied * 3 * 4);
  const normalsCPU = await readBuffer(device, decoded.perturb_normal, numOccupied * 3 * 4);

  // Build albedo RGBA texture
  const albedo = new Uint8Array(resolution * resolution * 4);
  for (let i = 0; i < numOccupied; i++) {
    const texIdx = occupiedIndices[i];
    albedo[texIdx * 4] = Math.max(0, Math.min(255, Math.round(featuresCPU[i * 3] * 255)));
    albedo[texIdx * 4 + 1] = Math.max(0, Math.min(255, Math.round(featuresCPU[i * 3 + 1] * 255)));
    albedo[texIdx * 4 + 2] = Math.max(0, Math.min(255, Math.round(featuresCPU[i * 3 + 2] * 255)));
    albedo[texIdx * 4 + 3] = 255;
  }

  // Build normal map: transform perturb_normal from world space to tangent space
  const normalMap = new Uint8Array(resolution * resolution * 4);
  // Default normal (pointing straight out): [0.5, 0.5, 1.0] in encoded space
  for (let i = 0; i < resolution * resolution; i++) {
    normalMap[i * 4 + 2] = 255; // blue channel = 1.0 (pointing along surface normal)
    normalMap[i * 4 + 3] = 255;
  }

  for (let i = 0; i < numOccupied; i++) {
    const texIdx = occupiedIndices[i];
    const tbnBase = texIdx * 9;

    // World-space perturb_normal (already normalized by decoder)
    const nx = normalsCPU[i * 3];
    const ny = normalsCPU[i * 3 + 1];
    const nz = normalsCPU[i * 3 + 2];

    // TBN basis vectors
    const tx = tbnData[tbnBase], ty = tbnData[tbnBase+1], tz = tbnData[tbnBase+2];
    const bx = tbnData[tbnBase+3], by = tbnData[tbnBase+4], bz = tbnData[tbnBase+5];
    const fnx = tbnData[tbnBase+6], fny = tbnData[tbnBase+7], fnz = tbnData[tbnBase+8];

    // Transform to tangent space: n_tangent = TBN^T * n_world
    let ntx = tx*nx + ty*ny + tz*nz;     // dot(tangent, normal_world)
    let nty = bx*nx + by*ny + bz*nz;     // dot(bitangent, normal_world)
    let ntz = fnx*nx + fny*ny + fnz*nz;  // dot(face_normal, normal_world)

    // Encode from [-1,1] to [0,1]: encoded = n * 0.5 + 0.5
    const r = Math.max(0, Math.min(255, Math.round((ntx * 0.5 + 0.5) * 255)));
    const g = Math.max(0, Math.min(255, Math.round((nty * 0.5 + 0.5) * 255)));
    const b = Math.max(0, Math.min(255, Math.round((ntz * 0.5 + 0.5) * 255)));

    normalMap[texIdx * 4] = r;
    normalMap[texIdx * 4 + 1] = g;
    normalMap[texIdx * 4 + 2] = b;
    normalMap[texIdx * 4 + 3] = 255;
  }

  // Dilate both textures
  _dilateTexture(albedo, mask, resolution, 6);
  _dilateTexture(normalMap, mask, resolution, 6);

  return { albedo, normalMap };
}

/**
 * Dilate texture to fill empty pixels by averaging nearest occupied neighbors.
 */
function _dilateTexture(texture, mask, resolution, iterations = 6) {
  const workMask = new Uint8Array(mask);

  for (let iter = 0; iter < iterations; iter++) {
    const newPixels = [];

    for (let y = 0; y < resolution; y++) {
      for (let x = 0; x < resolution; x++) {
        const idx = y * resolution + x;
        if (workMask[idx]) continue;

        let sumR = 0, sumG = 0, sumB = 0, count = 0;
        const neighbors = [[x-1,y],[x+1,y],[x,y-1],[x,y+1]];
        for (const [nx, ny] of neighbors) {
          if (nx < 0 || nx >= resolution || ny < 0 || ny >= resolution) continue;
          const nIdx = ny * resolution + nx;
          if (workMask[nIdx]) {
            sumR += texture[nIdx * 4];
            sumG += texture[nIdx * 4 + 1];
            sumB += texture[nIdx * 4 + 2];
            count++;
          }
        }

        if (count > 0) {
          newPixels.push({
            idx,
            r: Math.round(sumR / count),
            g: Math.round(sumG / count),
            b: Math.round(sumB / count),
          });
        }
      }
    }

    for (const p of newPixels) {
      texture[p.idx * 4] = p.r;
      texture[p.idx * 4 + 1] = p.g;
      texture[p.idx * 4 + 2] = p.b;
      texture[p.idx * 4 + 3] = 255;
      workMask[p.idx] = 1;
    }

    if (newPixels.length === 0) break;
  }
}

/**
 * Export mesh as GLB (binary glTF 2.0).
 *
 * @param {Float32Array} vertices - [N_v * 3]
 * @param {Float32Array} vertexNormals - [N_v * 3] pre-computed smooth normals
 * @param {Uint32Array} faces - [N_f * 3]
 * @param {Float32Array} uvs - [N_v * 2]
 * @param {Uint8Array} albedoTexture - [res, res, 4] RGBA
 * @param {Uint8Array|null} normalMapTexture - [res, res, 4] RGBA or null
 * @param {number} numVertices
 * @param {number} numFaces
 * @param {number} textureResolution
 * @param {number} roughness
 * @param {number} metallic
 * @returns {ArrayBuffer} GLB binary
 */
export async function exportGLB(vertices, vertexNormals, faces, uvs,
                                 albedoTexture, normalMapTexture,
                                 numVertices, numFaces, textureResolution = 1024,
                                 roughness = 0.5, metallic = 0.0) {
  if (numVertices === 0 || numFaces === 0) {
    throw new Error('Cannot export empty mesh as GLB');
  }

  // Apply coordinate transforms to match glTF conventions
  // Combined: rot(-90, X) then rot(+90, Y) gives (x,y,z) → (-y, z, -x)
  // Then invert face winding to match PyTorch's mesh.invert()
  const transformedVerts = new Float32Array(numVertices * 3);
  for (let i = 0; i < numVertices; i++) {
    const x = vertices[i * 3];
    const y = vertices[i * 3 + 1];
    const z = vertices[i * 3 + 2];
    const rx = x, ry = z, rz = -y;
    transformedVerts[i * 3] = rz;
    transformedVerts[i * 3 + 1] = ry;
    transformedVerts[i * 3 + 2] = -rx;
  }

  const invertedFaces = new Uint32Array(numFaces * 3);
  for (let f = 0; f < numFaces; f++) {
    invertedFaces[f * 3] = faces[f * 3];
    invertedFaces[f * 3 + 1] = faces[f * 3 + 2];
    invertedFaces[f * 3 + 2] = faces[f * 3 + 1];
  }

  // Apply same rotation to pre-computed smooth normals, then negate
  // for face inversion (winding flip makes outward normals point inward)
  // Rotation (x,y,z)→(-y,z,-x), then negate: (y,-z,x)
  const normals = new Float32Array(numVertices * 3);
  for (let i = 0; i < numVertices; i++) {
    const nx = vertexNormals[i * 3];
    const ny = vertexNormals[i * 3 + 1];
    const nz = vertexNormals[i * 3 + 2];
    normals[i * 3] = ny;
    normals[i * 3 + 1] = -nz;
    normals[i * 3 + 2] = nx;
  }

  // Encode textures as JPEG
  const albedoBlob = await _textureToJPEG(albedoTexture, textureResolution);
  if (!albedoBlob) throw new Error('Failed to encode albedo texture as JPEG');
  const albedoBytes = new Uint8Array(await albedoBlob.arrayBuffer());

  let normalBytes = null;
  if (normalMapTexture) {
    const normalBlob = await _textureToJPEG(normalMapTexture, textureResolution, 0.95);
    if (normalBlob) normalBytes = new Uint8Array(await normalBlob.arrayBuffer());
  }

  // Compute bounding box
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < numVertices; i++) {
    const x = transformedVerts[i*3], y = transformedVerts[i*3+1], z = transformedVerts[i*3+2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }

  const pad4 = (n) => (n + 3) & ~3;

  const vertexBytes = new Uint8Array(transformedVerts.buffer, transformedVerts.byteOffset, transformedVerts.byteLength);
  const vnormalBytes = new Uint8Array(normals.buffer, normals.byteOffset, normals.byteLength);
  const indexBytes = new Uint8Array(invertedFaces.buffer, invertedFaces.byteOffset, invertedFaces.byteLength);
  const uvBytes = new Uint8Array(uvs.buffer, uvs.byteOffset, uvs.byteLength);

  const vertexLen = pad4(vertexBytes.byteLength);
  const vnormalLen = pad4(vnormalBytes.byteLength);
  const indexLen = pad4(indexBytes.byteLength);
  const uvLen = pad4(uvBytes.byteLength);
  const albedoLen = pad4(albedoBytes.byteLength);
  const normalTexLen = normalBytes ? pad4(normalBytes.byteLength) : 0;
  const totalBinLen = vertexLen + vnormalLen + indexLen + uvLen + albedoLen + normalTexLen;

  let off = 0;
  const bufferViews = [
    { buffer: 0, byteOffset: (off), byteLength: vertexBytes.byteLength, target: 34962 },
    { buffer: 0, byteOffset: (off += vertexLen), byteLength: vnormalBytes.byteLength, target: 34962 },
    { buffer: 0, byteOffset: (off += vnormalLen), byteLength: indexBytes.byteLength, target: 34963 },
    { buffer: 0, byteOffset: (off += indexLen), byteLength: uvBytes.byteLength, target: 34962 },
    { buffer: 0, byteOffset: (off += uvLen), byteLength: albedoBytes.byteLength }, // albedo image
  ];
  const albedoImageBV = 4;
  let normalImageBV = -1;
  if (normalBytes) {
    normalImageBV = bufferViews.length;
    bufferViews.push({ buffer: 0, byteOffset: (off += albedoLen), byteLength: normalBytes.byteLength });
  }

  const accessors = [
    { bufferView: 0, componentType: 5126, count: numVertices, type: 'VEC3',
      min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
    { bufferView: 1, componentType: 5126, count: numVertices, type: 'VEC3' },
    { bufferView: 2, componentType: 5125, count: numFaces * 3, type: 'SCALAR' },
    { bufferView: 3, componentType: 5126, count: numVertices, type: 'VEC2' },
  ];

  const images = [{ bufferView: albedoImageBV, mimeType: 'image/jpeg' }];
  const textures = [{ source: 0, sampler: 0 }];
  if (normalBytes) {
    images.push({ bufferView: normalImageBV, mimeType: 'image/jpeg' });
    textures.push({ source: 1, sampler: 0 });
  }

  const material = {
    pbrMetallicRoughness: {
      baseColorTexture: { index: 0 },
      roughnessFactor: roughness,
      metallicFactor: metallic,
    },
  };
  if (normalBytes) {
    material.normalTexture = { index: 1 };
  }

  const gltf = {
    asset: { version: '2.0', generator: 'SF3D-WebGPU' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      primitives: [{
        attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 3 },
        indices: 2,
        material: 0,
      }],
    }],
    materials: [material],
    textures,
    images,
    samplers: [{ magFilter: 9729, minFilter: 9729 }],
    accessors,
    bufferViews,
    buffers: [{ byteLength: totalBinLen }],
  };

  const jsonStr = JSON.stringify(gltf);
  const jsonBytes = new TextEncoder().encode(jsonStr);
  const jsonPadLen = pad4(jsonBytes.byteLength);

  const glbLen = 12 + 8 + jsonPadLen + 8 + totalBinLen;
  const glb = new ArrayBuffer(glbLen); // zero-initialized per JS spec (BIN padding = 0x00)
  const view = new DataView(glb);
  const bytes = new Uint8Array(glb);

  view.setUint32(0, 0x46546C67, true); // "glTF"
  view.setUint32(4, 2, true);
  view.setUint32(8, glbLen, true);

  let offset = 12;
  view.setUint32(offset, jsonPadLen, true);
  view.setUint32(offset + 4, 0x4E4F534A, true); // "JSON"
  offset += 8;
  bytes.set(jsonBytes, offset);
  for (let i = jsonBytes.byteLength; i < jsonPadLen; i++) bytes[offset + i] = 0x20;
  offset += jsonPadLen;

  view.setUint32(offset, totalBinLen, true);
  view.setUint32(offset + 4, 0x004E4942, true); // "BIN\0"
  offset += 8;

  bytes.set(vertexBytes, offset); offset += vertexLen;
  bytes.set(vnormalBytes, offset); offset += vnormalLen;
  bytes.set(indexBytes, offset); offset += indexLen;
  bytes.set(uvBytes, offset); offset += uvLen;
  bytes.set(albedoBytes, offset); offset += albedoLen;
  if (normalBytes) { bytes.set(normalBytes, offset); }

  return glb;
}

function _textureToJPEG(texture, resolution, quality = 0.92) {
  const canvas = document.createElement('canvas');
  canvas.width = resolution;
  canvas.height = resolution;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(resolution, resolution);
  imgData.data.set(texture);
  ctx.putImageData(imgData, 0, 0);
  return new Promise((resolve) => {
    canvas.toBlob(resolve, 'image/jpeg', quality);
  });
}
