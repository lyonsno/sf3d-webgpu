/**
 * texture_baker.js — UV unwrapping + texture baking for SF3D WebGPU.
 *
 * Pipeline:
 *   1. Cube-projection UV unwrapping with bbox normalization
 *   2. CPU rasterization of UV space → 3D positions (with depth buffer)
 *   3. GPU triplane query + features decoder → RGB
 *   4. Texture dilation to fill seams
 *
 * The triplane query and decoder reuse triplane_decoder.js.
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';

/**
 * UV unwrap a mesh using cube projection with bounding-box normalization.
 *
 * Each triangle is assigned to one of 6 cube faces based on its face normal,
 * then projected onto that face's 2D plane using the mesh's actual bounding
 * box (not a fixed radius) for UV normalization. The 6 projections are packed
 * into a 3×2 grid in UV space.
 *
 * Vertices are duplicated per-face (no sharing across faces) to allow
 * per-face UVs without seam issues.
 *
 * @param {Float32Array} vertices - [N_v * 3] vertex positions
 * @param {Uint32Array} faces - [N_f * 3] triangle indices
 * @param {number} numVertices
 * @param {number} numFaces
 * @param {number} radius - model space radius (unused, kept for API compat)
 * @returns {{ uvs, newVertices, newNormals, newFaces, newNumVertices, newNumFaces, faceAssignment }}
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

  // Compute actual bounding box for UV normalization
  let bboxMin = [Infinity, Infinity, Infinity];
  let bboxMax = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < numVertices; i++) {
    for (let a = 0; a < 3; a++) {
      const v = vertices[i * 3 + a];
      if (v < bboxMin[a]) bboxMin[a] = v;
      if (v > bboxMax[a]) bboxMax[a] = v;
    }
  }
  // Half-extents per axis (like a per-axis radius)
  const extent = [
    Math.max(Math.abs(bboxMin[0]), Math.abs(bboxMax[0])) || 1,
    Math.max(Math.abs(bboxMin[1]), Math.abs(bboxMax[1])) || 1,
    Math.max(Math.abs(bboxMin[2]), Math.abs(bboxMax[2])) || 1,
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

  // Depth axis per cube face (perpendicular to projection plane)
  const depthAxis = [0, 0, 1, 1, 2, 2];
  const depthKeepMax = [true, false, true, false, true, false];

  // --- Step 1: Compute raw [0,1] UVs and depth centroids ---
  const rawU = new Float32Array(numFaces * 3);
  const rawV = new Float32Array(numFaces * 3);
  const centroidDepth = new Float32Array(numFaces);

  for (let f = 0; f < numFaces; f++) {
    const cubeF = faceAssignment[f];
    const [ax0, ax1, flip] = projAxes[cubeF];
    const dAxis = depthAxis[cubeF];
    let depthSum = 0;
    for (let vi = 0; vi < 3; vi++) {
      const origIdx = faces[f * 3 + vi];
      let u = vertices[origIdx * 3 + ax0];
      let vv = vertices[origIdx * 3 + ax1];
      if (flip) u = -u;
      rawU[f * 3 + vi] = (u / extent[ax0] + 1) * 0.5;
      rawV[f * 3 + vi] = (vv / extent[ax1] + 1) * 0.5;
      depthSum += vertices[origIdx * 3 + dAxis];
    }
    centroidDepth[f] = depthSum / 3;
  }

  // --- Step 2: Detect UV overlaps and assign atlas indices ---
  // 0-5 = primary, 6-11 = first overlap, 12 = remaining
  const atlasIndex = new Int32Array(numFaces);
  for (let f = 0; f < numFaces; f++) atlasIndex[f] = faceAssignment[f];

  const GRID = 128;
  _detectOverlapsGrid(numFaces, atlasIndex, rawU, rawV, centroidDepth, depthKeepMax, GRID, 0);
  _detectOverlapsGrid(numFaces, atlasIndex, rawU, rawV, centroidDepth, depthKeepMax, GRID, 6);

  // --- Step 2b: Per-island UV normalization for secondary tier (slots 6-11) ---
  // Matching PyTorch _handle_slice_uvs: rescale all faces in each secondary
  // slot so their UVs fill [0,1], with max 2x magnification (clip denom at 0.5).
  for (let slot = 6; slot < 12; slot++) {
    let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
    let count = 0;
    for (let f = 0; f < numFaces; f++) {
      if (atlasIndex[f] !== slot) continue;
      count++;
      for (let vi = 0; vi < 3; vi++) {
        const u = rawU[f*3+vi], v = rawV[f*3+vi];
        if (u < minU) minU = u; if (u > maxU) maxU = u;
        if (v < minV) minV = v; if (v > maxV) maxV = v;
      }
    }
    if (count === 0) continue;
    const rangeU = Math.max(maxU - minU, 0.5); // clip at 0.5 = max 2x magnification
    const rangeV = Math.max(maxV - minV, 0.5);
    for (let f = 0; f < numFaces; f++) {
      if (atlasIndex[f] !== slot) continue;
      for (let vi = 0; vi < 3; vi++) {
        rawU[f*3+vi] = (rawU[f*3+vi] - minU) / rangeU;
        rawV[f*3+vi] = (rawV[f*3+vi] - minV) / rangeV;
      }
    }
  }

  // --- Step 3: Build per-face-vertex arrays ---
  const newNumVertices = numFaces * 3;
  const newNumFaces = numFaces;
  const newVertices = new Float32Array(newNumVertices * 3);
  const newNormals = new Float32Array(newNumVertices * 3);
  const newFaces = new Uint32Array(newNumFaces * 3);
  const uvs = new Float32Array(newNumVertices * 2);

  for (let f = 0; f < numFaces; f++) {
    for (let vi = 0; vi < 3; vi++) {
      const origIdx = faces[f * 3 + vi];
      const newIdx = f * 3 + vi;
      newVertices[newIdx * 3] = vertices[origIdx * 3];
      newVertices[newIdx * 3 + 1] = vertices[origIdx * 3 + 1];
      newVertices[newIdx * 3 + 2] = vertices[origIdx * 3 + 2];
      newNormals[newIdx * 3] = smoothNormals[origIdx * 3];
      newNormals[newIdx * 3 + 1] = smoothNormals[origIdx * 3 + 1];
      newNormals[newIdx * 3 + 2] = smoothNormals[origIdx * 3 + 2];
      newFaces[f * 3 + vi] = newIdx;
    }
  }

  // --- Step 4: Pack UVs into atlas ---
  // Layout matching PyTorch _find_slice_offset_and_scale:
  //   Primary (0-5):   3×2 grid, cell=1/3×1/3, region [0,1]×[0,2/3]
  //   Secondary (6-11): 3×2 grid, cell=1/6×1/6, region [0,1/2]×[2/3,1]
  //   Remaining (12+):  sub-cell grid in region [1/2,1]×[2/3,1]
  const pad = 0.005;

  const slotCol = [0, 1, 2, 0, 1, 2];
  const slotRow = [0, 0, 0, 1, 1, 1];

  // Collect remaining faces (atlasIndex >= 12) and compute their sub-cell grid
  const remainingFaces = [];
  for (let f = 0; f < numFaces; f++) {
    if (atlasIndex[f] >= 12) remainingFaces.push(f);
  }
  // Sub-cell grid for remaining faces: pack into [0.5,1]×[2/3,1] = 0.5 × 1/3
  const remRegionW = 0.5, remRegionH = 1 / 3;
  const remRegionX = 0.5, remRegionY = 2 / 3;
  let remGridW = 1, remGridH = 1;
  if (remainingFaces.length > 0) {
    const ratio = remRegionW / remRegionH; // aspect ratio of remaining region
    const mult = Math.sqrt(remainingFaces.length / ratio);
    remGridW = Math.max(1, Math.ceil(ratio * mult));
    remGridH = Math.max(1, Math.ceil(remainingFaces.length / remGridW));
  }
  const remCellW = remRegionW / remGridW;
  const remCellH = remRegionH / remGridH;

  // Build a map from face index to remaining-grid position
  const remFaceGridIdx = new Map();
  for (let i = 0; i < remainingFaces.length; i++) {
    remFaceGridIdx.set(remainingFaces[i], i);
  }

  for (let f = 0; f < numFaces; f++) {
    const ai = atlasIndex[f];
    const tier = Math.floor(ai / 6);
    const slot = ai % 6;
    const sc = slotCol[slot], sr = slotRow[slot];

    let offX, offY, cellW, cellH;
    if (tier === 0) {
      cellW = 1/3; cellH = 1/3;
      offX = cellW * sc;
      offY = cellH * sr;
    } else if (tier === 1) {
      cellW = 1/6; cellH = 1/6;
      offX = cellW * sc;
      offY = 2/3 + cellH * sr;
    } else {
      // Remaining: each face gets its own sub-cell
      cellW = remCellW; cellH = remCellH;
      const gi = remFaceGridIdx.get(f) || 0;
      const gx = gi % remGridW;
      const gy = Math.floor(gi / remGridW);
      offX = remRegionX + gx * cellW;
      offY = remRegionY + gy * cellH;
    }

    for (let vi = 0; vi < 3; vi++) {
      const idx = f * 3 + vi;
      // Per-face UV normalization for remaining tier:
      // normalize each triangle's UVs to fill [0,1] within its sub-cell
      let u = rawU[idx], v = rawV[idx];
      if (tier >= 2) {
        // Normalize per-triangle: find min/max across this triangle's 3 verts
        const u0 = rawU[f*3], u1 = rawU[f*3+1], u2 = rawU[f*3+2];
        const v0 = rawV[f*3], v1 = rawV[f*3+1], v2 = rawV[f*3+2];
        const uMin = Math.min(u0, u1, u2), uMax = Math.max(u0, u1, u2);
        const vMin = Math.min(v0, v1, v2), vMax = Math.max(v0, v1, v2);
        const uRange = uMax - uMin || 1;
        const vRange = vMax - vMin || 1;
        // Clamp scale to prevent extreme magnification (match PyTorch clip_val)
        const clipVal = Math.min(cellW, cellH) * 1.5;
        u = (u - uMin) / Math.max(uRange, clipVal);
        v = (v - vMin) / Math.max(vRange, clipVal);
      }
      uvs[idx * 2] = offX + pad + u * (cellW - 2 * pad);
      uvs[idx * 2 + 1] = offY + pad + v * (cellH - 2 * pad);
    }
  }

  // Diagnostic: check for degenerate UVs and out-of-bounds
  let degenerateCount = 0, oobCount = 0;
  const degFaces = [];
  for (let f = 0; f < numFaces; f++) {
    const u0 = uvs[f*6], v0 = uvs[f*6+1];
    const u1 = uvs[f*6+2], v1 = uvs[f*6+3];
    const u2 = uvs[f*6+4], v2 = uvs[f*6+5];
    // UV triangle area via cross product
    const area = Math.abs((u1-u0)*(v2-v0) - (u2-u0)*(v1-v0)) * 0.5;
    if (area < 1e-10) {
      degenerateCount++;
      if (degFaces.length < 10) degFaces.push({ f, tier: Math.floor(atlasIndex[f]/6), ai: atlasIndex[f], area, u0, v0, u1, v1, u2, v2 });
    }
    // Check OOB
    if (u0 < 0 || u0 > 1 || v0 < 0 || v0 > 1 ||
        u1 < 0 || u1 > 1 || v1 < 0 || v1 > 1 ||
        u2 < 0 || u2 > 1 || v2 < 0 || v2 > 1) oobCount++;
  }
  console.log(`UV diagnostics: degenerate=${degenerateCount}, out-of-bounds=${oobCount}, total=${numFaces}`);
  if (degFaces.length > 0) console.log(`Sample degenerate faces: ${JSON.stringify(degFaces.slice(0, 5))}`);

  // Also check: faces where all 3 UV verts map to the same texel at 1024 res
  let sameTexelCount = 0;
  const sameTexelFaces = [];
  for (let f = 0; f < numFaces; f++) {
    const px0 = Math.floor(uvs[f*6] * 1024), py0 = Math.floor(uvs[f*6+1] * 1024);
    const px1 = Math.floor(uvs[f*6+2] * 1024), py1 = Math.floor(uvs[f*6+3] * 1024);
    const px2 = Math.floor(uvs[f*6+4] * 1024), py2 = Math.floor(uvs[f*6+5] * 1024);
    if (px0 === px1 && px1 === px2 && py0 === py1 && py1 === py2) {
      sameTexelCount++;
      if (sameTexelFaces.length < 5) sameTexelFaces.push({ f, tier: Math.floor(atlasIndex[f]/6), ai: atlasIndex[f] });
    }
  }
  console.log(`Faces mapping to single texel at 1024: ${sameTexelCount} (these get 0 rasterized texels)`);
  // Check: how many secondary/remaining faces have UV area < 1 texel at 1024?
  let subTexelSecondary = 0, subTexelRemaining = 0;
  for (let f = 0; f < numFaces; f++) {
    const tier = Math.floor(atlasIndex[f] / 6);
    if (tier === 0) continue;
    const u0 = uvs[f*6], v0 = uvs[f*6+1];
    const u1 = uvs[f*6+2], v1 = uvs[f*6+3];
    const u2 = uvs[f*6+4], v2 = uvs[f*6+5];
    const texelArea = Math.abs((u1-u0)*(v2-v0) - (u2-u0)*(v1-v0)) * 0.5 * 1024 * 1024;
    if (texelArea < 1.0) {
      if (tier === 1) subTexelSecondary++;
      else subTexelRemaining++;
    }
  }
  console.log(`Sub-texel faces: secondary=${subTexelSecondary}, remaining=${subTexelRemaining}`);
  if (sameTexelFaces.length > 0) console.log(`Sample same-texel faces: ${JSON.stringify(sameTexelFaces)}`);

  // Diagnostic: count faces per tier
  const tierCounts = [0, 0, 0];
  for (let f = 0; f < numFaces; f++) {
    const t = Math.min(Math.floor(atlasIndex[f] / 6), 2);
    tierCounts[t]++;
  }
  console.log(`Atlas tiers: primary=${tierCounts[0]}, secondary=${tierCounts[1]}, remaining=${tierCounts[2]}, total=${numFaces}`);

  return { uvs, newVertices, newNormals, newFaces, newNumVertices, newNumFaces, faceAssignment: atlasIndex };
}

/**
 * Detect UV overlaps within each slot group using a rasterized grid.
 * When two triangles in the same slot overlap in UV space, the occluded
 * one (by 3D centroid depth along the cube face axis) gets bumped to slot+6.
 */
function _detectOverlapsGrid(numFaces, atlasIndex, rawU, rawV, centroidDepth,
    depthKeepMax, gridSize, slotOffset) {

  for (let slot = slotOffset; slot < slotOffset + 6; slot++) {
    const slotFaces = [];
    for (let f = 0; f < numFaces; f++) {
      if (atlasIndex[f] === slot) slotFaces.push(f);
    }
    if (slotFaces.length === 0) continue;

    const baseSlot = slot % 6;
    const keepMax = depthKeepMax[baseSlot];

    // Grid cells: owner face index and depth
    const gridOwner = new Int32Array(gridSize * gridSize).fill(-1);
    const gridDepth = new Float32Array(gridSize * gridSize);
    const bumped = new Set();

    for (const f of slotFaces) {
      if (bumped.has(f)) continue;

      const u0 = rawU[f*3], u1 = rawU[f*3+1], u2 = rawU[f*3+2];
      const v0 = rawV[f*3], v1 = rawV[f*3+1], v2 = rawV[f*3+2];
      const minGx = Math.max(0, Math.floor(Math.min(u0, u1, u2) * gridSize));
      const maxGx = Math.min(gridSize-1, Math.floor(Math.max(u0, u1, u2) * gridSize));
      const minGy = Math.max(0, Math.floor(Math.min(v0, v1, v2) * gridSize));
      const maxGy = Math.min(gridSize-1, Math.floor(Math.max(v0, v1, v2) * gridSize));

      // Precompute barycentric denominator for point-in-triangle test
      const denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2);
      if (Math.abs(denom) < 1e-10) continue; // degenerate triangle
      const invDenom = 1.0 / denom;

      const fDepth = centroidDepth[f];

      for (let gy = minGy; gy <= maxGy; gy++) {
        for (let gx = minGx; gx <= maxGx; gx++) {
          // Point-in-triangle test at cell center
          const pu = (gx + 0.5) / gridSize;
          const pv = (gy + 0.5) / gridSize;
          const w0 = ((v1 - v2) * (pu - u2) + (u2 - u1) * (pv - v2)) * invDenom;
          const w1 = ((v2 - v0) * (pu - u2) + (u0 - u2) * (pv - v2)) * invDenom;
          const w2 = 1 - w0 - w1;
          if (w0 < -0.01 || w1 < -0.01 || w2 < -0.01) continue;

          const gi = gy * gridSize + gx;
          const owner = gridOwner[gi];
          if (owner === -1) {
            gridOwner[gi] = f;
            gridDepth[gi] = fDepth;
          } else if (owner !== f && !bumped.has(owner)) {
            // True overlap: bump the occluded face
            const ownerDepth = gridDepth[gi];
            let occluded;
            if (keepMax) {
              occluded = (fDepth >= ownerDepth) ? owner : f;
            } else {
              occluded = (fDepth <= ownerDepth) ? owner : f;
            }
            atlasIndex[occluded] = Math.min(atlasIndex[occluded] + 6, 12);
            bumped.add(occluded);
            if (occluded === owner) {
              gridOwner[gi] = f;
              gridDepth[gi] = fDepth;
            }
          }
        }
      }
    }
  }
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
 * @param {Uint8Array} [_faceAssignment] - unused, kept for API compat
 * @returns {{ positions3D: Float32Array, mask: Uint8Array, tbnData: Float32Array }}
 */
export function rasterizeUV(uvs, positions, faces, numFaces, resolution = 1024, _faceAssignment = null) {
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

  // Sub-texel face coverage: for faces whose UV triangle is smaller than
  // 1 texel, write face centroid position to every texel that any vertex
  // would sample. Writes unconditionally (overwrites other faces' data)
  // because the sub-texel face's vertices WILL sample these texels at
  // render time and must get correct colors.
  for (let f = 0; f < numFaces; f++) {
    const i0 = faces[f * 3], i1 = faces[f * 3 + 1], i2 = faces[f * 3 + 2];
    const u0 = uvs[i0*2], v0 = uvs[i0*2+1];
    const u1 = uvs[i1*2], v1 = uvs[i1*2+1];
    const u2 = uvs[i2*2], v2 = uvs[i2*2+1];

    // Check UV triangle area in texels
    const texelArea = Math.abs((u1-u0)*(v2-v0) - (u2-u0)*(v1-v0)) * 0.5 * resolution * resolution;
    if (texelArea >= 1.0) continue; // adequately rasterized, skip

    // Face centroid in 3D
    const p0x = positions[i0*3], p0y = positions[i0*3+1], p0z = positions[i0*3+2];
    const p1x = positions[i1*3], p1y = positions[i1*3+1], p1z = positions[i1*3+2];
    const p2x = positions[i2*3], p2y = positions[i2*3+1], p2z = positions[i2*3+2];
    const cx = (p0x + p1x + p2x) / 3;
    const cy = (p0y + p1y + p2y) / 3;
    const cz = (p0z + p1z + p2z) / 3;

    // Face TBN
    const e1x = p1x-p0x, e1y = p1y-p0y, e1z = p1z-p0z;
    const e2x = p2x-p0x, e2y = p2y-p0y, e2z = p2z-p0z;
    let fnx = e1y*e2z - e1z*e2y, fny = e1z*e2x - e1x*e2z, fnz = e1x*e2y - e1y*e2x;
    const fnLen = Math.sqrt(fnx*fnx + fny*fny + fnz*fnz) || 1;
    fnx /= fnLen; fny /= fnLen; fnz /= fnLen;
    let tx = 1, ty = 0, tz = 0;
    if (Math.abs(fnx) > 0.9) { tx = 0; ty = 1; }
    const tDotN = tx*fnx + ty*fny + tz*fnz;
    tx -= tDotN*fnx; ty -= tDotN*fny; tz -= tDotN*fnz;
    const tLen = Math.sqrt(tx*tx + ty*ty + tz*tz) || 1;
    tx /= tLen; ty /= tLen; tz /= tLen;
    const bx = fny*tz - fnz*ty, by = fnz*tx - fnx*tz, bz = fnx*ty - fny*tx;

    // Write to every unique texel that any vertex or centroid would sample
    const texels = new Set();
    for (const [pu, pv] of [[u0,v0],[u1,v1],[u2,v2],[(u0+u1+u2)/3,(v0+v1+v2)/3]]) {
      const px = Math.min(resolution-1, Math.max(0, Math.floor(pu * resolution)));
      const py = Math.min(resolution-1, Math.max(0, Math.floor(pv * resolution)));
      texels.add(py * resolution + px);
    }

    for (const pixIdx of texels) {
      // Unconditional write — this face's vertices will sample these texels
      mask[pixIdx] = 1;
      positions3D[pixIdx * 3] = cx;
      positions3D[pixIdx * 3 + 1] = cy;
      positions3D[pixIdx * 3 + 2] = cz;
      const tbnBase = pixIdx * 9;
      tbnData[tbnBase] = tx; tbnData[tbnBase+1] = ty; tbnData[tbnBase+2] = tz;
      tbnData[tbnBase+3] = bx; tbnData[tbnBase+4] = by; tbnData[tbnBase+5] = bz;
      tbnData[tbnBase+6] = fnx; tbnData[tbnBase+7] = fny; tbnData[tbnBase+8] = fnz;
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
  _dilateTexture(albedo, mask, resolution, 20);
  _dilateTexture(normalMap, mask, resolution, 20);

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
