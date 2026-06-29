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
 * Vertices are duplicated per-face (no sharing across faces) to allow
 * per-face UVs without seam issues.
 *
 * @param {Float32Array} vertices - [N_v * 3] vertex positions
 * @param {Uint32Array} faces - [N_f * 3] triangle indices
 * @param {number} numVertices
 * @param {number} numFaces
 * @returns {{ uvs: Float32Array, newVertices: Float32Array, newFaces: Uint32Array, newNumVertices: number, newNumFaces: number }}
 */
export function unwrapUV(vertices, faces, numVertices, numFaces) {
  // Compute face normals and assign each face to a cube face
  const faceAssignment = new Uint8Array(numFaces); // 0-5: +X,-X,+Y,-Y,+Z,-Z

  for (let f = 0; f < numFaces; f++) {
    const i0 = faces[f * 3], i1 = faces[f * 3 + 1], i2 = faces[f * 3 + 2];
    const v0x = vertices[i0*3], v0y = vertices[i0*3+1], v0z = vertices[i0*3+2];
    const v1x = vertices[i1*3], v1y = vertices[i1*3+1], v1z = vertices[i1*3+2];
    const v2x = vertices[i2*3], v2y = vertices[i2*3+1], v2z = vertices[i2*3+2];

    // Cross product = face normal
    const e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    const e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;

    // Assign to dominant axis
    const ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz);
    if (ax >= ay && ax >= az) {
      faceAssignment[f] = nx > 0 ? 0 : 1;
    } else if (ay >= ax && ay >= az) {
      faceAssignment[f] = ny > 0 ? 2 : 3;
    } else {
      faceAssignment[f] = nz > 0 ? 4 : 5;
    }
  }

  // Build per-face-vertex UVs (each face vertex gets its own UV)
  const newNumVertices = numFaces * 3;
  const newNumFaces = numFaces;
  const newVertices = new Float32Array(newNumVertices * 3);
  const newFaces = new Uint32Array(newNumFaces * 3);
  const uvs = new Float32Array(newNumVertices * 2);

  const radius = 0.87;

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

      // Copy position
      newVertices[newIdx * 3] = vertices[origIdx * 3];
      newVertices[newIdx * 3 + 1] = vertices[origIdx * 3 + 1];
      newVertices[newIdx * 3 + 2] = vertices[origIdx * 3 + 2];

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

  return { uvs, newVertices, newFaces, newNumVertices, newNumFaces };
}

/**
 * Rasterize UV space to get 3D positions at each texel.
 *
 * For each triangle in UV space, rasterize its bounding box and compute
 * barycentric interpolation of 3D positions.
 *
 * @param {Float32Array} uvs - [N_v * 2] UV coordinates
 * @param {Float32Array} positions - [N_v * 3] vertex positions
 * @param {Uint32Array} faces - [N_f * 3] face indices
 * @param {number} numFaces
 * @param {number} resolution - texture resolution (default 1024)
 * @returns {{ positions3D: Float32Array, mask: Uint8Array }}
 */
export function rasterizeUV(uvs, positions, faces, numFaces, resolution = 1024) {
  const positions3D = new Float32Array(resolution * resolution * 3);
  const mask = new Uint8Array(resolution * resolution);

  for (let f = 0; f < numFaces; f++) {
    const i0 = faces[f * 3], i1 = faces[f * 3 + 1], i2 = faces[f * 3 + 2];

    const u0 = uvs[i0 * 2], v0 = uvs[i0 * 2 + 1];
    const u1 = uvs[i1 * 2], v1 = uvs[i1 * 2 + 1];
    const u2 = uvs[i2 * 2], v2 = uvs[i2 * 2 + 1];

    const p0x = positions[i0*3], p0y = positions[i0*3+1], p0z = positions[i0*3+2];
    const p1x = positions[i1*3], p1y = positions[i1*3+1], p1z = positions[i1*3+2];
    const p2x = positions[i2*3], p2y = positions[i2*3+1], p2z = positions[i2*3+2];

    // Bounding box in pixel coords
    const minPx = Math.max(0, Math.floor(Math.min(u0, u1, u2) * resolution));
    const maxPx = Math.min(resolution - 1, Math.ceil(Math.max(u0, u1, u2) * resolution));
    const minPy = Math.max(0, Math.floor(Math.min(v0, v1, v2) * resolution));
    const maxPy = Math.min(resolution - 1, Math.ceil(Math.max(v0, v1, v2) * resolution));

    const denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2);
    if (Math.abs(denom) < 1e-10) continue; // degenerate
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
        }
      }
    }
  }

  return { positions3D, mask };
}

/**
 * Bake texture colors by querying the triplane at rasterized 3D positions.
 *
 * @param {GPUDevice} device
 * @param {Object} triplaneDecoder - TriplaneDecoder instance
 * @param {GPUBuffer} triplanesBuf - [3, 40, 384, 384] triplane features
 * @param {Object} decoderWeights - decoder weights
 * @param {Float32Array} positions3D - [res*res, 3] from rasterizeUV
 * @param {Uint8Array} mask - [res*res] from rasterizeUV
 * @param {number} resolution
 * @returns {Uint8Array} - [res, res, 4] RGBA texture
 */
export async function bakeTexture(device, triplaneDecoder, triplanesBuf, decoderWeights,
                                   positions3D, mask, resolution = 1024) {
  // Collect occupied texel positions
  const occupiedIndices = [];
  for (let i = 0; i < resolution * resolution; i++) {
    if (mask[i]) occupiedIndices.push(i);
  }

  const numOccupied = occupiedIndices.length;
  console.log(`Texture bake: ${numOccupied} occupied texels out of ${resolution * resolution}`);

  if (numOccupied === 0) {
    return new Uint8Array(resolution * resolution * 4);
  }

  // Pack occupied positions into a dense array
  const queryPositions = new Float32Array(numOccupied * 3);
  for (let i = 0; i < numOccupied; i++) {
    const idx = occupiedIndices[i];
    queryPositions[i * 3] = positions3D[idx * 3];
    queryPositions[i * 3 + 1] = positions3D[idx * 3 + 1];
    queryPositions[i * 3 + 2] = positions3D[idx * 3 + 2];
  }

  // Upload to GPU and decode features (RGB colors)
  const queryPosBuf = createStorageBuffer(device, queryPositions);

  const encoder = device.createCommandEncoder();
  const decoded = triplaneDecoder.decode(
    encoder, queryPosBuf, triplanesBuf, numOccupied,
    decoderWeights, ['features']);
  device.queue.submit([encoder.finish()]);

  // Read back RGB values (sigmoid already applied by decoder)
  const featuresCPU = await readBuffer(device, decoded.features, numOccupied * 3 * 4);

  // Build RGBA texture
  const texture = new Uint8Array(resolution * resolution * 4);
  for (let i = 0; i < numOccupied; i++) {
    const texIdx = occupiedIndices[i];
    texture[texIdx * 4] = Math.max(0, Math.min(255, Math.round(featuresCPU[i * 3] * 255)));
    texture[texIdx * 4 + 1] = Math.max(0, Math.min(255, Math.round(featuresCPU[i * 3 + 1] * 255)));
    texture[texIdx * 4 + 2] = Math.max(0, Math.min(255, Math.round(featuresCPU[i * 3 + 2] * 255)));
    texture[texIdx * 4 + 3] = 255;
  }

  // Dilate to fill empty texels near edges
  _dilateTexture(texture, mask, resolution, 6);

  return texture;
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
 * @param {Uint32Array} faces - [N_f * 3]
 * @param {Float32Array} uvs - [N_v * 2]
 * @param {Uint8Array} texture - [res, res, 4] RGBA
 * @param {number} numVertices
 * @param {number} numFaces
 * @param {number} textureResolution
 * @param {number} roughness
 * @param {number} metallic
 * @returns {ArrayBuffer} GLB binary
 */
export async function exportGLB(vertices, faces, uvs, texture,
                                 numVertices, numFaces, textureResolution = 1024,
                                 roughness = 0.5, metallic = 0.0) {
  // Apply coordinate transforms to match glTF conventions
  // SF3D: X-right, Y-up, Z-forward → glTF: X-right, Y-up, Z-backward
  // Match PyTorch: rot(-90, X) then rot(90, Y) then invert faces
  const transformedVerts = new Float32Array(numVertices * 3);
  for (let i = 0; i < numVertices; i++) {
    const x = vertices[i * 3];
    const y = vertices[i * 3 + 1];
    const z = vertices[i * 3 + 2];
    // rot(-90, X): (x, y, z) → (x, z, -y)
    const rx = x, ry = z, rz = -y;
    // rot(+90, Y): (x, y, z) → (z, y, -x)
    transformedVerts[i * 3] = rz;
    transformedVerts[i * 3 + 1] = ry;
    transformedVerts[i * 3 + 2] = -rx;
  }

  // Invert face winding (PyTorch mesh.invert())
  const invertedFaces = new Uint32Array(numFaces * 3);
  for (let f = 0; f < numFaces; f++) {
    invertedFaces[f * 3] = faces[f * 3];
    invertedFaces[f * 3 + 1] = faces[f * 3 + 2];
    invertedFaces[f * 3 + 2] = faces[f * 3 + 1];
  }

  // Encode texture as JPEG
  const texBlob = await _textureToJPEG(texture, textureResolution);
  const texBytes = new Uint8Array(await texBlob.arrayBuffer());

  // Compute bounding box
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < numVertices; i++) {
    const x = transformedVerts[i*3], y = transformedVerts[i*3+1], z = transformedVerts[i*3+2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }

  // Pad to 4-byte alignment
  const pad4 = (n) => (n + 3) & ~3;

  const vertexBytes = new Uint8Array(transformedVerts.buffer, transformedVerts.byteOffset, transformedVerts.byteLength);
  const indexBytes = new Uint8Array(invertedFaces.buffer, invertedFaces.byteOffset, invertedFaces.byteLength);
  const uvBytes = new Uint8Array(uvs.buffer, uvs.byteOffset, uvs.byteLength);

  const vertexLen = pad4(vertexBytes.byteLength);
  const indexLen = pad4(indexBytes.byteLength);
  const uvLen = pad4(uvBytes.byteLength);
  const texLen = pad4(texBytes.byteLength);
  const totalBinLen = vertexLen + indexLen + uvLen + texLen;

  const bufferViews = [
    { buffer: 0, byteOffset: 0, byteLength: vertexBytes.byteLength, target: 34962 },
    { buffer: 0, byteOffset: vertexLen, byteLength: indexBytes.byteLength, target: 34963 },
    { buffer: 0, byteOffset: vertexLen + indexLen, byteLength: uvBytes.byteLength, target: 34962 },
    { buffer: 0, byteOffset: vertexLen + indexLen + uvLen, byteLength: texBytes.byteLength },
  ];

  const accessors = [
    { bufferView: 0, componentType: 5126, count: numVertices, type: 'VEC3',
      min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
    { bufferView: 1, componentType: 5125, count: numFaces * 3, type: 'SCALAR' },
    { bufferView: 2, componentType: 5126, count: numVertices, type: 'VEC2' },
  ];

  const gltf = {
    asset: { version: '2.0', generator: 'SF3D-WebGPU' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      primitives: [{
        attributes: { POSITION: 0, TEXCOORD_0: 2 },
        indices: 1,
        material: 0,
      }],
    }],
    materials: [{
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0 },
        roughnessFactor: roughness,
        metallicFactor: metallic,
      },
    }],
    textures: [{ source: 0, sampler: 0 }],
    images: [{ bufferView: 3, mimeType: 'image/jpeg' }],
    samplers: [{ magFilter: 9729, minFilter: 9987 }],
    accessors,
    bufferViews,
    buffers: [{ byteLength: totalBinLen }],
  };

  const jsonStr = JSON.stringify(gltf);
  const jsonBytes = new TextEncoder().encode(jsonStr);
  const jsonPadLen = pad4(jsonBytes.byteLength);

  const glbLen = 12 + 8 + jsonPadLen + 8 + totalBinLen;
  const glb = new ArrayBuffer(glbLen);
  const view = new DataView(glb);
  const bytes = new Uint8Array(glb);

  // Header
  view.setUint32(0, 0x46546C67, true); // "glTF"
  view.setUint32(4, 2, true);
  view.setUint32(8, glbLen, true);

  // JSON chunk
  let offset = 12;
  view.setUint32(offset, jsonPadLen, true);
  view.setUint32(offset + 4, 0x4E4F534A, true); // "JSON"
  offset += 8;
  bytes.set(jsonBytes, offset);
  for (let i = jsonBytes.byteLength; i < jsonPadLen; i++) bytes[offset + i] = 0x20;
  offset += jsonPadLen;

  // BIN chunk
  view.setUint32(offset, totalBinLen, true);
  view.setUint32(offset + 4, 0x004E4942, true); // "BIN\0"
  offset += 8;

  bytes.set(vertexBytes, offset); offset += vertexLen;
  bytes.set(indexBytes, offset); offset += indexLen;
  bytes.set(uvBytes, offset); offset += uvLen;
  bytes.set(texBytes, offset);

  return glb;
}

function _textureToJPEG(texture, resolution) {
  const canvas = document.createElement('canvas');
  canvas.width = resolution;
  canvas.height = resolution;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(resolution, resolution);
  imgData.data.set(texture);
  ctx.putImageData(imgData, 0, 0);
  return new Promise((resolve) => {
    canvas.toBlob(resolve, 'image/jpeg', 0.92);
  });
}
