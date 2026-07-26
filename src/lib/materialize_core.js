/**
 * materialize_core.js — pure, DOM/GPU-free texture materialization.
 *
 * Extracted from bakeTexture so the exact same albedo build + normal-map
 * (default fill + TBN world→tangent transform) + dilation runs on either the
 * main thread OR a Web Worker, byte-identical. This is Cranial's assay's ~752ms
 * single-threaded CPU tail (albedo dilation ~375ms + normal dilation ~369ms);
 * moving it to a worker removes it from the main thread with no GPU sync and no
 * per-duty fence floor.
 *
 * Inputs are plain typed arrays; outputs are the albedo + normal Uint8Arrays.
 */

/**
 * Dilate texture to fill empty pixels by averaging nearest occupied neighbors.
 * Exact copy of bakeTexture's _dilateTexture (must stay byte-identical).
 */
export function dilateTexture(texture, mask, resolution, iterations = 6) {
  const workMask = new Uint8Array(mask);

  for (let iter = 0; iter < iterations; iter++) {
    const newPixels = [];

    for (let y = 0; y < resolution; y++) {
      for (let x = 0; x < resolution; x++) {
        const idx = y * resolution + x;
        if (workMask[idx]) continue;

        let sumR = 0, sumG = 0, sumB = 0, count = 0;
        const neighbors = [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]];
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
 * Build albedo + normal-map textures from decoded features/normals, transform
 * normals to tangent space, and dilate. Byte-identical to bakeTexture's inline
 * materialization.
 *
 * @param {object} o
 * @param {Float32Array} o.featuresCPU     [numOccupied*3] decoded albedo features
 * @param {Float32Array} o.normalsCPU      [numOccupied*3] decoded world-space normals
 * @param {Uint32Array}  o.occupiedIndices [numOccupied] texel indices
 * @param {Float32Array} o.tbnData         [resolution^2 * 9] per-texel TBN basis
 * @param {Uint8Array}   o.mask            [resolution^2] occupancy mask
 * @param {number}       o.resolution
 * @param {number}       o.numOccupied
 * @returns {{ albedo: Uint8Array, normalMap: Uint8Array }}
 */
export function materializeTextures({ featuresCPU, normalsCPU, occupiedIndices, tbnData, mask, resolution, numOccupied }) {
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
    normalMap[i * 4 + 2] = 255; // blue channel = 1.0
    normalMap[i * 4 + 3] = 255;
  }

  for (let i = 0; i < numOccupied; i++) {
    const texIdx = occupiedIndices[i];
    const tbnBase = texIdx * 9;

    const nx = normalsCPU[i * 3];
    const ny = normalsCPU[i * 3 + 1];
    const nz = normalsCPU[i * 3 + 2];

    const tx = tbnData[tbnBase], ty = tbnData[tbnBase + 1], tz = tbnData[tbnBase + 2];
    const bx = tbnData[tbnBase + 3], by = tbnData[tbnBase + 4], bz = tbnData[tbnBase + 5];
    const fnx = tbnData[tbnBase + 6], fny = tbnData[tbnBase + 7], fnz = tbnData[tbnBase + 8];

    // Transform to tangent space: n_tangent = TBN^T * n_world
    const ntx = tx * nx + ty * ny + tz * nz;
    const nty = bx * nx + by * ny + bz * nz;
    const ntz = fnx * nx + fny * ny + fnz * nz;

    // Encode from [-1,1] to [0,1]
    const r = Math.max(0, Math.min(255, Math.round((ntx * 0.5 + 0.5) * 255)));
    const g = Math.max(0, Math.min(255, Math.round((nty * 0.5 + 0.5) * 255)));
    const b = Math.max(0, Math.min(255, Math.round((ntz * 0.5 + 0.5) * 255)));

    normalMap[texIdx * 4] = r;
    normalMap[texIdx * 4 + 1] = g;
    normalMap[texIdx * 4 + 2] = b;
    normalMap[texIdx * 4 + 3] = 255;
  }

  // Dilate both textures (matching PyTorch: resolution // 150 ≈ 7 at 1024)
  const dilateIters = Math.max(1, Math.round(resolution / 150));
  dilateTexture(albedo, mask, resolution, dilateIters);
  dilateTexture(normalMap, mask, resolution, dilateIters);

  return { albedo, normalMap };
}
