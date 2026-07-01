# SF3D WebGPU

Single-image 3D mesh generation running entirely in WebGPU compute shaders. A browser port of Stability AI's [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d).

No server, no Python, no ONNX. Image in, textured GLB out.

## Quick Start

```bash
# Convert weights (requires PyTorch + SF3D checkout)
python tools/convert_weights.py /path/to/sf3d/checkpoint weights.bin
cp weights.bin public/

# Run
npx vite --port 5177
# Open http://localhost:5177/
# Click "Try Demo (Chair)" or drop any image
# Click "Generate 3D Mesh" (~25-35s on M4 Max)
# Click "Download GLB"
```

Weights must be at `public/weights.bin` (1.886 GB fp16, gitignored).

## Smoke Test

```bash
node tools/smoke_inference.mjs --image public/demo_chair.png
```

Produces `/tmp/sf3d-inference-smoke.glb` and a report at `/tmp/sf3d-inference-smoke-report.txt`. Previous smoke outputs are versioned in `/tmp/sf3d-smokes/` for A/B comparison.

## Pipeline

| Stage | Module | Runs on |
|-------|--------|---------|
| Image preprocessing | `inference.js` | CPU |
| Camera embedding | `inference.js` | GPU |
| DINOv2 ViT-Large backbone | `sf3d_backbone.js` | GPU |
| Two-stream interleave transformer | `two_stream.js` | GPU |
| PixelShuffle post-processor | `post_processor.js` | GPU |
| Triplane query + MaterialMLP decoder | `triplane_decoder.js` | GPU |
| Marching tetrahedra | `marching_tet.js` | CPU |
| UV unwrap (PCA + cube projection + BVH overlap) | `texture_baker.js` | CPU |
| Texture bake (triplane query per texel) | `texture_baker.js` | GPU |
| GLB export | `texture_baker.js` | CPU |

14 WGSL compute shaders (shared from MOGE port) + 5 inline shaders in `triplane_decoder.js`.

## Numerical Match to PyTorch

- Vertex count: 9988 vs 10008 (99.8%)
- Density at known inside vertices: within 4%
- SDF max: 28.34 vs 28.58
- Visual texture parity with PyTorch reference (operator-confirmed 2026-07-01)
- Remaining gap is fp16 precision + Lanczos resize interpolation difference

## Architecture

```
src/
  lib/
    inference.js          Pipeline orchestration, preprocessing, camera embed
    sf3d_backbone.js      DINOv2 ViT-Large with AdaNorm modulation
    two_stream.js         TwoStreamInterleaveTransformer backbone
    post_processor.js     PixelShuffle post-processor
    triplane_decoder.js   Triplane query + MaterialMLP decoder
    marching_tet.js       CPU marching tetrahedra mesh extraction
    texture_baker.js      UV unwrap, rasterize, bake albedo+normal, GLB export
    weights.js            Weight file loader with tensor name mapping
    gpu.js                WebGPU initialization + buffer helpers
    shader_ops.js         Shared shader dispatch helpers
  main.js                 Browser UI wiring
  shaders/                WGSL compute shaders
tools/
  convert_weights.py      PyTorch -> flat binary fp16 weight converter
  smoke_inference.mjs     Puppeteer-driven browser smoke test
  compare_density.py      PyTorch reference density comparison
  evidence/               Durable smoke artifacts
public/
  tets/                   Marching tetrahedra grid data
  demo_chair.png          Demo input image
```

## UV Unwrap Pipeline

The texture baker implements SF3D's cube-projection UV unwrapper with:

1. **PCA alignment** — rotates vertex positions so principal axes align with canonical X/Y/Z (Jacobi eigendecomposition, matching PyTorch `_align_mesh_with_main_axis`)
2. **Cube projection** — assigns faces to 6 cube faces by normal direction, projects matching PyTorch axis conventions
3. **Tangent-aligned UV rotation** — rotates UVs per cube face to align with canonical tangent direction (matching PyTorch `_rotate_uv_slices_consistent_space`)
4. **BVH overlap detection** — triangle-triangle intersection via Sutherland-Hodgman polygon clipping with area threshold, replacing initial grid-based approach
5. **Three-tier atlas packing** — primary (3x2 grid), secondary (3x2 half-size), remaining (per-face sub-cells)
6. **Conditional sub-texel coverage** — fills unoccupied texels for sub-texel faces without overwriting correctly-rasterized data

## Development History

| Session | Date | Key deliverables |
|---------|------|-----------------|
| 1 | 2026-06-27 | MPS bring-up, initial scaffold, DINOv2+backbone dispatch, weight converter |
| 2 | 2026-06-28 | End-to-end pipeline, 6 bug fixes, coherent mesh, operator smoke pass |
| 3 | 2026-06-29 | Texture baking, normal maps, smooth normals, GLB export, 2 reviews |
| 4 | 2026-06-30 | UV atlas splitting: bbox normalization, overlap detection, sub-texel fixes |
| 5 | 2026-07-01 | Tangent UV rotation, PyTorch-matching axes, PCA alignment, BVH overlap detection, visual parity |

## License

This is a port of [Stability AI's Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) for research and educational purposes. See the original repository for license terms.
