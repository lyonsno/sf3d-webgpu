# SF3D WebGPU

Single-image 3D mesh generation running entirely in WebGPU compute shaders. A browser port of Stability AI's [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d).

No server, no Python, no ONNX at inference time. Image in, textured GLB out.

Built on the [Kaminos WebGPU Inference Kit](https://github.com/lyonsno/kaminos/tree/main/webgpu-inference-kit),
whose cooperative scheduling lets long inference share one GPU with a live
WebGPU application — the same runtime spine used by the
[MoGe](https://github.com/lyonsno/moge-webgpu),
[Kimodo](https://github.com/lyonsno/kimodo-webgpu), and
[SHARP](https://github.com/lyonsno/sharp-webgpu) browser ports.

![Input photo of a chair transformed into a generated 3D mesh](docs/assets/hero-before-after.png)

*A single photo becomes a textured, UV-unwrapped GLB in ~33s on an M4 Max — generated entirely by WebGPU compute shaders in the browser. Output above is the bundled `demo_chair.png` example.*

## Quick Start

```bash
# 1. Install JS dependencies
npm install

# 2. Download the model weights (~2.1 GB) into public/
hf download BasinShapers/sf3d-webgpu-weights weights.bin --local-dir public/

# 3. Run
npm run dev            # or: npx vite --port 5177
# Open http://localhost:5177/
# Click "Try Demo (Chair)" or drop any image
# Click "Generate 3D Mesh" (~25-35s on M4 Max)
# Click "Download GLB"
```

### Model weights

Inference needs `public/weights.bin` — a ~2.1 GB fp16 flat binary (gitignored,
never committed). The quickest path is the pre-converted download above from
[`BasinShapers/sf3d-webgpu-weights`](https://huggingface.co/BasinShapers/sf3d-webgpu-weights)
(needs the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):
`pip install -U huggingface_hub`).

**Or build it yourself** from Stability AI's original checkpoint with
`tools/convert_weights.py`, which requires:

- **PyTorch** and the [`stable-fast-3d`](https://github.com/Stability-AI/stable-fast-3d)
  package importable on `PYTHONPATH` (the converter imports `sf3d.system`; point
  it at a checkout with `SF3D_REPO=/path/to/stable-fast-3d`).
- **HuggingFace access** to the gated
  [`stabilityai/stable-fast-3d`](https://huggingface.co/stabilityai/stable-fast-3d)
  model — accept its license and authenticate (`huggingface-cli login`), then:

  ```bash
  python tools/convert_weights.py --output public/weights.bin --dtype fp16
  ```

  Pass `--model-path /local/checkout` to convert from a local copy instead of
  downloading.

Both the hosted weights and the original model are governed by the
[Stability AI Community License](https://huggingface.co/BasinShapers/sf3d-webgpu-weights/blob/main/LICENSE.md)
(free for research, non-commercial, and commercial use under US $1M annual
revenue). **Powered by Stability AI.**

## Production Build

```bash
npm run build
```

Production builds preserve the application and non-model files from `public/`,
but intentionally omit `public/weights.bin` from `dist/`. Deployments must mount
or stage that model file separately at `/weights.bin`; an ordinary local build
must not create another multi-gigabyte copy.

## Smoke Test

```bash
node tools/smoke_inference.mjs --image public/demo_chair.png
```

Produces `/tmp/sf3d-inference-smoke.glb` and a report at `/tmp/sf3d-inference-smoke-report.txt`. Previous smoke outputs are versioned in `/tmp/sf3d-smokes/` for A/B comparison.

To regenerate the README imagery from a GLB:

```bash
node tools/render_glb_hero.mjs --glb /tmp/sf3d-inference-smoke.glb --out docs/assets/hero-chair.png
node tools/compose_before_after.mjs --input public/demo_chair.png \
  --render docs/assets/hero-chair.png --out docs/assets/hero-before-after.png
```

## Pipeline

| Stage | Module | Runs on |
|-------|--------|---------|
| Image preprocessing | `inference.js` | CPU (Web Worker) |
| Camera embedding | `inference.js` | GPU |
| DINOv2 ViT-Large backbone | `sf3d_backbone.js` | GPU (cooperative duties) |
| Two-stream interleave transformer | `two_stream.js` | GPU (cooperative duties) |
| PixelShuffle post-processor | `post_processor.js` | GPU (cooperative duties, bounded-prefix) |
| Triplane query + MaterialMLP decoder | `triplane_decoder.js` | GPU |
| Marching tetrahedra | `marching_tet.js` | CPU (Web Worker) |
| CLIP material estimate | `clip_estimator.js` | CPU prep (Web Worker) + GPU |
| UV unwrap (PCA + cube projection + BVH overlap) | `texture_baker.js` | CPU (Web Worker) |
| Texture bake (triplane query per texel) | `texture_baker.js` | GPU (cooperative duties) + CPU materialization (Web Worker) |
| GLB export | `texture_baker.js` | CPU |

14 WGSL compute shaders (shared from MOGE port) + 5 inline shaders in `triplane_decoder.js`.

## Kaminos WebGPU Inference Kit

SF3D WebGPU is a model port of the
[Kaminos WebGPU Inference Kit](https://github.com/lyonsno/kaminos/tree/main/webgpu-inference-kit)
([`@kaminos/webgpu-inference-kit`](https://www.npmjs.com/package/@kaminos/webgpu-inference-kit)
on npm) — a shared runtime that gives browser model ports a common session and
device lifecycle, persistent model routes, queued invocations, cooperative
scheduling, and runtime telemetry, while the port keeps ownership of its
weights, kernels, and output construction.

What the kit buys this port:

- **Cooperative execution.** The DINO backbone, two-stream transformer,
  post-processor, and texture bake are decomposed into bounded GPU duties the
  kit can schedule cooperatively, so a live application keeps rendering on the
  same device while a mesh generates.
- **Scheduling-independent output.** The final GLB is byte-identical across the
  monolithic and cooperative scheduling paths (see the
  [deterministic output receipt](#deterministic-output-receipt)).
- **Runtime primitives.** Kit resource caches compile the CLIP estimator's
  inline kernels once per device, and the kit's numerical-parity comparison
  drives the PyTorch parity tooling (`tools/smoke_parity.mjs`).
- **Route identity and receipts.** Inference runs under a registered route ID
  with kit-validated route receipts and staged-submit profiles, so what
  actually executed — backend, stages, kit version — is inspectable rather
  than assumed.

Sibling ports on the same runtime: [MoGe](https://github.com/lyonsno/moge-webgpu)
(depth, normals, and point maps from one image),
[Kimodo](https://github.com/lyonsno/kimodo-webgpu) (browser motion diffusion),
and [SHARP](https://github.com/lyonsno/sharp-webgpu) (long image-to-splat
inference). The kit itself lives in the
[Kaminos](https://github.com/lyonsno/kaminos) browser-native workbench.

## Foreground liveness

The app runs the **product route**: every long GPU stage executes as
cooperative duties through the kit, and the five heavy CPU stages (image
preprocessing, CLIP prep, marching tetrahedra, UV unwrap, texture
materialization) run on Web Workers, so the page — and any other WebGPU work
sharing the GPU — keeps its frame cadence while a mesh generates. The image
buffer upload and UV rasterization stay on the main thread; each stays under a
frame. This is the default in `src/main.js`, composed
in one place by [`src/lib/product_route.js`](src/lib/product_route.js):

| Stage | Mechanism |
|-------|-----------|
| Image preprocessing (Lanczos resize, blend, normalize) | Web Worker |
| DINOv2 ViT-Large | 24 cooperative GPU duties (one per block) |
| Two-stream transformer | cooperative attention-tile duties, 256 linear rows per duty (2,922 duties) |
| PixelShuffle post-processor | 702 cooperative channel-range duties, bounded-prefix completion, depth 2 |
| Marching tetrahedra | Web Worker owning the resident tet grid |
| CLIP material estimate | blend / resize / patch-embed on a Web Worker; transformer on GPU with kit-cached pipelines |
| UV unwrap | Web Worker |
| Texture bake | 4,096-texel cooperative GPU duties over a scratch arena; albedo/normal materialization on a Web Worker |

Measured with [`tools/smoke_product_route.mjs`](tools/smoke_product_route.mjs)
on an M4 Max in Chrome (`apple/metal-3`), same commit and `demo_chair.png`,
requestAnimationFrame intervals scoped to the inference window. `--contend`
adds a same-page WebGPU contender on a second `GPUDevice` of the same GPU that
submits compute work continuously, so "smooth" means smooth while sharing the
GPU:

| Route | Wall | Frame intervals | p95 / p99 | Max gap | > 33.3 ms | Contender submissions |
|-------|------|-----------------|-----------|---------|-----------|-----------------------|
| Single submit, CPU stages on the main thread | 22.3 s | 2,563 | 10.1 / 10.3 ms | 409.9 ms | 4 | — |
| Single submit + contender | 21.9 s | 2,545 | 10.0 / 10.3 ms | 199.9 ms | 4 | 11,635 |
| **Product route (default)** | 39.9 s | 4,741 | 9.8 / 10.2 ms | **26.4 ms** | **0** | — |
| **Product route + contender** | 39.8 s | 4,768 | 9.9 / 10.3 ms | **49.4 ms** | **1** | **72,097** |

The whole-route tail drops from hundreds of milliseconds (image preprocess,
CLIP, UV unwrap, texture bake) to under two frames, and the co-tenant on the
same GPU completes about 3.4× more submissions per second beside the product
route (1,811/s vs 531/s), because the two-stream stage no longer monopolizes
the GPU between submissions. The cost is wall time: the fine two-stream duties
make the route about 1.8× longer than the single-submit path. The GLB is
byte-identical in every row. This is a scheduling and output-identity witness,
not a new PyTorch parity measurement (see the next section). Receipts:
[`smoke-receipts/product-route-witness-*_2bebf7d.json`](smoke-receipts/).

```bash
npm run smoke:product-route                      # product route, idle page
npm run smoke:product-route -- --contend         # with a same-page WebGPU contender
npm run smoke:product-route -- --arm monolithic  # the single-submit baseline
```

## Numerical Match to PyTorch

Measured against the original PyTorch pipeline on the bundled `demo_chair.png`:

- Vertex count: 9988 vs 10008 (99.8%)
- Density at known inside vertices: within 4%
- SDF max: 28.34 vs 28.58
- Textured output matches the PyTorch reference render under side-by-side visual inspection
- Remaining gap is fp16 precision + Lanczos resize interpolation difference

### Deterministic output receipt

The final GLB is **bit-for-bit reproducible**. A clean `npm install` → convert
weights → run produces, for `demo_chair.png`:

```
SHA-256(demo_chair GLB) = e1f70de3407df24d571bf68f70fac2b59373bdd948075a2387f1834e4faff8b7
9988 vertices · 19976 faces · ~33s end-to-end (M4 Max)
```

The same hash is emitted by both scheduling paths (monolithic and
arena-plus-worker) across the A/B/C/D product harness — the output is
independent of GPU-duty scheduling. The cooperative post-processor also passes
[`@kaminos/webgpu-inference-kit`](https://github.com/lyonsno/kaminos/tree/main/webgpu-inference-kit)'s
`validateWebGpuCooperativeExecutionReport` against the exact 702-duty
bounded-prefix contract. Receipts are versioned under
[`smoke-receipts/`](smoke-receipts/).

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
    product_route.js      Product default composition (cooperative duties + workers)
    full_pipeline.js      The complete route as one callable (app + harnesses)
    clip_prep_core.js     CLIP CPU prep (pure); clip_prep_worker.js runs it
    marching_tet_worker.js  Marching tetrahedra on a worker owning the tet grid
    preprocess_worker.js / uv_unwrap_worker.js / materialize_worker.js  CPU offloads
    gpu.js                WebGPU initialization + buffer helpers
    shader_ops.js         Shared shader dispatch helpers
  main.js                 Browser UI wiring
  shaders/                WGSL compute shaders
tools/
  convert_weights.py      PyTorch -> flat binary fp16 weight converter
  smoke_inference.mjs     Puppeteer-driven browser smoke test
  smoke_product_route.mjs   Foreground-liveness witness (rAF probe, contender, receipts)
  smoke_parity.mjs        Per-stage PyTorch parity via the kit's parity primitives
  render_glb_hero.mjs     Render a GLB to a hero still (model-viewer)
  compose_before_after.mjs  Compose input-photo / generated-mesh banner
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
| 2 | 2026-06-28 | End-to-end pipeline, 6 bug fixes, coherent mesh, first full visual validation |
| 3 | 2026-06-29 | Texture baking, normal maps, smooth normals, GLB export, 2 reviews |
| 4 | 2026-06-30 | UV atlas splitting: bbox normalization, overlap detection, sub-texel fixes |
| 5 | 2026-07-01 | Tangent UV rotation, PyTorch-matching axes, PCA alignment, BVH overlap detection, visual parity |
| 6–11 | 2026-07 | Cooperative WebGPU execution: DINO/two-stream/post-processor duty decomposition, scratch-arena + worker offload, bounded-prefix scheduling, `@kaminos/webgpu-inference-kit` conformance, byte-identical output across scheduling paths |
| 12 | 2026-09-15 | Kit 0.1.48; product route composed by default (every cooperative boundary + five workers); two-stream submit-contract fix; CLIP-prep and marching-tet workers; four-arm liveness witness with same-page contender; kit parity primitives in the parity tooling |

## License

This is a port of [Stability AI's Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) for research and educational purposes. See the original repository for license terms.
