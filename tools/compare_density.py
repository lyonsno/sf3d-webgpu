#!/usr/bin/env python3
"""Compare PyTorch vs WebGPU density distributions.

Runs SF3D inference on PyTorch MPS, captures raw density values at the
isosurface grid vertices, and reports statistics for comparison with
the WebGPU diagnostic output.
"""

import sys
import os
import numpy as np
import torch

SF3D_REPO = os.environ.get("SF3D_REPO", os.path.expanduser("~/dev/sf3d"))
sys.path.insert(0, SF3D_REPO)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sf3d.system import SF3D
from sf3d.models.utils import scale_tensor
from sf3d.utils import default_cond_c2w, create_intrinsic_from_fov_deg
from PIL import Image


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png")

    print(f"Loading SF3D model...")
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.eval()
    device = model.device

    print(f"Running inference on: {image_path}")
    image = Image.open(image_path)

    with torch.no_grad():
        mask_cond, rgb_cond = model.prepare_image(image)
        c2w_cond = default_cond_c2w(model.cfg.default_distance).to(device)
        intrinsic, intrinsic_normed_cond = create_intrinsic_from_fov_deg(
            model.cfg.default_fovy_deg, model.cfg.cond_image_size, model.cfg.cond_image_size)
        batch = {
            "rgb_cond": rgb_cond.unsqueeze(0).to(device),
            "mask_cond": mask_cond.unsqueeze(0).to(device),
            "c2w_cond": c2w_cond.view(1, 1, 4, 4),
            "intrinsic_cond": intrinsic.to(device).view(1, 1, 3, 3),
            "intrinsic_normed_cond": intrinsic_normed_cond.to(device).view(1, 1, 3, 3),
        }

        # Get scene codes (triplane features)
        scene_codes, direct_codes = model.get_scene_codes(batch)
        print(f"Scene codes shape: {scene_codes.shape}")  # [1, 3, 40, 384, 384]

        # Query triplane at isosurface grid vertices
        triplane = scene_codes[0]  # [3, 40, 384, 384]
        grid_vertices = scale_tensor(
            model.isosurface_helper.grid_vertices.to(device),
            model.isosurface_helper.points_range,
            model.bbox,
        )
        print(f"Grid vertices: {grid_vertices.shape}")  # [N, 3]

        # Query triplane features
        values = model.query_triplane(grid_vertices, triplane)
        print(f"Triplane query output: {values.shape}")  # [N, 120]

        # Decode - get raw density before and after activation
        # Run just the density head manually to see intermediate values
        density_head = model.decoder.heads["density"]
        raw_density_input = values.squeeze(0)
        x = raw_density_input
        for i, layer in enumerate(density_head):
            x = layer(x)
            if i == len(list(density_head)) - 2:  # before last activation
                print(f"Density MLP output (before trunc_exp): min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")

        # Get the full decoded output
        decoded = model.decoder(values, include=["density", "vertex_offset"])
        density = decoded["density"]
        print(f"\nDensity (after trunc_exp + bias): shape={density.shape}")
        print(f"  min={density.min().item():.4f}, max={density.max().item():.4f}")
        print(f"  mean={density.mean().item():.4f}, std={density.std().item():.4f}")

        sdf = density - 10.0
        n_inside = (sdf > 0).sum().item()
        n_total = sdf.numel()
        print(f"\nSDF (density - 10.0):")
        print(f"  min={sdf.min().item():.4f}, max={sdf.max().item():.4f}")
        print(f"  inside (>0): {n_inside}/{n_total} ({100*n_inside/n_total:.1f}%)")

        # Also check triplane feature statistics
        print(f"\nTriplane features:")
        print(f"  min={values.min().item():.4f}, max={values.max().item():.4f}")
        print(f"  mean={values.mean().item():.4f}, std={values.std().item():.4f}")

        # Scene code statistics
        print(f"\nScene codes (post-processor output):")
        print(f"  min={scene_codes.min().item():.4f}, max={scene_codes.max().item():.4f}")
        print(f"  mean={scene_codes.mean().item():.4f}, std={scene_codes.std().item():.4f}")

        # Direct codes (backbone output before post-processor)
        print(f"\nDirect codes (backbone output):")
        print(f"  min={direct_codes.min().item():.4f}, max={direct_codes.max().item():.4f}")
        print(f"  mean={direct_codes.mean().item():.4f}, std={direct_codes.std().item():.4f}")


if __name__ == "__main__":
    main()
