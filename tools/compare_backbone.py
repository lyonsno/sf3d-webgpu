#!/usr/bin/env python3
"""Compare PyTorch backbone intermediate values for WebGPU debugging.

Captures values at each stage of the two-stream backbone:
- GroupNorm output
- Triplane projection
- Image projection
- Latent projection
- After each of the 4 main blocks
- proj_out
- Final output (with residual)
"""

import sys
import os
import numpy as np
import torch

SF3D_REPO = os.environ.get("SF3D_REPO", os.path.expanduser("~/dev/sf3d"))
sys.path.insert(0, SF3D_REPO)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sf3d.system import SF3D
from sf3d.utils import default_cond_c2w, create_intrinsic_from_fov_deg
from PIL import Image


def stats(name, t):
    t = t.float()
    print(f"  {name}: min={t.min().item():.4f}, max={t.max().item():.4f}, "
          f"mean={t.mean().item():.4f}, std={t.std().item():.4f}, shape={list(t.shape)}")


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
    backbone = model.backbone

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

        # Get camera embeddings
        camera_embeds = model.camera_embedder(**batch)
        print(f"\n=== Camera Embedder ===")
        stats("camera_embeds", camera_embeds)

        # Get image tokens from DINOv2
        from einops import rearrange
        rgb = batch["rgb_cond"]
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(1)
            batch["mask_cond"] = batch["mask_cond"].unsqueeze(1)
            batch["c2w_cond"] = batch["c2w_cond"]
            batch["intrinsic_cond"] = batch["intrinsic_cond"]
            batch["intrinsic_normed_cond"] = batch["intrinsic_normed_cond"]
        input_image_tokens = model.image_tokenizer(
            rearrange(rgb, "B Nv H W C -> B Nv C H W"),
            modulation_cond=camera_embeds,
        )
        input_image_tokens = rearrange(input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1)
        print(f"\n=== Image Tokens (DINOv2 output) ===")
        stats("image_tokens", input_image_tokens)

        # Get tokenizer embeddings (raw)
        hidden_states = model.tokenizer(1)  # [1, 1024, 27648]
        print(f"\n=== Tokenizer Output (rearranged) ===")
        stats("hidden_states", hidden_states)

        # Now trace through backbone manually
        print(f"\n=== Backbone Stages ===")
        encoder_hidden_states = input_image_tokens

        # GroupNorm
        triplane_tokens = backbone.norm_triplane(hidden_states)
        stats("after GroupNorm", triplane_tokens)

        triplane_tokens = triplane_tokens.permute(0, 2, 1)
        stats("after permute [B,N,C]", triplane_tokens)

        triplane_tokens = backbone.proj_triplane(triplane_tokens)
        stats("after proj_triplane", triplane_tokens)

        # Image tokens
        image_tokens = backbone.norm_image(encoder_hidden_states)
        stats("norm_image", image_tokens)
        image_tokens = backbone.proj_image(image_tokens)
        stats("proj_image", image_tokens)

        # Latent init
        init_latents = backbone.latent_init.expand(1, -1, -1)
        stats("latent_init (raw)", init_latents)
        init_latents = backbone.norm_latent(init_latents)
        stats("latent_init (normed)", init_latents)
        init_latents = backbone.proj_latent(init_latents)
        stats("latent_init (projected)", init_latents)

        # Concat
        latent_tokens = torch.cat([image_tokens, init_latents], dim=1)
        stats("latent_tokens (concat)", latent_tokens)

        # Main blocks
        for i, block in enumerate(backbone.main_blocks):
            latent_tokens, triplane_tokens = block(
                latent_tokens, triplane_tokens, encoder_hidden_states)
            stats(f"block {i} latent", latent_tokens)
            stats(f"block {i} triplane", triplane_tokens)

        # Project out
        output = backbone.proj_out(triplane_tokens).permute(0, 2, 1).contiguous()
        stats("proj_out (permuted)", output)

        # Residual
        output = output + hidden_states
        stats("final (with residual)", output)

        # Detokenize for comparison
        scene_codes = model.post_processor(model.tokenizer.detokenize(output))
        stats("\nscene_codes (post-proc)", scene_codes)


if __name__ == "__main__":
    main()
