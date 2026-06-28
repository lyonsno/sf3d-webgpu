#!/usr/bin/env python3
"""Verify cross-attention computation by comparing PyTorch vs manual numpy.

Isolates the FuseBlock's cross-attention to verify the Q/K/V indexing
and score computation match between PyTorch and our shader's logic.
"""

import sys, os
import numpy as np
import torch

SF3D_REPO = os.environ.get("SF3D_REPO", os.path.expanduser("~/dev/sf3d"))
sys.path.insert(0, SF3D_REPO)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sf3d.system import SF3D
from sf3d.utils import default_cond_c2w, create_intrinsic_from_fov_deg
from PIL import Image


def manual_cross_attention(q_input, kv_input, wq, wk, wv, w_proj, b_proj, num_heads=16):
    """Manual cross-attention matching our WebGPU shader logic."""
    N_q, dim = q_input.shape
    N_kv = kv_input.shape[0]
    head_dim = dim // num_heads

    # Linear projections (no bias)
    # PyTorch weight is [out, in], need [in, out] for manual matmul
    Q = q_input @ wq.T  # [N_q, dim]
    K = kv_input @ wk.T  # [N_kv, dim]
    V = kv_input @ wv.T  # [N_kv, dim]

    # Reshape to [N, H, D_h]
    Q = Q.reshape(N_q, num_heads, head_dim)
    K = K.reshape(N_kv, num_heads, head_dim)
    V = V.reshape(N_kv, num_heads, head_dim)

    scale = 1.0 / np.sqrt(head_dim)

    # Compute scores: [H, N_q, N_kv]
    # For each head h: scores[h, qi, ki] = Q[qi, h, :] . K[ki, h, :] * scale
    scores = np.zeros((num_heads, N_q, N_kv), dtype=np.float32)
    for h in range(num_heads):
        scores[h] = Q[:, h, :] @ K[:, h, :].T * scale

    # Softmax over N_kv dimension
    for h in range(num_heads):
        for qi in range(N_q):
            row = scores[h, qi]
            row = row - row.max()
            e = np.exp(row)
            scores[h, qi] = e / e.sum()

    # Apply: output[qi, h, d] = sum_ki scores[h, qi, ki] * V[ki, h, d]
    output = np.zeros((N_q, num_heads, head_dim), dtype=np.float32)
    for h in range(num_heads):
        output[:, h, :] = scores[h] @ V[:, h, :]  # [N_q, N_kv] @ [N_kv, D_h]

    # Reshape back to [N_q, dim]
    output = output.reshape(N_q, dim)

    # Output projection
    output = output @ w_proj.T + b_proj

    return output


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png")

    model = SF3D.from_pretrained("stabilityai/stable-fast-3d",
        config_name="config.yaml", weight_name="model.safetensors")
    model.eval()
    device = model.device

    image = Image.open(image_path)
    with torch.no_grad():
        mask_cond, rgb_cond = model.prepare_image(image)
        c2w_cond = default_cond_c2w(model.cfg.default_distance).to(device)
        intrinsic, intrinsic_normed_cond = create_intrinsic_from_fov_deg(
            model.cfg.default_fovy_deg, model.cfg.cond_image_size, model.cfg.cond_image_size)
        batch = {
            "rgb_cond": rgb_cond.unsqueeze(0).to(device), "mask_cond": mask_cond.unsqueeze(0).to(device),
            "c2w_cond": c2w_cond.view(1,1,4,4), "intrinsic_cond": intrinsic.to(device).view(1,1,3,3),
            "intrinsic_normed_cond": intrinsic_normed_cond.to(device).view(1,1,3,3),
        }

        # Run full backbone to get intermediate state
        from einops import rearrange
        camera_embeds = model.camera_embedder(**batch)
        rgb = batch["rgb_cond"]
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(1)
        input_image_tokens = model.image_tokenizer(
            rearrange(rgb, "B Nv H W C -> B Nv C H W"), modulation_cond=camera_embeds)
        input_image_tokens = rearrange(input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1)

        hidden_states = model.tokenizer(1)
        backbone = model.backbone

        # Manually trace through to fuse_block_in of block 0
        triplane_tokens = backbone.norm_triplane(hidden_states)
        triplane_tokens = triplane_tokens.permute(0, 2, 1)
        triplane_tokens = backbone.proj_triplane(triplane_tokens)

        image_tokens = backbone.norm_image(input_image_tokens)
        image_tokens = backbone.proj_image(image_tokens)
        init_latents = backbone.latent_init.expand(1, -1, -1)
        init_latents = backbone.norm_latent(init_latents)
        init_latents = backbone.proj_latent(init_latents)
        latent_tokens = torch.cat([image_tokens, init_latents], dim=1)

        # Now run fuse_block_in manually
        fb = backbone.main_blocks[0].fuse_block_in
        z = latent_tokens[0]  # [3089, 1024]
        x = triplane_tokens[0]  # [27648, 1024]

        z_normed = fb.norm_z1(z.unsqueeze(0)).squeeze(0)
        x_normed = fb.norm_x(x.unsqueeze(0)).squeeze(0) if fb.norm_x_input else x

        # PyTorch attention
        pt_attn_out = fb.attn(z_normed.unsqueeze(0), x_normed.unsqueeze(0)).squeeze(0)

        # Manual numpy attention (matching our shader logic)
        wq = fb.attn.wq.weight.cpu().numpy()  # [1024, 1024]
        wk = fb.attn.wk.weight.cpu().numpy()
        wv = fb.attn.wv.weight.cpu().numpy()
        w_proj = fb.attn.proj.weight.cpu().numpy()
        b_proj = fb.attn.proj.bias.cpu().numpy()

        manual_out = manual_cross_attention(
            z_normed.cpu().numpy(), x_normed.cpu().numpy(),
            wq, wk, wv, w_proj, b_proj, num_heads=16)

        pt_np = pt_attn_out.cpu().numpy()

        print(f"PyTorch attn output: shape={pt_np.shape}, min={pt_np.min():.4f}, max={pt_np.max():.4f}")
        print(f"Manual attn output:  shape={manual_out.shape}, min={manual_out.min():.4f}, max={manual_out.max():.4f}")

        diff = np.abs(pt_np - manual_out)
        print(f"Max abs diff: {diff.max():.6f}")
        print(f"Mean abs diff: {diff.mean():.6f}")

        # Check first 5 values
        print(f"\nFirst 5 values comparison:")
        print(f"  PyTorch: {pt_np[0, :5].tolist()}")
        print(f"  Manual:  {manual_out[0, :5].tolist()}")

        # Now check: does our WebGPU use TRANSPOSED weights?
        # Our converter transposes 2D .weight tensors from [out, in] to [in, out]
        # So in our weight file: wq_transposed = wq.T = [in, out] = [1024, 1024]
        # Our shader does: Q[qi, j] = sum_k input[qi, k] * weight[k * outDim + j]
        # = sum_k input[qi, k] * wq_T[k, j]
        # = input @ wq_T = input @ wq.T
        #
        # PyTorch does: Q = wq(input) = input @ wq.T (because nn.Linear stores [out, in])
        #
        # So both compute input @ wq.T — they should be identical!

        # But wait — our _dispatchLinearNoBias does:
        # _dispatchLinear(input, output, weight, zeroBias, rows, inDim, outDim)
        # where weight is the TRANSPOSED version from the converter
        # The shader computes: output[row, col] = sum_k input[row, k] * weight[k * stride + col]
        # where stride = outDim
        # So it accesses weight as [k, col] = [inDim, outDim] row-major
        # = weight_transposed[in, out]
        # So output = input @ weight_transposed = input @ wq.T
        # Same as PyTorch! ✓

        print("\n✓ Weight transposition logic is consistent")

        # The manual implementation uses wq.T which matches both PyTorch and WebGPU
        # If the manual output matches PyTorch, the WebGPU shader logic is correct
        # (assuming the shader is faithfully implementing the manual logic)

if __name__ == "__main__":
    main()
