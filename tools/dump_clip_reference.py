"""Dump CLIP reference values for sf3d-webgpu comparison.
Run via greenroom: gpu-greenroom submit sf3d_weight_convert ... 
Or directly: python tools/dump_clip_reference.py
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SF3D_REPO = os.environ.get("SF3D_REPO", os.path.expanduser("~/dev/sf3d"))
sys.path.insert(0, SF3D_REPO)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def main():
    from sf3d.system import SF3D

    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.eval()

    # Load image — accept path from CLI arg or env var
    img_path = (sys.argv[1] if len(sys.argv) > 1 else
        os.environ.get("IMAGE_PATH",
        os.path.expanduser("~/.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png")))
    img = Image.open(img_path).convert("RGBA")
    print(f"Image: {img.size}")

    # Replicate SF3D's prepare_image
    cond_size = model.cfg.cond_image_size  # 512
    img_resized = img.resize((cond_size, cond_size))
    img_arr = np.asarray(img_resized).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_arr).float().clip(0, 1)
    mask_cond = img_t[:, :, -1:]
    bg = torch.tensor([0.5, 0.5, 0.5])[None, None, :]
    rgb_cond = torch.lerp(bg, img_t[:, :, :3], mask_cond)

    # This is what the estimator receives: rgb_cond * mask_cond
    estimator_input = (rgb_cond * mask_cond).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 3]
    print(f"Estimator input shape: {estimator_input.shape}")
    print(f"Input range: [{estimator_input.min():.6f}, {estimator_input.max():.6f}]")

    # Run the estimator's preprocessing
    with torch.no_grad():
        est = model.image_estimator
        ci = estimator_input.flatten(0, 1).permute(0, 3, 1, 2).contiguous()
        ci_224 = F.interpolate(ci, size=(224, 224), mode='bilinear', align_corners=False)
        
        from torchvision.transforms import Normalize
        import open_clip
        ci_norm = Normalize(
            mean=open_clip.constants.OPENAI_DATASET_MEAN,
            std=open_clip.constants.OPENAI_DATASET_STD,
        )(ci_224)
        print(f"\nNormalized input shape: {ci_norm.shape}")
        print(f"Normalized[:,:,0,0]: {ci_norm[0, :, 0, 0].tolist()}")
        print(f"Normalized[:,:,112,112]: {ci_norm[0, :, 112, 112].tolist()}")

        # Run visual encoder
        features = est.model.encode_image(ci_norm)
        print(f"\nFeatures shape: {features.shape}")
        print(f"Features[:8]: {features[0, :8].tolist()}")
        print(f"Features norm: {features.norm().item():.6f}")

        # Run heads — keys might be nested under head config names
        result = est(estimator_input)
        print(f"\nEstimator result keys: {list(result.keys())}")
        for k, v in result.items():
            if hasattr(v, 'item'):
                print(f"  {k}: {v.item():.6f}")
            elif hasattr(v, 'mean'):
                print(f"  {k}: mean={v.mean.item():.6f}, mode={getattr(v, 'mode', 'N/A')}")

        # Also run heads manually for comparison
        shared_rough = est.heads['roughness'][0](features)
        d1_rough = est.heads['roughness'][1](shared_rough).squeeze(-1)
        d2_rough = est.heads['roughness'][2](shared_rough).squeeze(-1)
        alpha_r = F.softplus(d1_rough + 1.0)
        beta_r = F.softplus(d2_rough + 1.0)
        mode_r = (alpha_r - 1) / (alpha_r + beta_r - 2)
        print(f"\nManual roughness: d1={d1_rough.item():.6f}, d2={d2_rough.item():.6f}")
        print(f"  alpha={alpha_r.item():.6f}, beta={beta_r.item():.6f}, mode={mode_r.item():.6f}")

        shared_metal = est.heads['metallic'][0](features)
        d1_metal = est.heads['metallic'][1](shared_metal).squeeze(-1)
        d2_metal = est.heads['metallic'][2](shared_metal).squeeze(-1)
        alpha_m = F.softplus(d1_metal + 1.0)
        beta_m = F.softplus(d2_metal + 1.0)
        mode_m = (alpha_m - 1) / (alpha_m + beta_m - 2)
        print(f"Manual metallic: d1={d1_metal.item():.6f}, d2={d2_metal.item():.6f}")
        print(f"  alpha={alpha_m.item():.6f}, beta={beta_m.item():.6f}, mode={mode_m.item():.6f}")

        # Dump detailed intermediates
        ref = {
            "input_pixel_0_0": ci_norm[0, :, 0, 0].tolist(),
            "input_pixel_112_112": ci_norm[0, :, 112, 112].tolist(),
            "features_first_8": features[0, :8].tolist(),
            "features_norm": features.norm().item(),
            "roughness_d1": d1_rough.item(),
            "roughness_d2": d2_rough.item(),
            "metallic_d1": d1_metal.item(),
            "metallic_d2": d2_metal.item(),
        }
        
        # Also dump block-by-block outputs
        visual = est.model.visual
        x = visual.conv1(ci_norm)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(1, -1, -1), x], dim=1)
        x = x + visual.positional_embedding
        x = visual.ln_pre(x)
        ref["after_ln_pre_cls_first_8"] = x[0, 0, :8].tolist()
        ref["after_ln_pre_patch0_first_8"] = x[0, 1, :8].tolist()
        
        x_lnd = x.permute(1, 0, 2)
        for i, block in enumerate(visual.transformer.resblocks):
            if i == 0:
                # Detailed block 0 decomposition
                ln1_out = block.ln_1(x_lnd)
                qkv_out = F.linear(ln1_out, block.attn.in_proj_weight, block.attn.in_proj_bias)
                ref["block0_qkv_cls_first_8"] = qkv_out[0, 0, :8].tolist()

                # Run attention via nn.MultiheadAttention
                attn_out = block.attention(ln1_out)
                ref["block0_attnOut_cls_first_8"] = attn_out[0, 0, :8].tolist()

                # Out proj is inside nn.MultiheadAttention, get post-residual
                x_post_attn = x_lnd + block.ls_1(attn_out)
                ref["block0_afterAttnResidual_cls_first_8"] = x_post_attn[0, 0, :8].tolist()

                # MLP
                ln2_out = block.ln_2(x_post_attn)
                fc_out = block.mlp.c_fc(ln2_out)
                ref["block0_fc_cls_first_8"] = fc_out[0, 0, :8].tolist()

                mlp_out = block.mlp(ln2_out)
                ref["block0_mlpOut_cls_first_8"] = mlp_out[0, 0, :8].tolist()
            x_lnd = block(x_lnd)
            if i == 0:
                ref["after_block0_cls_first_8"] = x_lnd[0, 0, :8].tolist()
            if i == 11:
                ref["after_block11_cls_first_8"] = x_lnd[0, 0, :8].tolist()
        
        x = x_lnd.permute(1, 0, 2)
        x = visual.ln_post(x)
        ref["after_ln_post_cls_first_8"] = x[0, 0, :8].tolist()
        
        pooled = x[:, 0]
        projected = pooled @ visual.proj
        ref["projected_first_8"] = projected[0, :8].tolist()
        
        out_path = os.path.join(os.path.dirname(__file__), "clip_reference.json")
        with open(out_path, "w") as f:
            json.dump(ref, f, indent=2)
        print(f"\nReference dumped to {out_path}")

if __name__ == "__main__":
    main()
