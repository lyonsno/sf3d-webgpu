"""Dump per-stage PyTorch reference tensors for numerical parity verification.

Run via greenroom:
  cd ~/dev/gpu-greenroom && .venv/bin/gpu-greenroom submit sf3d_parity_reference \
    ~/.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png /tmp/sf3d-parity-ref/

Outputs .npy files for each pipeline stage, plus a summary JSON.
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
    img_path = (sys.argv[1] if len(sys.argv) > 1 else
        os.path.expanduser("~/.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png"))
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "tools/parity_reference"
    os.makedirs(out_dir, exist_ok=True)

    from sf3d.system import SF3D
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.eval()

    # --- Prepare image exactly as SF3D does ---
    img = Image.open(img_path).convert("RGBA")
    print(f"Image: {img.size}")

    cond_size = model.cfg.cond_image_size  # 512
    img_resized = img.resize((cond_size, cond_size))
    img_arr = np.asarray(img_resized).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_arr).float().clip(0, 1)
    mask_cond = img_t[:, :, -1:]
    bg = torch.tensor(model.cfg.background_color)[None, None, :]
    rgb_cond = torch.lerp(bg, img_t[:, :, :3], mask_cond)

    # Build the batch exactly as system.py does
    batch = {
        "rgb_cond": rgb_cond.unsqueeze(0),      # [1, H, W, 3]
        "mask_cond": mask_cond.unsqueeze(0),     # [1, H, W, 1]
    }

    # Camera (default distance, matching system.py run_image)
    from sf3d.utils import create_intrinsic_from_fov_deg, default_cond_c2w
    default_distance = model.cfg.default_distance  # 1.6
    c2w_cond = default_cond_c2w(default_distance)
    intrinsic, intrinsic_normed = create_intrinsic_from_fov_deg(
        model.cfg.default_fovy_deg, cond_size, cond_size
    )
    batch["c2w_cond"] = c2w_cond.view(1, 1, 4, 4)
    batch["intrinsic_cond"] = intrinsic.unsqueeze(0).unsqueeze(0)
    batch["intrinsic_normed_cond"] = intrinsic_normed.unsqueeze(0).unsqueeze(0)

    # Add view dimension for rgb/mask
    batch["rgb_cond"] = batch["rgb_cond"].unsqueeze(1)
    batch["mask_cond"] = batch["mask_cond"].unsqueeze(1)

    summary = {}

    with torch.no_grad():
        # Use run_image for the end-to-end mesh (Stage 7), but also decompose
        # the pipeline manually for per-stage intermediates.

        # --- Manual decomposition for intermediate tensors ---
        from einops import rearrange
        from sf3d.utils import create_intrinsic_from_fov_deg, default_cond_c2w

        # Replicate run_image's batch construction
        rgb_cond_t = rgb_cond.unsqueeze(0)  # [1, H, W, 3]
        mask_cond_t = mask_cond.unsqueeze(0)

        c2w_cond = default_cond_c2w(model.cfg.default_distance)
        intrinsic, intrinsic_normed = create_intrinsic_from_fov_deg(
            model.cfg.default_fovy_deg, cond_size, cond_size)

        manual_batch = {
            "rgb_cond": rgb_cond_t.unsqueeze(1),           # [1, 1, H, W, 3]
            "mask_cond": mask_cond_t.unsqueeze(1),          # [1, 1, H, W, 1]
            "c2w_cond": c2w_cond.view(1, 1, 4, 4),         # [1, 1, 4, 4]
            "intrinsic_cond": intrinsic.unsqueeze(0).unsqueeze(0),
            "intrinsic_normed_cond": intrinsic_normed.unsqueeze(0).unsqueeze(0),
        }

        # Stage 2: Camera embedding
        camera_embeds = model.camera_embedder(**manual_batch)
        cam_np = camera_embeds.cpu().numpy()
        np.save(os.path.join(out_dir, "camera_embed.npy"), cam_np)
        summary["camera_embed"] = {
            "shape": list(cam_np.shape),
            "min": float(cam_np.min()), "max": float(cam_np.max()),
            "first_8": cam_np.flatten()[:8].tolist(),
        }
        print(f"Camera embed: {cam_np.shape}, range=[{cam_np.min():.4f}, {cam_np.max():.4f}]")

        # Stage 3: DINOv2 image tokenization
        input_image_tokens = model.image_tokenizer(
            rearrange(manual_batch["rgb_cond"], "B Nv H W C -> B Nv C H W"),
            modulation_cond=camera_embeds,
        )
        dinov2_np = input_image_tokens.cpu().numpy()
        np.save(os.path.join(out_dir, "dinov2_output.npy"), dinov2_np)
        summary["dinov2_output"] = {
            "shape": list(dinov2_np.shape),
            "min": float(dinov2_np.min()), "max": float(dinov2_np.max()),
            "first_8": dinov2_np.flatten()[:8].tolist(),
        }
        print(f"DINOv2 output: {dinov2_np.shape}, range=[{dinov2_np.min():.4f}, {dinov2_np.max():.4f}]")

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1)

        # Stage 4: Two-stream backbone
        tokens = model.tokenizer(1)
        backbone_output = model.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
            modulation_cond=None,
        )
        backbone_np = backbone_output.cpu().numpy()
        np.save(os.path.join(out_dir, "backbone_output.npy"), backbone_np)
        summary["backbone_output"] = {
            "shape": list(backbone_np.shape),
            "min": float(backbone_np.min()), "max": float(backbone_np.max()),
            "first_8": backbone_np.flatten()[:8].tolist(),
        }
        print(f"Backbone output: {backbone_np.shape}, range=[{backbone_np.min():.4f}, {backbone_np.max():.4f}]")

        # Stage 5: Detokenize + Post-processor
        direct_codes = model.tokenizer.detokenize(backbone_output)
        scene_codes = model.post_processor(direct_codes)
        scene_np = scene_codes.cpu().numpy()
        np.save(os.path.join(out_dir, "scene_codes.npy"), scene_np)
        summary["scene_codes"] = {
            "shape": list(scene_np.shape),
            "min": float(scene_np.min()), "max": float(scene_np.max()),
            "first_8": scene_np.flatten()[:8].tolist(),
        }
        print(f"Scene codes (triplane): {scene_np.shape}, range=[{scene_np.min():.4f}, {scene_np.max():.4f}]")

        # Stage 6: Density at grid positions via query_triplane + decoder
        bbox = model.bbox.cpu()
        grid_verts = model.isosurface_helper._grid_vertices.cpu().float()
        grid_positions = grid_verts * (bbox[1] - bbox[0]) + bbox[0]
        np.save(os.path.join(out_dir, "grid_positions.npy"), grid_positions.numpy())

        # Query triplane features at grid positions (need batch dim)
        grid_pos_gpu = grid_positions.unsqueeze(0).to(scene_codes.device)
        triplane_features = model.query_triplane(grid_pos_gpu, scene_codes)
        tri_feat_np = triplane_features.cpu().numpy()
        np.save(os.path.join(out_dir, "triplane_features.npy"), tri_feat_np)
        summary["triplane_features"] = {
            "shape": list(tri_feat_np.shape),
            "min": float(tri_feat_np.min()), "max": float(tri_feat_np.max()),
            "first_8": tri_feat_np.flatten()[:8].tolist(),
        }
        print(f"Triplane features: {tri_feat_np.shape}, range=[{tri_feat_np.min():.4f}, {tri_feat_np.max():.4f}]")

        # Decode density + vertex_offset
        decoded = model.decoder(triplane_features, include=["density", "vertex_offset"])

        density_np = decoded["density"].cpu().numpy()
        np.save(os.path.join(out_dir, "density.npy"), density_np)
        summary["density"] = {
            "shape": list(density_np.shape),
            "min": float(density_np.min()), "max": float(density_np.max()),
            "mean": float(density_np.mean()),
            "num_positive": int((density_np > 0).sum()),
            "first_8": density_np.flatten()[:8].tolist(),
        }
        print(f"Density: {density_np.shape}, range=[{density_np.min():.4f}, {density_np.max():.4f}], positive={int((density_np > 0).sum())}")

        if "vertex_offset" in decoded:
            offset_np = decoded["vertex_offset"].cpu().numpy()
            np.save(os.path.join(out_dir, "vertex_offset.npy"), offset_np)
            summary["vertex_offset"] = {
                "shape": list(offset_np.shape),
                "min": float(offset_np.min()), "max": float(offset_np.max()),
                "first_8": offset_np.flatten()[:8].tolist(),
            }
            print(f"Vertex offset: {offset_np.shape}, range=[{offset_np.min():.4f}, {offset_np.max():.4f}]")

        # --- Stage 7: Full mesh via run_image ---
        img_pil = Image.open(img_path).convert("RGBA")
        mesh, global_dict = model.run_image(img_pil, bake_resolution=1024)
        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        np.save(os.path.join(out_dir, "vertices.npy"), verts)
        np.save(os.path.join(out_dir, "faces.npy"), faces)
        summary["mesh"] = {
            "num_vertices": len(verts),
            "num_faces": len(faces),
            "vert_bounds_min": verts.min(axis=0).tolist(),
            "vert_bounds_max": verts.max(axis=0).tolist(),
            "first_5_verts": verts[:5].tolist(),
        }
        print(f"Mesh: {len(verts)} vertices, {len(faces)} faces")
        print(f"  Bounds: {verts.min(axis=0)} to {verts.max(axis=0)}")

        # Materials
        roughness = global_dict.get("roughness", global_dict.get("decoder_roughness"))
        metallic = global_dict.get("metallic", global_dict.get("decoder_metallic"))
        if roughness is not None:
            r_val = roughness.item() if hasattr(roughness, 'item') else float(roughness)
            m_val = metallic.item() if hasattr(metallic, 'item') else float(metallic)
            summary["materials"] = {"roughness": r_val, "metallic": m_val}
            print(f"Materials: roughness={r_val:.6f}, metallic={m_val:.6f}")

    # Write summary
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    print(f"Reference tensors written to {out_dir}/")


if __name__ == "__main__":
    main()
