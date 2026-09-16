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

        # --- Stage 7: Raw mesh via triplane_to_meshes (no post-processing) ---
        from contextlib import nullcontext
        from sf3d.utils import get_device
        dev = get_device()
        autocast_ctx = (
            torch.autocast(device_type=dev, enabled=False)
            if dev in ("cuda", "mps") or "cuda" in dev else nullcontext()
        )
        with autocast_ctx:
            raw_meshes = model.triplane_to_meshes(scene_codes)
        raw_mesh = raw_meshes[0]
        raw_verts = raw_mesh.v_pos.cpu().float().numpy()
        raw_faces = raw_mesh.t_pos_idx.cpu().numpy()
        np.save(os.path.join(out_dir, "raw_vertices.npy"), raw_verts)
        np.save(os.path.join(out_dir, "raw_faces.npy"), raw_faces)
        summary["raw_mesh"] = {
            "num_vertices": len(raw_verts),
            "num_faces": len(raw_faces),
            "vert_bounds_min": raw_verts.min(axis=0).tolist(),
            "vert_bounds_max": raw_verts.max(axis=0).tolist(),
            "first_5_verts": raw_verts[:5].tolist(),
        }
        print(f"Raw mesh (pre-post-processing): {len(raw_verts)} vertices, {len(raw_faces)} faces")
        print(f"  Bounds: {raw_verts.min(axis=0)} to {raw_verts.max(axis=0)}")

        # Also get run_image mesh for comparison
        img_pil = Image.open(img_path).convert("RGBA")
        mesh_final, global_dict = model.run_image(img_pil, bake_resolution=1024)
        final_verts = np.array(mesh_final.vertices)
        final_faces = np.array(mesh_final.faces)
        summary["final_mesh"] = {
            "num_vertices": len(final_verts),
            "num_faces": len(final_faces),
        }
        print(f"Final mesh (run_image): {len(final_verts)} vertices, {len(final_faces)} faces")

        # SDF stats for isosurface comparison
        sdf_from_density = density_np.flatten() - 10.0  # isosurface_threshold = 10.0
        near_surface = np.abs(sdf_from_density) < 0.5
        summary["sdf_near_surface_count"] = int(near_surface.sum())
        summary["isosurface_threshold"] = 10.0
        print(f"SDF near-surface (<0.5 from threshold): {near_surface.sum()} grid points")

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

    # Provenance manifest: binds every artifact to the input, the PyTorch/SF3D
    # source, the model snapshot and weights, and this generator, so a stale,
    # synthetic or unrelated directory can never pass as parity evidence
    # (review 2026-09-16, MEDIUM). smoke_parity.mjs refuses a reference
    # directory without a matching manifest.
    manifest_path = write_reference_manifest(out_dir, img_path, model)
    print(f"Manifest written to {manifest_path}")


MANIFEST_SCHEMA = "sf3d.parity-reference-manifest.v0"


def _sha256_file(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_identity(repo):
    import subprocess
    try:
        commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True).strip() != ""
        return {"repo": os.path.abspath(repo), "commit": commit, "dirty": dirty}
    except Exception as exc:  # noqa: BLE001 - provenance must be honest, not fatal
        return {"repo": os.path.abspath(repo), "commit": None, "dirty": None, "error": str(exc)}


def _model_identity(model):
    repo_id = "stabilityai/stable-fast-3d"
    out = {"repo_id": repo_id, "config_name": "config.yaml", "weight_name": "model.safetensors"}
    try:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(repo_id, "model.safetensors", local_files_only=True)
        config = hf_hub_download(repo_id, "config.yaml", local_files_only=True)
        # The hub cache hands back .../snapshots/<commit>/<file> as a symlink into
        # blobs/; parse the snapshot commit from the returned path, not its realpath.
        snapshot = None
        parts = os.path.abspath(weights).split(os.sep)
        if "snapshots" in parts:
            snapshot = parts[parts.index("snapshots") + 1]
        out.update({
            "snapshot_commit": snapshot,
            "weights_path": weights,
            "weights_blob_path": os.path.realpath(weights),
            "weights_sha256": _sha256_file(weights),
            "weights_bytes": os.path.getsize(weights),
            "config_sha256": _sha256_file(config),
        })
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    out["config"] = {k: getattr(model.cfg, k, None) for k in ("cond_image_size", "default_distance", "default_fovy_deg", "isosurface_resolution")}
    return out


def write_reference_manifest(out_dir, img_path, model):
    import datetime
    artifacts = {}
    for name in sorted(os.listdir(out_dir)):
        if name == "manifest.json":
            continue
        full = os.path.join(out_dir, name)
        if os.path.isfile(full) and (name.endswith(".npy") or name == "summary.json"):
            artifacts[name] = {"sha256": _sha256_file(full), "bytes": os.path.getsize(full)}
    script = os.path.abspath(__file__)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input": {"path": os.path.abspath(img_path), "sha256": _sha256_file(img_path), "bytes": os.path.getsize(img_path)},
        "generator": {
            "script": os.path.relpath(script, os.getcwd()),
            "script_sha256": _sha256_file(script),
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "sf3d_webgpu": _git_identity(os.getcwd()),
        },
        "sf3d": _git_identity(SF3D_REPO),
        "model": _model_identity(model),
        "torch": {"version": torch.__version__, "mps_available": bool(torch.backends.mps.is_available()), "device": "mps" if torch.backends.mps.is_available() else "cpu"},
        "artifacts": artifacts,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


if __name__ == "__main__":
    main()
