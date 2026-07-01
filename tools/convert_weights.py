#!/usr/bin/env python3
"""Convert SF3D PyTorch weights to flat binary format for WebGPU.

Produces a binary file with the same format as moge-webgpu/sharp-webgpu:
  Header (16 bytes): magic + version + num_tensors + header_size
  Tensor table (160 bytes each): name + dtype + ndim + shape + offset + size
  Weight data: packed tensors with 16-byte alignment

Usage:
  # From the sf3d repo directory:
  python tools/convert_weights.py --output public/weights.bin --dtype fp16

  # Or with a local checkpoint:
  python tools/convert_weights.py --model-path /path/to/sf3d --output public/weights.bin
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

# Add sf3d repo to path
SF3D_REPO = os.environ.get("SF3D_REPO", os.path.expanduser("~/dev/sf3d"))
sys.path.insert(0, SF3D_REPO)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MAGIC = 0x33445346  # "SF3D" in little-endian
VERSION = 1
ENTRY_SIZE = 160
NAME_SIZE = 128
ALIGN = 16

# Skip these prefixes — not needed for WebGPU inference
SKIP_PREFIXES = [
    # Global estimator (illumination) — skip for v1, hardcode defaults
    "global_estimator.",
    # Buffers we handle differently
    "isosurface_helper.",
    "image_tokenizer.model.embeddings.mask_token",
    "bbox",
]

# These are integer buffers needed for marching tetrahedra — export separately
MARCHING_TET_BUFFERS = [
    "isosurface_helper.triangle_table",
    "isosurface_helper.num_triangles_table",
    "isosurface_helper.base_tet_edges",
    "isosurface_helper._grid_vertices",
    "isosurface_helper.indices",
    "isosurface_helper.center_indices",
    "isosurface_helper.boundary_indices",
]


def should_skip(name):
    for prefix in SKIP_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def should_transpose(name, shape):
    """Determine if a weight tensor needs transposition for WebGPU.

    PyTorch stores linear weights as [out_features, in_features].
    WebGPU shaders expect [in_features, out_features] for row-major access.
    """
    if len(shape) != 2:
        return False

    # All 2D weight tensors (not biases, not embeddings with special shapes)
    if name.endswith(".weight"):
        # Skip position embeddings, cls_token, etc.
        if "embeddings" in name and "projection" not in name:
            return False
        if "token_embedding" in name:
            return False
        return True

    # Explicit projection matrices
    if name.endswith(".proj"):
        return True

    return False


def convert_weights(model_path, output_path, dtype="fp16", manifest_path=None):
    from sf3d.system import SF3D

    print(f"Loading SF3D model from {model_path}...")
    model = SF3D.from_pretrained(
        model_path,
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.eval()

    # Collect all parameters and buffers
    state = {}
    for name, param in model.named_parameters():
        state[name] = param.detach().cpu()
    for name, buf in model.named_buffers():
        state[name] = buf.detach().cpu()

    # Filter and process tensors
    tensors = []
    for name, tensor in sorted(state.items()):
        if should_skip(name):
            continue

        arr = tensor.float().numpy()

        # Transpose 2D linear weights
        if should_transpose(name, arr.shape):
            arr = arr.T.copy()

        # Interpolate DINOv2 position embeddings from pretrained grid to 36x36
        # Must match PyTorch's interpolate_pos_encoding() exactly
        if name == "image_tokenizer.model.embeddings.position_embeddings":
            target_h, target_w = 36, 36
            D = arr.shape[-1]
            cls_pos = arr[0, 0:1, :]  # [1, D]
            patch_pos = arr[0, 1:, :]  # [num_pretrained, D]
            num_pretrained = patch_pos.shape[0]
            pretrained_size = int(np.sqrt(num_pretrained))
            print(f"  Interpolating pos embed: [{pretrained_size}x{pretrained_size}] -> [{target_h}x{target_w}]")

            # Use PyTorch's F.interpolate to match exactly
            import torch.nn.functional as F
            patch_pos_t = torch.from_numpy(patch_pos).float()
            # Reshape to [1, D, H, W] for F.interpolate
            patch_pos_2d = patch_pos_t.T.reshape(1, D, pretrained_size, pretrained_size)
            # Match DINOv2's interpolate_pos_encoding: bicubic, align_corners=False
            interpolated = F.interpolate(
                patch_pos_2d,
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
            )
            # Reshape back to [target_h*target_w, D]
            interpolated = interpolated.reshape(D, -1).T.numpy()  # [1296, D]
            arr = np.concatenate([cls_pos, interpolated], axis=0)[np.newaxis]  # [1, 1297, D]
            print(f"  New pos embed shape: {arr.shape}")

        # Convert to target dtype
        # CLIP image estimator weights need fp32 — fp16 quantization drifts
        # through 12 transformer blocks and corrupts roughness/metallic prediction
        needs_fp32 = name.startswith("image_estimator.")
        if dtype == "fp16" and not needs_fp32:
            arr = arr.astype(np.float16)
            dtype_code = 1
        else:
            arr = arr.astype(np.float32)
            dtype_code = 0

        tensors.append({
            "name": name,
            "dtype": dtype_code,
            "shape": list(arr.shape),
            "data": arr.tobytes(),
        })

    # Also export marching tetrahedra data as separate files
    tet_data = {}
    for name in MARCHING_TET_BUFFERS:
        if name in state:
            t = state[name]
            if t.dtype in (torch.int64, torch.long):
                tet_data[name] = t.numpy().astype(np.int32)
            else:
                tet_data[name] = t.float().numpy()

    # Write binary file
    num_tensors = len(tensors)
    header_size = 16 + num_tensors * ENTRY_SIZE

    print(f"Writing {num_tensors} tensors to {output_path}...")

    # Calculate offsets with alignment
    data_offset = 0
    for t in tensors:
        t["offset"] = data_offset
        t["size"] = len(t["data"])
        data_offset += t["size"]
        # Align to 16 bytes
        pad = (ALIGN - (data_offset % ALIGN)) % ALIGN
        data_offset += pad

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", num_tensors))
        f.write(struct.pack("<I", header_size))

        # Tensor table
        for t in tensors:
            # Name (128 bytes, null-padded)
            name_bytes = t["name"].encode("ascii")[:NAME_SIZE]
            f.write(name_bytes.ljust(NAME_SIZE, b"\0"))

            # dtype, ndim
            f.write(struct.pack("<I", t["dtype"]))
            ndim = len(t["shape"])
            f.write(struct.pack("<I", ndim))

            # shape (4 x u32, padded)
            for i in range(4):
                if i < ndim:
                    f.write(struct.pack("<I", t["shape"][i]))
                else:
                    f.write(struct.pack("<I", 0))

            # offset, size
            f.write(struct.pack("<I", t["offset"]))
            f.write(struct.pack("<I", t["size"]))

        # Weight data
        for t in tensors:
            f.write(t["data"])
            # Alignment padding
            pad = (ALIGN - (len(t["data"]) % ALIGN)) % ALIGN
            if pad:
                f.write(b"\0" * pad)

    file_size = os.path.getsize(output_path)
    print(f"Written {file_size / (1024*1024):.1f} MB ({num_tensors} tensors)")

    # Write manifest JSON
    if manifest_path is None:
        manifest_path = output_path.replace(".bin", ".json")

    manifest = {
        "magic": hex(MAGIC),
        "version": VERSION,
        "dtype": dtype,
        "num_tensors": num_tensors,
        "file_size_bytes": file_size,
        "tensors": [
            {
                "name": t["name"],
                "dtype": "fp16" if t["dtype"] == 1 else "fp32",
                "shape": t["shape"],
                "offset": t["offset"],
                "size": t["size"],
                "transposed": should_transpose(t["name"], state[t["name"]].shape),
            }
            for t in tensors
        ],
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")

    # Write marching tetrahedra data
    tet_dir = os.path.join(os.path.dirname(output_path), "tets")
    os.makedirs(tet_dir, exist_ok=True)
    for name, arr in tet_data.items():
        short_name = name.split(".")[-1]
        tet_path = os.path.join(tet_dir, f"{short_name}.bin")
        arr.tofile(tet_path)
        print(f"  Tet data: {tet_path} ({arr.shape}, {arr.dtype})")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Convert SF3D weights to WebGPU binary format")
    parser.add_argument("--model-path", default="stabilityai/stable-fast-3d",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output", default="public/weights.bin",
                        help="Output binary file path")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                        help="Output dtype (default: fp16)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    convert_weights(args.model_path, args.output, args.dtype)


if __name__ == "__main__":
    main()
