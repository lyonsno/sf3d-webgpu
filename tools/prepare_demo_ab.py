"""Prepare an exploratory SF3D framing A/B; production preprocessing is unchanged."""
from PIL import Image
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def official_frame(image, ratio=0.85, size=512):
    if image.mode != "RGBA":
        raise ValueError("RGBA input required")
    bounds = image.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("Empty mask")
    x1, y1, right, bottom = bounds
    # Upstream's flatnonzero returns inclusive maxima, unlike Pillow getbbox.
    x2, y2 = right - 1, bottom - 1
    scale = max(y2 - y1, x2 - x1) / ratio
    left = int((x1 + x2) / 2 - scale / 2)
    top = int((y1 + y2) / 2 - scale / 2)
    side = int(scale)
    if side < 1:
        raise ValueError("Foreground crop is empty")
    box = (left, top, left + side, top + side)
    return image.crop(box).resize((size, size), Image.Resampling.BICUBIC), box


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    for name in ["animal_character", "tree"]:
        source = args.inputs / (name + ".png")
        with Image.open(source) as image:
            framed, crop = official_frame(image)
            for arm in ["current", "official-framing"]:
                target = args.out / f"{name}-{arm}.png"
                if arm == "current":
                    shutil.copyfile(source, target)
                else:
                    framed.save(target)
                cases.append({"id": target.stem, "input": str(target.resolve()), "arm": arm,
                              "source": str(source.resolve()),
                              "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                              "inputSha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                              "crop": crop if arm != "current" else None,
                              "foregroundRatio": 0.85 if arm != "current" else None})
    (args.out / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(json.dumps(cases, indent=2))
