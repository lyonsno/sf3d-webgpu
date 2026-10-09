"""Regenerate small Pillow references for the browser's foreground preparation."""
import json
from pathlib import Path
from PIL import Image, __version__
from prepare_demo_ab import official_frame

cases = []
for name, width, height, size in [("downsample", 35, 27, 8), ("upsample", 7, 9, 16), ("opaque", 8, 6, 8)]:
    image = Image.new("RGBA", (width, height))
    for y in range(height):
        for x in range(width):
            alpha = 255 if name == "opaque" else (0 if x < 2 or y < 1 else (x * 43 + y * 19) % 256)
            image.putpixel((x, y), ((x * 51) % 256, (y * 73) % 256, (x * 23 + y * 37) % 256, alpha))
    output, crop = official_frame(image, size=size)
    cases.append(dict(name=name, width=width, height=height, size=size, crop=crop,
                      input=list(image.tobytes()), output=list(output.tobytes())))
target = Path(__file__).parent / "fixtures" / "foreground-reference.json"
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(dict(pillow=__version__,
    upstream="Stability-AI/stable-fast-3d@ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2 sf3d/utils.py resize_foreground",
    cases=cases)) + "\n")
