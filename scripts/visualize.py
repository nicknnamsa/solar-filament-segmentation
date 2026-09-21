"""Overlay filament annotations on the solar images.

Each output PNG shows the raw image followed by one panel per annotator (the same file can be
annotated up to 3 times). By default each filament instance gets its own color and number; with
--color-by category, masks are colored by Left/Right/Unidentifiable/Ambiguous instead. Spines are white lines.

    python scripts/visualize.py --n 6                  # 6 random training files
    python scripts/visualize.py --file 20140609195854Bh.jpeg
    python scripts/visualize.py --n 6 --no-spine
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

CATEGORY_COLORS = {
    1: (255, 80, 80),     # Left
    2: (80, 200, 255),    # Right
    3: (255, 220, 60),    # Unidentifiable
    4: (200, 120, 255),   # Ambiguous
}
INSTANCE_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40),
]
PANEL_SIZE = 1024


def load_image(path):
    """Grayscale image with a percentile contrast stretch so faint structures are visible."""
    img = Image.open(path).convert("L")
    lo, hi = img.getextrema()
    hist = img.histogram()
    total, acc, p_lo, p_hi = sum(hist), 0, lo, hi
    for v, c in enumerate(hist):
        acc += c
        if acc >= 0.01 * total:
            p_lo = v
            break
    acc = 0
    for v in range(255, -1, -1):
        acc += hist[v]
        if acc >= 0.01 * total:
            p_hi = v
            break
    scale = 255.0 / max(p_hi - p_lo, 1)
    return img.point(lambda p: max(0, min(255, int((p - p_lo) * scale))))


def draw_panel(base, anns, scale, show_spine, color_by):
    panel = base.convert("RGBA")
    fill = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill)
    line_draw = ImageDraw.Draw(fill)
    for i, a in enumerate(anns):
        if color_by == "instance":
            color = INSTANCE_PALETTE[i % len(INSTANCE_PALETTE)]
        else:
            color = CATEGORY_COLORS.get(a["category_id"], (255, 255, 255))
        for poly in a["segmentation"]:
            pts = [(x * scale, y * scale) for x, y in zip(poly[0::2], poly[1::2])]
            if len(pts) >= 3:
                fill_draw.polygon(pts, fill=color + (110,), outline=color + (255,))
        if color_by == "instance":
            xs = [x * scale for poly in a["segmentation"] for x in poly[0::2]]
            ys = [y * scale for poly in a["segmentation"] for y in poly[1::2]]
            line_draw.text((min(xs), min(ys) - 12), str(i + 1), fill=color + (255,))
        if show_spine and a.get("spine"):
            s = a["spine"]
            pts = [(x * scale, y * scale) for x, y in zip(s[0::2], s[1::2])]
            if len(pts) >= 2:
                line_draw.line(pts, fill=(255, 255, 255, 255), width=1)
    return Image.alpha_composite(panel, fill).convert("RGB")


def label(img, text):
    ImageDraw.Draw(img).text((8, 8), text, fill=(255, 255, 255))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maskrcnn_baseline.yaml")
    parser.add_argument("--file", help="file name to visualize, e.g. 20140609195854Bh.jpeg")
    parser.add_argument("--n", type=int, default=6, help="number of random files if --file is not given")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--color-by", choices=["instance", "category"], default="instance",
                        help="instance: one color + number per filament; category: Left/Right/Unidentifiable/Ambiguous")
    parser.add_argument("--no-spine", action="store_true")
    parser.add_argument("--out", default="outputs/visualizations")
    args = parser.parse_args()

    with open(args.config) as f:
        d = yaml.safe_load(f)["data"]
    with open(d["annotations"]) as f:
        coco = json.load(f)

    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)
    entries_by_file = defaultdict(list)
    for im in coco["images"]:
        entries_by_file[im["file_name"]].append(im["id"])

    files = [args.file] if args.file else random.Random(args.seed).sample(sorted(entries_by_file), args.n)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fn in files:
        base = load_image(Path(d["train_images"]) / fn)
        scale = PANEL_SIZE / base.width
        base = base.resize((PANEL_SIZE, round(base.height * scale)), Image.LANCZOS)
        panels = [label(base.convert("RGB"), fn)]
        for image_id in sorted(entries_by_file[fn]):
            anns = anns_by_image[image_id]
            panels.append(label(draw_panel(base, anns, scale, not args.no_spine, args.color_by),
                                f"{image_id.split('-')[0]}  ({len(anns)} filaments)"))
        sheet = Image.new("RGB", (PANEL_SIZE * len(panels), panels[0].height))
        for i, p in enumerate(panels):
            sheet.paste(p, (i * PANEL_SIZE, 0))
        path = out_dir / f"{Path(fn).stem}.png"
        sheet.save(path)
        print(path)


if __name__ == "__main__":
    main()
