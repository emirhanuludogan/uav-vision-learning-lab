"""Audit PNG pairs without changing files; run with python -m scripts.audit_uavid."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

# RGB values from the official UAVidToolKit colorTransformer.py.
COLORS = {
    "clutter": (0, 0, 0),
    "building": (128, 0, 0),
    "road": (128, 64, 128),
    "tree": (0, 128, 0),
    "low_vegetation": (128, 128, 0),
    "moving_car": (64, 0, 128),
    "static_car": (192, 0, 192),
    "human": (64, 64, 0),
}


def png_files(directory):
    return {p.name: p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"}


def audit(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("Veri kökü mevcut bir klasör olmalı.")
    sequences = []
    errors = []
    totals = Counter({name: 0 for name in COLORS})
    unknown = 0
    # Locate both sides so an orphan Labels directory is reported too.
    parents = {p.parent for p in root.rglob("*")
               if p.is_dir() and p.name in {"Images", "Labels"}}
    if not parents:
        errors.append("Images veya Labels klasörü bulunamadı.")
    by_color = {rgb: name for name, rgb in COLORS.items()}
    for sequence in sorted(parents):
        image_dir, label_dir = sequence / "Images", sequence / "Labels"
        images = png_files(image_dir) if image_dir.is_dir() else {}
        labels = png_files(label_dir) if label_dir.is_dir() else {}
        rel = sequence.relative_to(root).as_posix()
        missing_labels = sorted(images.keys() - labels.keys())
        missing_images = sorted(labels.keys() - images.keys())
        entry = {"sequence": rel, "images": len(images), "labels": len(labels),
                 "missing_labels": missing_labels, "missing_images": missing_images,
                 "checked_pairs": 0, "image_sizes": {}}
        if not images or not labels or missing_labels or missing_images:
            errors.append(f"{rel}: boş klasör veya eksik eşleşme.")
        sizes = Counter()
        for name in sorted(images.keys() & labels.keys()):
            try:
                with Image.open(images[name]) as image:
                    image.load()
                    size = image.size
                with Image.open(labels[name]) as label:
                    label.load()
                    if label.size != size:
                        errors.append(f"{rel}/{name}: görüntü ve etiket boyutları farklı.")
                        continue
                    # P-mode palettes are decoded to RGB; grayscale masks are
                    # rejected instead of silently interpreting them as classes.
                    if label.mode not in {"RGB", "P"}:
                        errors.append(f"{rel}/{name}: desteklenmeyen etiket modu {label.mode}.")
                        continue
                    rgb = label.convert("RGB")
                    histogram = rgb.getcolors(maxcolors=rgb.width * rgb.height)
                    frame_unknown = 0
                    for count, color in histogram:
                        class_name = by_color.get(color)
                        if class_name is None:
                            frame_unknown += count
                        else:
                            totals[class_name] += count
                    unknown += frame_unknown
                    if frame_unknown:
                        errors.append(f"{rel}/{name}: {frame_unknown} bilinmeyen renkli piksel.")
                sizes[f"{size[0]}x{size[1]}"] += 1
                entry["checked_pairs"] += 1
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                errors.append(f"{rel}/{name}: okunamadı ({type(exc).__name__}).")
        entry["image_sizes"] = dict(sizes)
        sequences.append(entry)
    return {"sequences": sequences, "class_pixels": dict(totals),
            "unknown_pixels": unknown, "errors": errors, "ok": not errors,
            "scope": "Dosya/eşleşme/boyut/renk kontrolü; kaynak ve sızıntı doğrulaması değildir."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.root)
    except ValueError as exc:
        parser.error(str(exc))
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
