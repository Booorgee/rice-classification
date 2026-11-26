# jasmine_check.py
# Classifies pre-segmented rice grain crops in a folder.
#
# Usage:
#   python jasmine_check.py --model densenet121 --folder jasmine_cropped
#   python jasmine_check.py --model tf_efficientnetv2_s.in1k --folder jasmine_cropped
#
# Optional:
#   --out annotated.jpg     write an annotated contact sheet (one image, tiles of up to 100 crops)
#   --csv results.csv       write per-crop predictions

import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms


def load_model(model_name, runs_dir, device):
    ckpt_path  = f"{runs_dir}/{model_name}_best.pt"
    class_path = f"{runs_dir}/{model_name}_classes.json"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"classes json not found: {class_path}")

    with open(class_path) as f:
        classes = json.load(f)

    model = timm.create_model(model_name, pretrained=False,
                              num_classes=len(classes))
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return model, classes, tf


@torch.no_grad()
def classify_crop(model, tf, device, crop_bgr):
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    x = tf(pil).unsqueeze(0).to(device)
    prob = torch.softmax(model(x), 1)[0].cpu().numpy()
    return int(prob.argmax()), float(prob.max()), prob


def build_contact_sheet(rows, cols=10, tile=120):
    """Tile up to rows*cols crops into one annotated image for visual inspection."""
    n = len(rows)
    if n == 0:
        return None
    R = (n + cols - 1) // cols
    canvas = np.full((R * tile, cols * tile, 3), 30, dtype=np.uint8)
    for i, (path, crop, label, conf, color) in enumerate(rows[: R * cols]):
        r, c = divmod(i, cols)
        thumb = cv2.resize(crop, (tile, tile), interpolation=cv2.INTER_AREA)
        canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = thumb
        cv2.rectangle(canvas,
                      (c * tile, r * tile),
                      (c * tile + tile - 1, r * tile + tile - 1),
                      color, 1)
        cv2.putText(canvas, f"{label[:3]} {conf:.2f}",
                    (c * tile + 3, r * tile + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="tf_efficientnetv2_s.in1k or densenet121")
    parser.add_argument("--folder", required=True,
                        help="folder of pre-cropped rice grains (one per image)")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default=None,
                        help="optional annotated contact sheet path")
    parser.add_argument("--csv", default=None,
                        help="optional per-crop results CSV")
    parser.add_argument("--low-confidence", type=float, default=0.70,
                        help="flag crops below this confidence in the summary")
    args = parser.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    model, classes, tf = load_model(args.model, args.runs, device)
    if "jasmine" not in classes:
        raise ValueError(f"'jasmine' not in classes: {classes}")
    jasmine_idx = classes.index("jasmine")
    color_map = {classes[0]: (0, 200, 0), classes[1]: (0, 0, 220)}  # BGR

    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Classes: {classes}")
    print(f"Folder : {args.folder}")

    paths = sorted(
        glob.glob(f"{args.folder}/*.jpg")
        + glob.glob(f"{args.folder}/*.jpeg")
        + glob.glob(f"{args.folder}/*.png")
        + glob.glob(f"{args.folder}/*.JPG")
        + glob.glob(f"{args.folder}/*.PNG")
    )
    if not paths:
        print(f"No images found in {args.folder}/")
        return

    counts = {c: 0 for c in classes}
    rows_csv = []
    rows_vis = []
    low_conf = []

    for path in paths:
        crop = cv2.imread(path)
        if crop is None:
            print(f"[skip unreadable] {path}")
            continue
        k, conf, probs = classify_crop(model, tf, device, crop)
        label = classes[k]
        counts[label] += 1
        p_jasmine = float(probs[jasmine_idx])

        rows_csv.append({
            "file": os.path.basename(path),
            "label": label,
            "confidence": f"{conf:.4f}",
            "p_jasmine": f"{p_jasmine:.4f}",
        })
        rows_vis.append((path, crop, label, conf, color_map.get(label, (200, 200, 0))))

        if conf < args.low_confidence:
            low_conf.append((os.path.basename(path), label, conf, p_jasmine))

    n_total = sum(counts.values())
    if n_total == 0:
        print("No images could be read.")
        return

    print("\n" + "=" * 56)
    print(f"TOTAL GRAINS: {n_total}")
    for c, v in counts.items():
        print(f"  {c:10s}: {v:4d}  ({v / n_total * 100:5.1f}%)")
    pct = counts["jasmine"] / n_total * 100
    print(f"\n>>> JASMINE: {pct:.1f}%  "
          f"({counts['jasmine']}/{n_total})")
    print("=" * 56)

    if low_conf:
        print(f"\nLow-confidence crops (< {args.low_confidence:.2f}) — "
              f"{len(low_conf)} of {n_total}:")
        for name, lab, conf, pj in sorted(low_conf, key=lambda x: x[2])[:15]:
            print(f"  {name:40s} -> {lab:10s} "
                  f"(p={conf:.3f}, p_jasmine={pj:.3f})")
        if len(low_conf) > 15:
            print(f"  ... and {len(low_conf) - 15} more")
        print("  → look at these: they're where the model is least sure.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "label",
                                              "confidence", "p_jasmine"])
            w.writeheader()
            w.writerows(rows_csv)
        print(f"\nSaved per-crop CSV: {args.csv}")

    if args.out:
        sheet = build_contact_sheet(rows_vis)
        if sheet is not None:
            cv2.imwrite(args.out, sheet)
            print(f"Saved annotated contact sheet: {args.out}")


if __name__ == "__main__":
    main()