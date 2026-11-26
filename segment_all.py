# segment_all_classes.py
#
# Segments every raw photo in dataset/<class>/ into individual grain crops,
# removes the coin from each photo, and writes:
#   - segmented/<class>/<photo>_grainNNN.jpg     (individual crops)
#   - segmented/<class>_measurements.csv         (per-grain mm measurements)
#
# Usage:
#   python segment_all_classes.py --src dataset --dst segmented --coin-mm 24.0
#
# Expected input structure:
#   dataset/
#   ├── malagkit/      *.png | *.jpg
#   ├── dinorado/      *.png | *.jpg
#   ├── jasmine/       *.png | *.jpg
#   └── rice_defects/  *.png | *.jpg

import argparse
import csv
import glob
import os

import cv2
import numpy as np
from skimage.measure import regionprops, label


# ---------------------------------------------------------------
# Coin detection — HoughCircles with Otsu-largest-component fallback
# ---------------------------------------------------------------
def detect_coin(gray):
    """Returns (cx, cy, r_px) or None."""
    H, W = gray.shape
    min_r = max(20, int(min(H, W) * 0.02))
    max_r = int(min(H, W) * 0.30)
    if min_r >= max_r:
        return None

    # Primary: Hough circles
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min(H, W) // 4,
        param1=100, param2=30,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is not None and len(circles[0]) > 0:
        # pick the largest-radius detection
        c = max(circles[0], key=lambda c: c[2])
        return int(c[0]), int(c[1]), int(c[2])

    # Fallback: largest connected component in Otsu mask
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if n < 2:
        return None
    idx = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, w, h, _ = stats[idx]
    # Require it to be roughly circular (bbox ~square)
    if min(w, h) == 0 or abs(w - h) > 0.25 * max(w, h):
        return None
    return int(x + w / 2), int(y + h / 2), int((w + h) / 4)


# ---------------------------------------------------------------
# Process a single class folder
# ---------------------------------------------------------------
def process_class(class_name, src_dir, dst_dir, csv_path, coin_mm,
                  min_area_mm2=1.0, max_area_mm2=50.0, pad_px=6):
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.isdir(src_dir):
        print(f"  [{class_name}] folder not found: {src_dir} — skipping")
        return

    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    paths = []
    for e in exts:
        paths += glob.glob(f"{src_dir}/{e}")
    paths = sorted(set(paths))

    if not paths:
        print(f"  [{class_name}] no images in {src_dir}/ — skipping")
        return

    rows = []
    n_photos_ok = 0
    n_photos_skipped = 0

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip unreadable] {os.path.basename(path)}")
            n_photos_skipped += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        coin = detect_coin(gray)
        if coin is None:
            print(f"  [skip no coin ] {os.path.basename(path)}")
            n_photos_skipped += 1
            continue
        cx, cy, r_px = coin
        px_per_mm = (2 * r_px) / coin_mm
        mm2_per_px2 = 1.0 / (px_per_mm ** 2)

        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # wipe the coin and a margin for its shadow
        cv2.circle(thresh, (cx, cy), int(r_px * 1.10), 0, -1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                                  np.ones((3, 3), np.uint8), iterations=1)

        base = os.path.splitext(os.path.basename(path))[0]
        saved = 0
        for p in regionprops(label(thresh)):
            area_mm2 = p.area * mm2_per_px2
            if area_mm2 < min_area_mm2 or area_mm2 > max_area_mm2:
                continue

            if p.minor_axis_length <= 0:
                continue

            length_mm = p.major_axis_length / px_per_mm
            width_mm  = p.minor_axis_length / px_per_mm
            aspect    = p.major_axis_length / p.minor_axis_length

            y0, x0, y1, x1 = p.bbox
            x0 = max(0, x0 - pad_px); y0 = max(0, y0 - pad_px)
            x1 = min(W, x1 + pad_px); y1 = min(H, y1 + pad_px)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            crop_name = f"{base}_grain{saved:03d}.jpg"
            cv2.imwrite(f"{dst_dir}/{crop_name}", crop)

            rows.append({
                "class":        class_name,
                "source_photo": os.path.basename(path),
                "crop_file":    crop_name,
                "length_mm":    f"{length_mm:.3f}",
                "width_mm":     f"{width_mm:.3f}",
                "aspect":       f"{aspect:.3f}",
                "area_mm2":     f"{area_mm2:.3f}",
                "px_per_mm":    f"{px_per_mm:.3f}",
                "centroid_x":   f"{p.centroid[1]:.1f}",
                "centroid_y":   f"{p.centroid[0]:.1f}",
            })
            saved += 1

        n_photos_ok += 1
        print(f"  [{class_name}] {base}: {saved} grains "
              f"(px_per_mm={px_per_mm:.2f})")

    # Write CSV (even if empty — makes the file exist so downstream scripts don't crash)
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["class", "source_photo", "crop_file",
                      "length_mm", "width_mm", "aspect", "area_mm2",
                      "px_per_mm", "centroid_x", "centroid_y"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"  [{class_name}] SUMMARY: {len(rows)} grains from "
          f"{n_photos_ok} photos ({n_photos_skipped} skipped) → {csv_path}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset",
                    help="root folder containing <class>/ subfolders")
    ap.add_argument("--dst", default="segmented",
                    help="output folder for crops + CSVs")
    ap.add_argument("--coin-mm", type=float, required=True,
                    help="real diameter of your coin in mm "
                         "(1-peso PH=24.0, 5-peso PH=27.0, 10-peso PH=27.0, "
                         "US quarter=24.26)")
    ap.add_argument("--classes", nargs="+",
                    default=["MALAGKIT", "DINORADO", "JASMINE", "DEFECTS"])
    ap.add_argument("--min-area-mm2", type=float, default=1.0,
                    help="drop blobs smaller than this (dust)")
    ap.add_argument("--max-area-mm2", type=float, default=50.0,
                    help="drop blobs larger than this (merged grains, coin miss)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    print(f"Source : {args.src}")
    print(f"Output : {args.dst}")
    print(f"Coin   : {args.coin_mm} mm")
    print(f"Classes: {args.classes}")
    print("=" * 60)

    for cls in args.classes:
        print(f"\n=== {cls} ===")
        process_class(
            class_name=cls,
            src_dir=f"{args.src}/{cls}",
            dst_dir=f"{args.dst}/{cls}",
            csv_path=f"{args.dst}/{cls}_measurements.csv",
            coin_mm=args.coin_mm,
            min_area_mm2=args.min_area_mm2,
            max_area_mm2=args.max_area_mm2,
        )

    print("\n" + "=" * 60)
    print("Done.")