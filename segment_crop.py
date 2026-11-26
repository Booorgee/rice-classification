# segment_crop.py
# Reads all photos from a folder, segments each grain, saves one crop per grain.
# Usage:
#   python segment_crop.py --src my_jasmine_photos --dst my_jasmine_crops

import argparse
import glob
import os

import cv2
import numpy as np


def segment_and_crop(src_dir, dst_dir, min_area=100, coin_frac=0.02, pad=6):
    os.makedirs(dst_dir, exist_ok=True)

    paths = sorted(
        glob.glob(f"{src_dir}/*.jpg")
        + glob.glob(f"{src_dir}/*.jpeg")
        + glob.glob(f"{src_dir}/*.png")
    )
    if not paths:
        print(f"No images found in {src_dir}/")
        return

    total_grains = 0
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[skip] cannot read {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        H, W = gray.shape
        coin_area = H * W * coin_frac

        n, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8)

        base = os.path.splitext(os.path.basename(path))[0]
        saved = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area:          # dust / noise
                continue
            if area > coin_area:         # coin / giant blob — skip
                print(f"[skip] blob area {area} on {base} (likely coin)")
                continue

            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(W, x + w + pad)
            y1 = min(H, y + h + pad)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            out_path = f"{dst_dir}/{base}_grain{saved:03d}.jpg"
            cv2.imwrite(out_path, crop)
            saved += 1

        print(f"{base}: {saved} grains")
        total_grains += saved

    print(f"\nDone. Total grains saved: {total_grains} → {dst_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="my_jasmine_photos")
    parser.add_argument("--dst", default="my_jasmine_crops")
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--coin-frac", type=float, default=0.02,
                        help="skip blobs larger than this fraction of image")
    parser.add_argument("--pad", type=int, default=6)
    args = parser.parse_args()

    segment_and_crop(args.src, args.dst,
                     min_area=args.min_area,
                     coin_frac=args.coin_frac,
                     pad=args.pad)