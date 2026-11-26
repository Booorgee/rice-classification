# make_figure.py
#
# Renders the segmentation pipeline stages of one photo as figure panels for
# the paper, using the exact functions in segment_grains.py:
#   (a) original            (b) illumination-corrected (background subtracted)
#   (c) binary mask         (d) result: kept grains green, rejected red, coin blue
#   (e) resulting 224x224 crops
#
# Usage:
#   .venv/bin/python make_figure.py --photo DATASET/JASMINE/345A8883.JPG
#
# Writes figures/<photo>/: a_original.png ... e_crops.png, figure.png (all
# panels combined) and a_original_full.png (whole frame with the zoom area).

import argparse
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np

from segment_grains import (build_parser, class_limits, foreground_mask,
                            process_photo)

WORK = 0.5  # working scale used by process_photo


def label(img, text, width):
    """Resize a panel to `width` and put a caption bar above it."""
    h = int(img.shape[0] * width / img.shape[1])
    img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bar = np.full((70, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0),
                2, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", default="DATASET/JASMINE/345A8883.JPG")
    ap.add_argument("--seg-params", default="grains/seg_params.json")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--defects", action="store_true",
                    help="use the broken-grain limits (DEFECTS photos)")
    ap.add_argument("--panel-width", type=int, default=1200)
    args = ap.parse_args()

    seg = vars(build_parser().parse_args([]))
    if os.path.exists(args.seg_params):
        with open(args.seg_params) as f:
            seg.update(json.load(f))
    seg = SimpleNamespace(**seg)

    res, err = process_photo(args.photo, seg, class_limits(seg, args.defects))
    if err:
        raise SystemExit(f"{args.photo}: {err}")
    img = res["img"]
    H, W = img.shape[:2]

    # intermediate images, from the same function process_photo uses
    small = cv2.resize(img, None, fx=WORK, fy=WORK, interpolation=cv2.INTER_AREA)
    steps = {}
    foreground_mask(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), WORK, steps)

    # zoom area: all kept grains + coin, with a margin
    pts = np.vstack([g["contour"].reshape(-1, 2) for g in res["grains"]])
    if res["coin"] is not None:
        cx, cy, r = res["coin"]
        pts = np.vstack([pts, [[cx - r, cy - r], [cx + r, cy + r]]])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    m = 0.12 * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
    x1, y1 = int(min(W, x1 + m)), int(min(H, y1 + m))
    roi = lambda a, k=1.0: a[int(y0 * k):int(y1 * k), int(x0 * k):int(x1 * k)]

    # (b) is dim (values above background only): stretch for display
    corr = steps["corrected"].astype(np.float32)
    corr = np.clip(corr * 255.0 / max(np.percentile(corr, 99.9), 1), 0, 255)
    corr = corr.astype(np.uint8)

    # (d) result overlay, drawn at full resolution
    res_vis = img.copy()
    lw = max(2, int(W / 1000))
    if res["coin"] is not None:
        cv2.circle(res_vis, (int(cx), int(cy)), int(r), (255, 80, 0), lw * 2)
    cv2.drawContours(res_vis, [g["contour"] for g in res["grains"]], -1,
                     (0, 255, 0), lw)
    cv2.drawContours(res_vis, [c for c, _ in res["rejects"]], -1,
                     (0, 0, 255), lw)

    # (e) crops of this photo, as saved in grains/
    crops = [g["crop"] for g in res["grains"][:10]]
    strip = np.hstack([np.pad(c, ((4, 4), (4, 4), (0, 0)),
                              constant_values=255) for c in crops])

    name = os.path.splitext(os.path.basename(args.photo))[0]
    out = f"{args.out}/{name}"
    os.makedirs(out, exist_ok=True)
    pw = args.panel_width
    n_rej = len(res["rejects"])
    panels = {
        "a_original": label(roi(img), "(a) Original image", pw),
        "b_corrected": label(roi(corr, WORK), "(b) Illumination-corrected", pw),
        "c_binary": label(roi(steps["binary"], WORK),
                          f"(c) Binary mask (Otsu, T={steps['threshold']:.0f})",
                          pw),
        "d_result": label(roi(res_vis),
                          f"(d) Result: {len(res['grains'])} grains kept"
                          + (f", {n_rej} rejected" if n_rej else ""), pw),
    }
    for k, v in panels.items():
        cv2.imwrite(f"{out}/{k}.png", v)
    e = label(strip, "(e) Extracted grain images (224 x 224 px)", 2 * pw + 20)
    cv2.imwrite(f"{out}/e_crops.png", e)

    gap = np.full((panels["a_original"].shape[0], 20, 3), 255, np.uint8)
    top = np.hstack([panels["a_original"], gap, panels["b_corrected"]])
    mid = np.hstack([panels["c_binary"], gap, panels["d_result"]])
    sep = np.full((20, top.shape[1], 3), 255, np.uint8)
    cv2.imwrite(f"{out}/figure.png", np.vstack([top, sep, mid, sep, e]))

    full = img.copy()
    cv2.rectangle(full, (x0, y0), (x1, y1), (0, 255, 255), lw * 3)
    cv2.imwrite(f"{out}/a_original_full.png",
                cv2.resize(full, None, fx=0.3, fy=0.3,
                           interpolation=cv2.INTER_AREA))

    print(f"{args.photo}: {len(res['grains'])} grains kept, {n_rej} rejected, "
          f"Otsu T={steps['threshold']:.0f}, {res['px_per_mm']:.1f} px/mm")
    print(f"Saved panels to {out}/ (figure.png = all combined)")


if __name__ == "__main__":
    main()
