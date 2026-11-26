# segment_grains.py
#
# Segments every JPG in DATASET/<class>/ into individual rice-grain crops and
# writes them in the ImageFolder layout train.py expects:
#   grains/train/<class>/<photo>_gNNN.png
#   grains/val/<class>/<photo>_gNNN.png
#   grains/test/<class>/<photo>_gNNN.png
#   grains/seg_params.json             (settings, reused by test_photos.py)
#   grains/grains.csv                  (per-grain measurements + split)
#   grains/debug/<class>/<photo>.jpg   (overlay of what was kept / rejected)
#
# The train/val/test split (default 70/15/15) is done by SOURCE PHOTO, not by
# grain, so grains from one photo never appear in two sets (otherwise val/test
# accuracy is inflated).
#
# Crops are fixed-size squares centred on each grain at a common mm-per-pixel
# scale (using the coin), so grain size and shape are preserved after the
# Resize((224, 224)) in train.py. Pixels outside the grain are set to black,
# which removes neighbouring grains and cloth texture from each crop.
#
# Usage:
#   .venv/bin/python segment_grains.py
#   .venv/bin/python segment_grains.py --val-frac 0.15 --test-frac 0.15

import argparse
import csv
import glob
import json
import os
import random
import re

import cv2
import numpy as np


# ---------------------------------------------------------------
# Foreground mask: flatten uneven lighting, then threshold
# ---------------------------------------------------------------
def foreground_mask(gray, scale, steps=None):
    """steps: optional dict, filled with the intermediate images (used by
    make_figure.py to illustrate the pipeline)."""
    # Background estimate from a large median blur (window >> coin size),
    # computed at 1/8 resolution for speed then upsampled
    tiny = cv2.resize(gray, None, fx=0.125 / scale, fy=0.125 / scale,
                      interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(tiny, 81)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    diff = cv2.subtract(gray, bg)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(t, 25)  # never go below this: keeps faint chalk/scratches out
    _, m = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # fill holes (bright grains sometimes have darker centres)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    if steps is not None:
        steps.update(gray=gray, background=bg, corrected=diff, threshold=t,
                     binary=filled)
    return filled


def detect_coin(cnts):
    """Coin = the most circular large contour. Returns (cx, cy, r).

    Uses 4*pi*A/P^2 (perimeter-based), which is ~0.9 for the smooth coin edge
    but low for round piles of rice, whose outlines are jagged.
    """
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 2000:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        fill = a / (np.pi * r * r)
        circ = 4 * np.pi * a / cv2.arcLength(c, True) ** 2
        if fill > 0.75 and circ > 0.7 and (best is None or circ > best[3]):
            best = (cx, cy, r, circ)
    return None if best is None else best[:3]


# ---------------------------------------------------------------
# Process one photo
# ---------------------------------------------------------------
def process_photo(path, args, lim, work_scale=0.5):
    img = cv2.imread(path)
    if img is None:
        return None, "unreadable"
    return process_image(img, args, lim, work_scale)


def process_image(img, args, lim, work_scale=0.5, px_per_mm=None):
    """Segment a BGR image. px_per_mm: fixed scale from a calibrated rig;
    when given, the coin is optional (still excluded if it is in view)."""
    H, W = img.shape[:2]

    small = cv2.resize(img, None, fx=work_scale, fy=work_scale,
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask_s = foreground_mask(gray, work_scale)
    cnts, _ = cv2.findContours(mask_s, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)

    coin = detect_coin(cnts)
    if coin is not None:
        ccx, ccy, cr = (v / work_scale for v in coin)
        if px_per_mm is None:
            px_per_mm = 2 * cr / args.coin_mm
    elif px_per_mm is None:
        return None, "no coin"

    # Full-resolution mask for precise crops
    mask = cv2.resize(mask_s, (W, H), interpolation=cv2.INTER_NEAREST)

    grains, rejects = [], []
    margin = int(0.01 * min(H, W))
    for c in cnts:
        c = (c.astype(np.float32) / work_scale).astype(np.int32)
        area_px = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        area_mm2 = area_px / px_per_mm ** 2
        if area_mm2 < 1.0:          # dust / specks: ignore silently
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if coin is not None and np.hypot(cx - ccx, cy - ccy) < cr * 1.2:
            continue                # the coin itself

        reason = None
        if (x < margin or y < margin or x + w > W - margin
                or y + h > H - margin):
            reason = "edge"
        elif len(c) < 5:
            reason = "tiny"
        else:
            (_, _), (d1, d2), ang = cv2.fitEllipse(c)
            length_mm = max(d1, d2) / px_per_mm
            width_mm = min(d1, d2) / px_per_mm
            hull_a = cv2.contourArea(cv2.convexHull(c))
            solidity = area_px / hull_a if hull_a else 0
            if area_mm2 > lim["max_area"]:
                reason = "too big"
            elif area_mm2 < lim["min_area"]:
                reason = "too small"
            elif solidity < lim["solidity"]:
                reason = "touching"   # two grains merged: concave outline
            elif not (lim["min_len"] <= length_mm <= lim["max_len"]):
                reason = "length"
            elif not (lim["min_width"] <= width_mm <= 4.5):
                reason = "width"
        if reason:
            rejects.append((c, reason))
            continue
        grains.append(dict(contour=c, cx=cx, cy=cy, area_mm2=area_mm2,
                           length_mm=length_mm, width_mm=width_mm,
                           aspect=length_mm / width_mm, solidity=solidity))

    # Keep only the main spread of grains: link grains whose centres are
    # within --link-mm of each other and keep the largest group. This drops
    # strays that rolled off the piles of other rice at the edges of the shot.
    if grains and args.link_mm > 0:
        pts = np.array([(g["cx"], g["cy"]) for g in grains])
        link = args.link_mm * px_per_mm
        d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
        group = -np.ones(len(pts), int)
        for s in range(len(pts)):
            if group[s] >= 0:
                continue
            group[s] = s
            stack = [s]
            while stack:
                i = stack.pop()
                for j in np.nonzero((d[i] < link) & (group < 0))[0]:
                    group[j] = s
                    stack.append(j)
        main = np.bincount(group).argmax()
        for g, gid in zip(grains, group):
            if gid != main:
                rejects.append((g["contour"], "stray"))
        grains = [g for g, gid in zip(grains, group) if gid == main]

    # Fixed physical window -> same mm/pixel in every crop
    win = int(round(args.crop_mm * px_per_mm))
    half = win // 2
    padded = cv2.copyMakeBorder(img, half, half, half, half,
                                cv2.BORDER_CONSTANT, value=0)
    for g in grains:
        gm = np.zeros((H, W), np.uint8)
        cv2.drawContours(gm, [g["contour"]], -1, 255, -1)
        gm = cv2.dilate(gm, np.ones((5, 5), np.uint8))
        gm = cv2.copyMakeBorder(gm, half, half, half, half,
                                cv2.BORDER_CONSTANT, value=0)
        cx, cy = int(g["cx"]) + half, int(g["cy"]) + half
        crop = padded[cy - half:cy - half + win, cx - half:cx - half + win]
        cm = gm[cy - half:cy - half + win, cx - half:cx - half + win]
        if not args.keep_background:
            crop = cv2.bitwise_and(crop, crop, mask=cm)
        g["crop"] = cv2.resize(crop, (args.crop_px, args.crop_px),
                               interpolation=cv2.INTER_AREA)

    return dict(img=img, coin=(ccx, ccy, cr) if coin is not None else None,
                px_per_mm=px_per_mm,
                grains=grains, rejects=rejects), None


def save_debug(res, out_path, title=""):
    vis = res["img"].copy()
    if res["coin"] is not None:
        ccx, ccy, cr = res["coin"]
        cv2.circle(vis, (int(ccx), int(ccy)), int(cr), (255, 0, 0), 6)
    cv2.drawContours(vis, [g["contour"] for g in res["grains"]], -1,
                     (0, 255, 0), 4)
    for c, reason in res["rejects"]:
        cv2.drawContours(vis, [c], -1, (0, 0, 255), 4)
        x, y, _, _ = cv2.boundingRect(c)
        cv2.putText(vis, reason, (x, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    vis = cv2.resize(vis, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

    # summary banner above the photo: grain count + why others were skipped
    rej = {}
    for _, r in res["rejects"]:
        rej[r] = rej.get(r, 0) + 1
    k = vis.shape[1] / 1728               # 1728 = 0.3 x 5760 (DSLR width)
    lines = [
        (title, (255, 255, 255), 0.8),
        (f"Grains kept (green): {len(res['grains'])}", (0, 255, 0), 1.4),
        (f"Not kept (red): {len(res['rejects'])}"
         + (" - " + ", ".join(f"{r} {n}" for r, n in sorted(rej.items()))
            if rej else ""), (90, 90, 255), 0.8),
        (f"Scale: {res['px_per_mm']:.1f} px/mm"
         + ("  (blue circle = coin)" if res["coin"] is not None else ""),
         (200, 200, 200), 0.7),
    ]
    lines = [ln for ln in lines if ln[0]]
    row = int(48 * k)
    bar = np.zeros((row * len(lines) + int(20 * k), vis.shape[1], 3), np.uint8)
    for i, (txt, col, fs) in enumerate(lines):
        cv2.putText(bar, txt, (int(15 * k), int((i + 1) * row)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * k, col,
                    max(1, int(2 * k)), cv2.LINE_AA)
    vis = np.vstack([bar, vis])
    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------
# Shared with test_photos.py
# ---------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="DATASET")
    ap.add_argument("--dst", default="grains")
    ap.add_argument("--classes", nargs="+",
                    default=["DINORADO", "JASMINE", "MALAGKIT", "DEFECTS"])
    ap.add_argument("--defect-class", default="DEFECTS",
                    help="class whose photos hold broken grains: uses the "
                         "looser --defect-* size/shape limits")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of PHOTOS per class held out for val")
    ap.add_argument("--test-frac", type=float, default=0.15,
                    help="fraction of PHOTOS per class held out for test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coin-mm", type=float, default=23.0,
                    help="coin diameter in mm (PH 1-peso NGC = 23.0)")
    ap.add_argument("--crop-mm", type=float, default=11.0,
                    help="side of the square crop window in mm")
    ap.add_argument("--crop-px", type=int, default=224)
    ap.add_argument("--min-area-mm2", type=float, default=4.0)
    ap.add_argument("--max-area-mm2", type=float, default=25.0)
    ap.add_argument("--min-len-mm", type=float, default=3.5)
    ap.add_argument("--max-len-mm", type=float, default=10.5)
    ap.add_argument("--defect-min-area-mm2", type=float, default=1.5)
    ap.add_argument("--defect-min-len-mm", type=float, default=1.5)
    ap.add_argument("--link-mm", type=float, default=15.0,
                    help="max gap between grains of the main spread "
                         "(0 = keep every grain, no stray filtering)")
    ap.add_argument("--keep-background", action="store_true",
                    help="don't black out pixels outside the grain")
    ap.add_argument("--debug-only", action="store_true",
                    help="only redraw <dst>/debug/ images; leave the crops, "
                         "grains.csv and seg_params.json untouched")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process N photos per class (for testing)")
    return ap


def class_limits(args, loose):
    """Size/shape limits. loose=True keeps broken grains (short, angular)."""
    lim = dict(min_area=args.min_area_mm2, max_area=args.max_area_mm2,
               min_len=args.min_len_mm, max_len=args.max_len_mm,
               min_width=1.0, solidity=0.88)
    if loose:
        lim.update(min_area=args.defect_min_area_mm2,
                   min_len=args.defect_min_len_mm,
                   min_width=0.8, solidity=0.80)
    return lim


def list_photos(folder):
    return sorted(set(glob.glob(f"{folder}/*.JPG")
                      + glob.glob(f"{folder}/*.jpg")
                      + glob.glob(f"{folder}/*.jpeg")
                      + glob.glob(f"{folder}/*.png")))


def safe_name(path):
    # "DINORADO(1).JPG" -> "DINORADO_1"
    name0 = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9]+", "_", name0).strip("_")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
SPLITS = ("train", "val", "test")

# Varieties present in each defect photo, keyed by filename prefix. The
# DINORADO* and JASMINE* defect photos contain broken grains of both varieties
# mixed together; only the MALAGKIT one is a single variety. This only fills
# the "variety" column of grains.csv: the training label is "defects" either way.
DEFECT_VARIETIES = {
    "DINORADO": "jasmine+dinorado",
    "JASMINE": "jasmine+dinorado",
    "MALAGKIT": "malagkit",
}


def main():
    args = build_parser().parse_args()

    os.makedirs(args.dst, exist_ok=True)
    if not args.debug_only:
        # test_photos.py reads these so new photos get identical crops
        with open(f"{args.dst}/seg_params.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    rng = random.Random(args.seed)
    rows = []
    for cls in args.classes:
        paths = list_photos(f"{args.src}/{cls}")
        if args.limit:
            paths = paths[:args.limit]
        # Split by PHOTO: shuffle, then first n_test -> test, next n_val -> val
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(paths) * args.test_frac))
        n_val = max(1, round(len(paths) * args.val_frac))
        split_of = {p: "test" for p in shuffled[:n_test]}
        split_of.update({p: "val" for p in shuffled[n_test:n_test + n_val]})

        os.makedirs(f"{args.dst}/debug/{cls}", exist_ok=True)
        is_defect = cls == args.defect_class
        lim = class_limits(args, loose=is_defect)
        counts = dict.fromkeys(SPLITS, 0)
        for path in paths:
            name0 = os.path.splitext(os.path.basename(path))[0]
            base = safe_name(path)
            # defect photos are named by prefix ("DINORADO(1)"); see
            # DEFECT_VARIETIES for which varieties each one actually holds
            prefix = re.sub(r"\W*\d*\W*$", "", name0).upper()
            variety = (DEFECT_VARIETIES.get(prefix, "unknown")
                       if is_defect else cls.lower())
            res, err = process_photo(path, args, lim)
            if err:
                print(f"  [{cls}] {base}: SKIPPED ({err})")
                continue
            split = split_of.get(path, "train")
            save_debug(res, f"{args.dst}/debug/{cls}/{base}.jpg",
                       f"{cls} / {os.path.basename(path)}   split: {split}")
            if args.debug_only:
                print(f"  [{cls}] {base}: {len(res['grains'])} grains")
                continue
            out_dir = f"{args.dst}/{split}/{cls.lower()}"
            os.makedirs(out_dir, exist_ok=True)
            for i, g in enumerate(res["grains"]):
                name = f"{base}_g{i:03d}.png"
                cv2.imwrite(f"{out_dir}/{name}", g["crop"])
                rows.append({
                    "split": split, "class": cls.lower(),
                    "variety": variety,
                    "source_photo": os.path.basename(path), "crop_file": name,
                    "length_mm": f"{g['length_mm']:.3f}",
                    "width_mm": f"{g['width_mm']:.3f}",
                    "aspect": f"{g['aspect']:.3f}",
                    "area_mm2": f"{g['area_mm2']:.3f}",
                    "solidity": f"{g['solidity']:.3f}",
                    "px_per_mm": f"{res['px_per_mm']:.3f}",
                })
            counts[split] += len(res["grains"])
            rej = {}
            for _, r in res["rejects"]:
                rej[r] = rej.get(r, 0) + 1
            print(f"  [{cls}] {base} ({split}): {len(res['grains'])} grains, "
                  f"px/mm={res['px_per_mm']:.1f}, rejected={rej or '-'}")
        if args.debug_only:
            continue
        tot = sum(counts.values()) or 1
        print(f"[{cls}] " + "  ".join(
            f"{s}={counts[s]} ({counts[s] / tot:.0%})" for s in SPLITS)
            + f"  | photos: test {n_test}, val {n_val}, "
              f"train {len(paths) - n_test - n_val}\n")

    if args.debug_only:
        print(f"Done. Debug images -> {args.dst}/debug/")
        return
    with open(f"{args.dst}/grains.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"Done. {len(rows)} grains -> {args.dst}/  "
          f"(check {args.dst}/debug/ overlays)")


if __name__ == "__main__":
    main()
