# test_photos.py
#
# Robustness test on a NEW batch of whole photos: segments each photo with the
# same settings used to build the training crops (grains/seg_params.json),
# classifies every grain with a trained model, and reports how it did.
#
# Put the new photos in a folder. If they are sorted into sub-folders named
# after the classes, the folder name is used as the true label for every grain
# in that photo and accuracy / confusion matrix are reported:
#   TEST/
#   ├── DINORADO/  *.JPG
#   ├── JASMINE/   *.JPG
#   ├── MALAGKIT/  *.JPG
#   ├── DEFECTS/   *.JPG
#   └── *.JPG      (loose photos: unlabeled, composition only)
#
# Usage:
#   .venv/bin/python test_photos.py --model densenet121 --photos TEST
#   .venv/bin/python test_photos.py --model tf_efficientnetv2_s.in1k --photos TEST
#   .venv/bin/python test_photos.py --model densenet121 --color on --photos TEST
#
# Writes to --out (default test_results/<model>[_color]/):
#   predictions.csv          one row per grain
#   annotated/<photo>.jpg    each grain outlined + labelled with its prediction
#
# These photos must NOT be ones already in DATASET/: the model has seen those.

import argparse
import csv
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms

from segment_grains import (build_parser, class_limits, list_photos,
                            process_photo, safe_name)

# BGR colours for the annotated photos, by class index
PALETTE = [(80, 200, 80), (60, 160, 255), (255, 160, 60), (60, 60, 230),
           (200, 80, 200), (200, 200, 60)]


def load_model(model_name, runs_dir, device, color="off"):
    tag = model_name.replace("/", "_") + ("_color" if color == "on" else "")
    ckpt_path = f"{runs_dir}/{tag}_best.pt"
    meta_path = f"{runs_dir}/{tag}_meta.json"
    for p in (ckpt_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found — train the model first")
    with open(meta_path) as f:
        meta = json.load(f)

    model = timm.create_model(meta["model"], pretrained=False,
                              num_classes=len(meta["classes"]))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((meta["img"], meta["img"])),
        transforms.ToTensor(),
        transforms.Normalize(meta["mean"], meta["std"]),
    ])
    return model, meta["classes"], tf


@torch.no_grad()
def classify(model, tf, device, crops, batch=64):
    probs = []
    for i in range(0, len(crops), batch):
        x = torch.stack([
            tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
            for c in crops[i:i + batch]
        ]).to(device)
        probs.append(torch.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0, 0))


def annotate(res, preds, confs, classes, low_conf, out_path):
    vis = res["img"].copy()
    for c, _ in res["rejects"]:
        cv2.drawContours(vis, [c], -1, (128, 128, 128), 3)
    for g, k, conf in zip(res["grains"], preds, confs):
        col = PALETTE[k % len(PALETTE)]
        cv2.drawContours(vis, [g["contour"]], -1, col, 5)
        x, y, _, _ = cv2.boundingRect(g["contour"])
        txt = classes[k][:3].upper() + ("?" if conf < low_conf else "")
        cv2.putText(vis, txt, (x, max(30, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4, cv2.LINE_AA)
    # legend
    for i, name in enumerate(classes):
        cv2.putText(vis, name, (40, 80 + i * 70), cv2.FONT_HERSHEY_SIMPLEX,
                    2.2, PALETTE[i % len(PALETTE)], 5, cv2.LINE_AA)
    vis = cv2.resize(vis, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="tf_efficientnetv2_s.in1k or densenet121")
    ap.add_argument("--photos", required=True,
                    help="folder of new photos (optionally in class sub-folders)")
    ap.add_argument("--color", choices=["off", "on"], default="off",
                    help="on: use the model trained with train.py --color on")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--grains", default="grains",
                    help="folder with seg_params.json from segment_grains.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--low-confidence", type=float, default=0.70,
                    help="grains below this confidence get a '?' and are counted")
    ap.add_argument("--strict", action="store_true",
                    help="use the stricter whole-grain size limits; by default "
                         "broken grains are kept so defects can be detected")
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model, classes, tf = load_model(args.model, args.runs, device, args.color)

    # Same segmentation settings as the training crops
    seg = vars(build_parser().parse_args([]))
    params_path = f"{args.grains}/seg_params.json"
    if os.path.exists(params_path):
        with open(params_path) as f:
            seg.update(json.load(f))
    else:
        print(f"[warn] {params_path} not found — using segment_grains defaults")
    seg = SimpleNamespace(**seg)
    lim = class_limits(seg, loose=not args.strict)

    out_dir = args.out or (f"test_results/{args.model.replace('/', '_')}"
                           + ("_color" if args.color == "on" else ""))
    os.makedirs(f"{out_dir}/annotated", exist_ok=True)

    # (path, true class index or None)
    jobs = [(p, None) for p in list_photos(args.photos)]
    lower = [c.lower() for c in classes]
    for d in sorted(os.listdir(args.photos)):
        if not os.path.isdir(f"{args.photos}/{d}"):
            continue
        if d.lower() not in lower:
            print(f"[warn] sub-folder '{d}' is not a class {classes} — skipped")
            continue
        jobs += [(p, lower.index(d.lower()))
                 for p in list_photos(f"{args.photos}/{d}")]
    if not jobs:
        print(f"No photos found in {args.photos}/")
        return

    print(f"Device : {device}")
    print(f"Model  : {args.model}  color={args.color}  classes={classes}")
    print(f"Photos : {len(jobs)} from {args.photos}/")
    print(f"Limits : {'strict (whole grains)' if args.strict else 'loose (keeps broken grains)'}")
    print("=" * 78)

    rows, ys, ps = [], [], []
    n_skipped_photos, n_rejected, n_low = 0, 0, 0
    for path, true_k in jobs:
        res, err = process_photo(path, seg, lim)
        name = os.path.relpath(path, args.photos)
        if err:
            print(f"{name}: SKIPPED ({err})")
            n_skipped_photos += 1
            continue

        probs = classify(model, tf, device, [g["crop"] for g in res["grains"]])
        preds = probs.argmax(1) if len(probs) else np.array([], int)
        confs = probs.max(1) if len(probs) else np.array([])
        n_rejected += len(res["rejects"])
        n_low += int((confs < args.low_confidence).sum())

        for i, (g, k, conf) in enumerate(zip(res["grains"], preds, confs)):
            rows.append({
                "photo": name, "grain": i,
                "true": classes[true_k] if true_k is not None else "",
                "pred": classes[k], "confidence": f"{conf:.4f}",
                **{f"p_{c}": f"{probs[i, j]:.4f}" for j, c in enumerate(classes)},
                "length_mm": f"{g['length_mm']:.3f}",
                "width_mm": f"{g['width_mm']:.3f}",
                "area_mm2": f"{g['area_mm2']:.3f}",
                "centroid_x": f"{g['cx']:.0f}", "centroid_y": f"{g['cy']:.0f}",
            })
            if true_k is not None:
                ys.append(true_k)
                ps.append(int(k))

        annotate(res, preds, confs, classes, args.low_confidence,
                 f"{out_dir}/annotated/{safe_name(path)}.jpg")

        n = len(preds)
        mix = ", ".join(f"{c} {np.mean(preds == j):.0%}"
                        for j, c in enumerate(classes) if np.any(preds == j))
        acc = (f" | acc {np.mean(preds == true_k):.1%}"
               if true_k is not None and n else "")
        print(f"{name}: {n} grains ({len(res['rejects'])} not segmented) "
              f"-> {mix or '-'}{acc}")

    if rows:
        with open(f"{out_dir}/predictions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print("=" * 78)
    print(f"Grains classified : {len(rows)}")
    print(f"Not segmented     : {n_rejected} objects (touching / edge / size "
          f"/ stray — see grey outlines in annotated/)")
    print(f"Low confidence    : {n_low} grains (< {args.low_confidence:.2f})")
    if n_skipped_photos:
        print(f"Photos skipped    : {n_skipped_photos} (no coin found)")

    if ys:
        labels = list(range(len(classes)))
        print(f"\nAccuracy on labelled photos: "
              f"{np.mean(np.array(ys) == np.array(ps)):.2%} ({len(ys)} grains)")
        print("\nConfusion matrix (rows=true, cols=pred):")
        print("      " + " ".join(f"{c[:8]:>9s}" for c in classes))
        for c, r in zip(classes, confusion_matrix(ys, ps, labels=labels)):
            print(f"{c[:8]:>8s}" + "".join(f"{v:>10d}" for v in r))
        print("\nClassification report:")
        print(classification_report(ys, ps, labels=labels, target_names=classes,
                                    digits=4, zero_division=0))
        print("Note: a broken grain in e.g. a JASMINE/ photo counts as 'jasmine' "
              "here,\nso a 'defects' prediction on it may be correct — check "
              "annotated/.")

    print(f"\nSaved: {out_dir}/predictions.csv and {out_dir}/annotated/")


if __name__ == "__main__":
    main()
