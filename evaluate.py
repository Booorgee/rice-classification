# evaluate.py
#
# Evaluates every trained model in runs/ (each <tag>_best.pt + <tag>_meta.json)
# on grains/val and grains/test, with the same preprocessing as train.py's
# validation transform, and writes a comparison.
#
# Usage:
#   .venv/bin/python evaluate.py                     # all models in runs/
#   .venv/bin/python evaluate.py --tags densenet121 densenet121_color
#
# Writes to eval/:
#   summary.csv              one row per model x split (accuracy, macro P/R/F1,
#                            per-class P/R/F1, dinorado<->jasmine errors)
#   <tag>_<split>.txt        confusion matrix + classification report
#   predictions.csv          one row per crop per model (for error analysis)

import argparse
import csv
import glob
import json
import os

import numpy as np
import timm
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def evaluate(model, ds, device, batch):
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), 1).cpu().numpy()
            probs.append(p)
            ps.extend(p.argmax(1).tolist())
            ys.extend(y.tolist())
    return np.array(ys), np.array(ps), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="grains")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--tags", nargs="+", default=None,
                    help="model tags (file prefixes in runs/); default: all")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    tags = args.tags or sorted(
        os.path.basename(p)[:-len("_meta.json")]
        for p in glob.glob(f"{args.runs}/*_meta.json")
        if os.path.exists(p.replace("_meta.json", "_best.pt")))
    os.makedirs(args.out, exist_ok=True)
    print(f"Device: {device}   models: {tags}")

    summary, pred_rows = [], []
    for tag in tags:
        with open(f"{args.runs}/{tag}_meta.json") as f:
            meta = json.load(f)
        classes = meta["classes"]
        model = timm.create_model(meta["model"], pretrained=False,
                                  num_classes=len(classes))
        model.load_state_dict(torch.load(f"{args.runs}/{tag}_best.pt",
                                         map_location="cpu"))
        model = model.to(device).eval()
        tf = transforms.Compose([
            transforms.Resize((meta["img"], meta["img"])),
            transforms.ToTensor(),
            transforms.Normalize(meta["mean"], meta["std"]),
        ])
        labels = list(range(len(classes)))
        di = classes.index("dinorado") if "dinorado" in classes else None
        ja = classes.index("jasmine") if "jasmine" in classes else None

        for split in args.splits:
            ds = datasets.ImageFolder(f"{args.data}/{split}", tf)
            if ds.classes != classes:
                raise RuntimeError(f"{tag}: classes {classes} != "
                                   f"{split} folders {ds.classes}")
            ys, ps, probs = evaluate(model, ds, device, args.batch)
            cm = confusion_matrix(ys, ps, labels=labels)
            p, r, f, _ = precision_recall_fscore_support(
                ys, ps, labels=labels, zero_division=0)
            row = {
                "model": meta["model"], "color": meta.get("color", "off"),
                "split": split, "n": len(ys),
                "accuracy": round(float(np.mean(ys == ps)), 4),
                "macro_precision": round(float(p.mean()), 4),
                "macro_recall": round(float(r.mean()), 4),
                "macro_f1": round(float(f.mean()), 4),
            }
            for c, pc, rc, fc in zip(classes, p, r, f):
                row[f"{c}_precision"] = round(float(pc), 4)
                row[f"{c}_recall"] = round(float(rc), 4)
                row[f"{c}_f1"] = round(float(fc), 4)
            if di is not None and ja is not None:
                row["din_as_jas"] = int(cm[di, ja])
                row["jas_as_din"] = int(cm[ja, di])
            summary.append(row)

            with open(f"{args.out}/{tag}_{split}.txt", "w") as fh:
                fh.write(f"{tag} on {split} ({len(ys)} images)\n\n")
                fh.write("Confusion matrix (rows=true, cols=pred), order: "
                         f"{classes}\n{cm}\n\n")
                fh.write(classification_report(
                    ys, ps, labels=labels, target_names=classes, digits=4,
                    zero_division=0))
            for (path, _), y, k, pr in zip(ds.samples, ys, ps, probs):
                pred_rows.append({"model": tag, "split": split,
                                  "file": os.path.relpath(path, args.data),
                                  "true": classes[y], "pred": classes[k],
                                  "confidence": round(float(pr.max()), 4)})
            print(f"  {tag:32s} {split:5s} acc {row['accuracy']:.4f}  "
                  f"macroF1 {row['macro_f1']:.4f}")

    with open(f"{args.out}/summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(f"{args.out}/predictions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pred_rows[0]))
        w.writeheader()
        w.writerows(pred_rows)
    print(f"\nSaved {args.out}/summary.csv, per-model reports and "
          f"predictions.csv")


if __name__ == "__main__":
    main()
