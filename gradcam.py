# gradcam.py
#
# Grad-CAM heatmaps (Selvaraju et al., 2017) for the four trained models:
# which parts of a grain image drove each model's prediction.
#
# Grad-CAM weights the last convolutional feature map by the gradient of the
# predicted class score, giving a coarse map (7x7 for a 224 px input) that is
# upsampled to the image. Bright = regions that pushed the model towards the
# predicted class.
#
# Usage:
#   .venv/bin/python gradcam.py
#
# Writes figures/gradcam/:
#   gradcam_comparison.png   same test grains under all four models
#                            (grains the CP models fixed + easy classes)
#   gradcam_errors.png       DenseNet121-CP mistakes (Dinorado <-> Jasmine)
#   attention_on_grain.csv   share of the heatmap that falls on the grain
#                            (not the black background), all test grains

import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch

MODELS = [("densenet121", "DenseNet121-SA"),
          ("densenet121_color", "DenseNet121-CP"),
          ("tf_efficientnetv2_s.in1k", "EfficientNetV2-S-SA"),
          ("tf_efficientnetv2_s.in1k_color", "EfficientNetV2-S-CP")]
SHORT = {"defects": "Broken", "dinorado": "Dinorado", "jasmine": "Jasmine",
         "malagkit": "Malagkit"}
# single-hue (orange) ramp from the reference palette: transparent -> strong
HEAT = matplotlib.colors.LinearSegmentedColormap.from_list(
    "heat", ["#f6c3ac", "#eb6834", "#a8360c"])
INK, INK2 = "#0b0b0b", "#52514e"


DEVICE = "cpu"


def load(tag, runs):
    meta = json.load(open(f"{runs}/{tag}_meta.json"))
    m = timm.create_model(meta["model"], pretrained=False,
                          num_classes=len(meta["classes"]))
    m.load_state_dict(torch.load(f"{runs}/{tag}_best.pt", map_location="cpu"))
    return m.to(DEVICE).eval(), meta


def to_tensor(crops_bgr, meta):
    x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr])
    x = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor(meta["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(meta["std"]).view(1, 3, 1, 1)
    return ((x - mean) / std).to(DEVICE)


def gradcam(model, x, target=None):
    """Returns (cams in [0,1] at input size, predicted class, probs)."""
    feats = model.forward_features(x)
    feats.retain_grad()
    logits = model.forward_head(feats)
    probs = torch.softmax(logits, 1).detach()
    pred = logits.argmax(1)
    tgt = pred if target is None else target
    model.zero_grad()
    logits.gather(1, tgt[:, None]).sum().backward()
    w = feats.grad.mean((2, 3), keepdim=True)
    cam = torch.relu((w * feats).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:],
                                          mode="bilinear", align_corners=False)
    cam = cam[:, 0].detach().cpu().numpy()
    cam /= cam.reshape(len(cam), -1).max(1)[:, None, None] + 1e-8
    return cam, pred.cpu().numpy(), probs.cpu().numpy()


def overlay(ax, crop_bgr, cam, title, title_color=INK):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.imshow(cam, cmap=HEAT, alpha=np.clip(cam, 0, 1) * 0.85, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8, color=title_color, pad=3)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="grains")
    ap.add_argument("--pred", default="eval/predictions.csv")
    ap.add_argument("--out", default="figures/gradcam")
    ap.add_argument("--stats-only", action="store_true",
                    help="skip the figures, only compute attention_on_grain")
    args = ap.parse_args()
    global DEVICE
    DEVICE = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(True)

    # test predictions per model (from evaluate.py) to pick examples
    pred = defaultdict(dict)
    for r in csv.DictReader(open(args.pred)):
        if r["split"] == "test":
            pred[r["model"]][r["file"]] = r
    files = sorted(pred["densenet121"])
    true = {f: pred["densenet121"][f]["true"] for f in files}
    ok = lambda m, f: pred[m][f]["pred"] == true[f]

    # comparison rows: 2 Dinorado + 2 Jasmine that SA got wrong and CP right
    # (DenseNet), 1 Malagkit + 1 Broken that every model got right
    pick = []
    for cls in ("dinorado", "jasmine"):
        cand = [f for f in files if true[f] == cls and ok("densenet121_color", f)
                and not ok("densenet121", f)]
        pick += cand[::max(1, len(cand) // 2)][:2]
    for cls in ("malagkit", "defects"):
        cand = [f for f in files if true[f] == cls
                and all(ok(m, f) for m, _ in MODELS)]
        pick += cand[len(cand) // 2:len(cand) // 2 + 1]
    # errors of the best model
    errs = [f for f in files if not ok("densenet121_color", f)
            and true[f] in ("dinorado", "jasmine")]
    err_pick = (sorted([f for f in errs if true[f] == "dinorado"])[::12][:3]
                + sorted([f for f in errs if true[f] == "jasmine"])[::6][:3])

    crops = {f: cv2.imread(f"{args.data}/{f}") for f in files}
    loaded = {tag: load(tag, args.runs) for tag, _ in MODELS}

    # ---- figure 1: same grains, four models ----
    if args.stats_only:
        pick, err_pick = [], []
    fig, axes = plt.subplots(len(pick), 5, figsize=(9, 1.95 * len(pick)))
    for i, f in enumerate(pick):
        ax = axes[i, 0]
        ax.imshow(cv2.cvtColor(crops[f], cv2.COLOR_BGR2RGB))
        ax.set_title(f"True: {SHORT[true[f]]}", fontsize=8, color=INK, pad=3)
        ax.axis("off")
        for j, (tag, name) in enumerate(MODELS):
            model, meta = loaded[tag]
            cam, p, pr = gradcam(model, to_tensor([crops[f]], meta))
            lab = meta["classes"][p[0]]
            right = lab == true[f]
            overlay(axes[i, j + 1], crops[f], cam[0],
                    f"{SHORT[lab]} {pr[0].max() * 100:.0f}%"
                    + ("" if right else "  (wrong)"),
                    INK if right else "#d03b3b")
    for j, (_, name) in enumerate([("", "Original")] + MODELS):
        axes[0, j].text(0.5, 1.32, name, transform=axes[0, j].transAxes,
                        ha="center", fontsize=9, fontweight="bold", color=INK)
    fig.text(0.5, 0.004, "Orange = regions that contributed most to the "
             "predicted class (Grad-CAM). Label: prediction and confidence.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    fig.savefig(f"{args.out}/gradcam_comparison.png", dpi=300,
                facecolor="white")
    plt.close(fig)

    # ---- figure 2: mistakes of the best model ----
    model, meta = loaded["densenet121_color"]
    fig, axes = plt.subplots(2, len(err_pick), figsize=(1.6 * len(err_pick), 3.7))
    for j, f in enumerate(err_pick):
        cam, p, pr = gradcam(model, to_tensor([crops[f]], meta))
        axes[0, j].imshow(cv2.cvtColor(crops[f], cv2.COLOR_BGR2RGB))
        axes[0, j].set_title(f"True: {SHORT[true[f]]}", fontsize=8, pad=3)
        axes[0, j].axis("off")
        overlay(axes[1, j], crops[f], cam[0],
                f"Pred: {SHORT[meta['classes'][p[0]]]} {pr[0].max()*100:.0f}%",
                "#d03b3b")
    fig.suptitle("DenseNet121-CP misclassifications (test set)", fontsize=10,
                 color=INK)
    fig.tight_layout()
    fig.savefig(f"{args.out}/gradcam_errors.png", dpi=300, facecolor="white")
    plt.close(fig)

    # ---- attention on grain vs background, all test grains ----
    rows = []
    for tag, name in MODELS:
        model, meta = loaded[tag]
        shares = defaultdict(list)
        for i in range(0, len(files), 32):
            batch = files[i:i + 32]
            cam, _, _ = gradcam(model, to_tensor([crops[f] for f in batch], meta))
            for f, c in zip(batch, cam):
                grain = crops[f].max(2) > 15        # background is black
                shares[true[f]].append(c[grain].sum() / (c.sum() + 1e-8))
        allv = np.concatenate([v for v in shares.values()])
        row = {"model": name, "all": round(float(np.mean(allv)), 3)}
        row.update({SHORT[k]: round(float(np.mean(v)), 3)
                    for k, v in sorted(shares.items())})
        rows.append(row)
        print(f"{name:22s} heatmap on grain: {row['all']:.1%}  "
              + "  ".join(f"{k} {v:.1%}" for k, v in row.items()
                          if k not in ('model', 'all')))
    with open(f"{args.out}/attention_on_grain.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    grain_px = np.mean([np.mean(crops[f].max(2) > 15) for f in files])
    print(f"(grain covers {grain_px:.1%} of an average crop)")
    print(f"Saved figures to {args.out}/")


if __name__ == "__main__":
    main()
