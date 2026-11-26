# analyze.py
#
# Statistical comparison of the four models from evaluate.py's per-grain
# predictions (eval/predictions.csv). For the paper:
#
#   eval/stats/metrics_<split>.csv     accuracy, macro P/R/F1, Cohen's kappa,
#                                      per-class P/R/F1, with 95% CIs
#   eval/stats/pairwise_<split>.csv    McNemar test (z, p, Holm-corrected p)
#                                      for every pair of models, plus the
#                                      accuracy difference with its 95% CI
#   eval/stats/confusion_<split>.png   confusion matrices of the four models
#   eval/stats/comparison_<split>.png  macro-F1 per model with 95% CI
#
# Confidence intervals use a photo-level (cluster) bootstrap: photographs are
# resampled within each class, because grains from one photo are not
# independent. This is wider, and more honest, than resampling grains.
#
# Usage:
#   .venv/bin/python analyze.py                 # test split
#   .venv/bin/python analyze.py --split val

import argparse
import csv
import itertools
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)

# display names: architecture + SA (color randomly changed) / CP (color kept)
NAMES = {
    "densenet121": "DenseNet121-SA",
    "densenet121_color": "DenseNet121-CP",
    "tf_efficientnetv2_s.in1k": "EfficientNetV2-S-SA",
    "tf_efficientnetv2_s.in1k_color": "EfficientNetV2-S-CP",
}
CLASSES = ["defects", "dinorado", "jasmine", "malagkit"]
LABELS = {"defects": "Broken", "dinorado": "Dinorado", "jasmine": "Jasmine",
          "malagkit": "Malagkit"}

# reference palette (dataviz skill): sequential blue, categorical slots 1-2
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
SERIES = {"SA": "#2a78d6", "CP": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, p, labels=range(len(CLASSES)), zero_division=0)
    return {"accuracy": np.mean(y == p), "macro_precision": pr.mean(),
            "macro_recall": rc.mean(), "macro_f1": f1.mean(),
            "kappa": cohen_kappa_score(y, p),
            **{f"{c}_{m}": v[i] for i, c in enumerate(CLASSES)
               for m, v in (("precision", pr), ("recall", rc), ("f1", f1))}}


def photo_bootstrap(photo_of, true, B, seed=0):
    """Index arrays: photos resampled with replacement within each class."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(lambda: defaultdict(list))
    for i, (ph, t) in enumerate(zip(photo_of, true)):
        groups[t][ph].append(i)
    out = []
    for _ in range(B):
        idx = []
        for t, phs in groups.items():
            keys = list(phs)
            for k in rng.choice(len(keys), len(keys)):
                idx.extend(phs[keys[k]])
        out.append(np.array(idx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="eval/predictions.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="eval/stats")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [r for r in csv.DictReader(open(args.pred))
            if r["split"] == args.split and r["model"] in NAMES]
    by_model = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["file"]] = r
    models = [m for m in NAMES if m in by_model]
    files = sorted(by_model[models[0]])
    for m in models:
        if sorted(by_model[m]) != files:
            raise RuntimeError(f"{m} was evaluated on different files")

    ci = lambda r: CLASSES.index(r)
    y = np.array([ci(by_model[models[0]][f]["true"]) for f in files])
    P = {m: np.array([ci(by_model[m][f]["pred"]) for f in files])
         for m in models}
    photo = [re.sub(r"_g\d+\.png$", "", os.path.basename(f)) for f in files]
    boots = photo_bootstrap(photo, y, args.boot)
    print(f"{args.split}: {len(files)} grains from {len(set(photo))} photos, "
          f"{args.boot} photo-level bootstrap resamples")

    # ---------------- per-model metrics with 95% CI ----------------
    table = []
    for m in models:
        point = metrics(y, P[m])
        bs = [metrics(y[i], P[m][i]) for i in boots]
        row = {"model": NAMES[m], "n": len(y)}
        for k, v in point.items():
            lo, hi = np.percentile([b[k] for b in bs], [2.5, 97.5])
            row[k] = round(v, 4)
            row[f"{k}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        table.append(row)
    with open(f"{args.out}/metrics_{args.split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    print(f"\n{'model':22s} {'accuracy':>22s} {'macro F1':>22s} {'kappa':>7s}")
    for r in table:
        print(f"{r['model']:22s} {r['accuracy']:.4f} {r['accuracy_ci']:>15s} "
              f"{r['macro_f1']:.4f} {r['macro_f1_ci']:>15s} {r['kappa']:7.4f}")

    # ---------------- pairwise McNemar tests ----------------
    pairs = []
    for a, b in itertools.combinations(models, 2):
        ca, cb = P[a] == y, P[b] == y
        n10 = int(np.sum(ca & ~cb))      # A right, B wrong
        n01 = int(np.sum(~ca & cb))      # A wrong, B right
        z = (n10 - n01) / np.sqrt(n10 + n01) if n10 + n01 else 0.0
        p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        d = [np.mean(ca[i]) - np.mean(cb[i]) for i in boots]
        lo, hi = np.percentile(d, [2.5, 97.5])
        pairs.append({"model_A": NAMES[a], "model_B": NAMES[b],
                      "acc_A": round(float(ca.mean()), 4),
                      "acc_B": round(float(cb.mean()), 4),
                      "diff_A_minus_B": round(float(ca.mean() - cb.mean()), 4),
                      "diff_ci": f"[{lo:.3f}, {hi:.3f}]",
                      "A_right_B_wrong": n10, "A_wrong_B_right": n01,
                      "z": round(float(z), 3), "p": p})
    # Holm correction for the 6 comparisons
    order = np.argsort([q["p"] for q in pairs])
    k, running = len(pairs), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * pairs[i]["p"]))
        pairs[i]["p_holm"] = running
    for q in pairs:
        q["significant_0.05"] = "yes" if q["p_holm"] < 0.05 else "no"
        q["p"] = f"{q['p']:.2e}"
        q["p_holm"] = f"{q['p_holm']:.2e}"
    with open(f"{args.out}/pairwise_{args.split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)

    print(f"\n{'A':22s} {'B':22s} {'diff':>7s} {'95% CI':>16s} "
          f"{'b':>4s} {'c':>4s} {'z':>7s} {'p(Holm)':>9s}")
    for q in pairs:
        print(f"{q['model_A']:22s} {q['model_B']:22s} "
              f"{q['diff_A_minus_B']:+7.4f} {q['diff_ci']:>16s} "
              f"{q['A_right_B_wrong']:4d} {q['A_wrong_B_right']:4d} "
              f"{q['z']:7.2f} {q['p_holm']:>9s} {q['significant_0.05']}")

    # ---------------- figures ----------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", BLUE_RAMP)

    # confusion matrices: colour = row percentage (recall), text = count + %
    def draw_cm(ax, m):
        cm = confusion_matrix(y, P[m], labels=range(len(CLASSES)))
        pct = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                dark = pct[i, j] > 55
                ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1f}%", ha="center",
                        va="center", fontsize=9,
                        color="#ffffff" if dark else INK)
        names = [LABELS[c] for c in CLASSES]
        ax.set_xticks(range(4), names)
        ax.set_yticks(range(4), names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        acc = np.mean(P[m] == y) * 100
        ax.set_title(f"{NAMES[m]}  (accuracy {acc:.1f}%)", fontsize=11,
                     color=INK, pad=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    note = ("Cell: number of grains and % of the true class (row). "
            "Shading follows the row percentage.")
    # one image per model ...
    for m in models:
        fig, ax = plt.subplots(figsize=(5.2, 4.9))
        draw_cm(ax, m)
        fig.text(0.5, 0.01, note, ha="center", fontsize=7, color=INK2)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(f"{args.out}/confusion_{args.split}_{NAMES[m]}.png",
                    dpi=300, facecolor="white")
        plt.close(fig)
    # ... and all four together
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6))
    for ax, m in zip(axes.flat, models):
        draw_cm(ax, m)
    fig.text(0.5, 0.005, note, ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(f"{args.out}/confusion_{args.split}.png", dpi=300,
                facecolor="white")
    plt.close(fig)

    # comparison: macro F1 per architecture, SA vs CP, with 95% CI
    archs = [("DenseNet121", "densenet121"),
             ("EfficientNetV2-S", "tf_efficientnetv2_s.in1k")]
    tr = {r["model"]: r for r in table}
    # dot + 95% CI whisker (not bars: the axis does not start at zero)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, (setting, suffix, label) in enumerate(
            [("SA", "", "Color randomly changed (SA)"),
             ("CP", "_color", "Color kept as photographed (CP)")]):
        for i, (aname, key) in enumerate(archs):
            r = tr[NAMES[key + suffix]]
            lo, hi = map(float, r["macro_f1_ci"].strip("[]").split(","))
            x, v = i + (k - 0.5) * 0.36, r["macro_f1"]
            ax.plot([x, x], [lo, hi], color=SERIES[setting], lw=2,
                    solid_capstyle="round", zorder=2)
            ax.plot(x, v, "o", ms=9, color=SERIES[setting],
                    markeredgecolor="white", markeredgewidth=2, zorder=3,
                    label=label if i == 0 else None)
            ax.text(x + 0.05, v, f"{v:.3f}", va="center", ha="left",
                    fontsize=9, color=INK)
    ax.set_xticks(range(len(archs)), [a for a, _ in archs])
    ax.set_xlim(-0.6, len(archs) - 0.4)
    ax.set_ylabel("Macro F1-score (test set)")
    ax.set_ylim(0.75, 0.95)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9,
              bbox_to_anchor=(0.5, 1.12))
    fig.text(0.5, 0.01, "Dot: macro F1-score. Line: 95% confidence interval "
             "(photo-level bootstrap).", ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{args.out}/comparison_{args.split}.png", dpi=300,
                facecolor="white")
    plt.close(fig)
    print(f"\nSaved tables and figures to {args.out}/")


if __name__ == "__main__":
    main()
