import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tf_efficientnetv2_s.in1k",
                        choices=["tf_efficientnetv2_s.in1k", "densenet121"])
    parser.add_argument("--data", default="grains")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--unfreeze_at", type=int, default=5)
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from "
                             "runs/<model>[_color]_last.pt (saved every epoch)")
    parser.add_argument("--color", choices=["off", "on"], default="off",
                        help="on: keep grain colour as a cue (augment only "
                             "brightness/contrast, no hue/saturation jitter) "
                             "and save as <model>_color_*; off: original "
                             "behaviour")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    os.makedirs(args.out, exist_ok=True)
    # file prefix in runs/: colour models get their own name so they never
    # overwrite the original ones
    tag = args.model.replace('/', '_') + ("_color" if args.color == "on" else "")

    print("=" * 70)
    print(f"Device  : {DEVICE}")
    print(f"Model   : {args.model}")
    print(f"Color   : {args.color}  (saves as {args.out}/{tag}_*)")
    print(f"Epochs  : {args.epochs} (unfreeze at epoch {args.unfreeze_at + 1})")
    print(f"Workers : {args.workers}")
    print("=" * 70)

    # Each pretrained model expects the normalization it was trained with
    # (tf_efficientnetv2 uses 0.5/0.5, densenet121 uses ImageNet mean/std)
    cfg = timm.create_model(args.model, pretrained=False).pretrained_cfg
    MEAN, STD = list(cfg["mean"]), list(cfg["std"])
    print(f"Norm    : mean={MEAN} std={STD}")

    train_tf = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        # off: hue/saturation jitter teaches the model to ignore grain
        # colour; on: only vary brightness/contrast so colour stays a cue
        (transforms.ColorJitter(brightness=0.2, contrast=0.2)
         if args.color == "on" else
         transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                saturation=0.15, hue=0.05)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(f"{args.data}/train", train_tf)
    val_ds   = datasets.ImageFolder(f"{args.data}/val",   val_tf)

    if train_ds.classes != val_ds.classes:
        raise RuntimeError(
            f"Class mismatch: train={train_ds.classes} vs val={val_ds.classes}")

    with open(f"{args.out}/{tag}_classes.json", "w") as f:
        json.dump(train_ds.classes, f)
    # everything test_photos.py needs to preprocess crops the same way
    with open(f"{args.out}/{tag}_meta.json", "w") as f:
        json.dump({"model": args.model, "classes": train_ds.classes,
                   "img": args.img, "mean": MEAN, "std": STD,
                   "color": args.color}, f, indent=2)

    test_ds = (datasets.ImageFolder(f"{args.data}/test", val_tf)
               if os.path.isdir(f"{args.data}/test") else None)
    if test_ds is not None and test_ds.classes != train_ds.classes:
        raise RuntimeError(
            f"Class mismatch: train={train_ds.classes} vs test={test_ds.classes}")

    counts = np.bincount(train_ds.targets, minlength=len(train_ds.classes))

    print(f"Classes : {train_ds.classes}")
    print(f"Train   : {len(train_ds)} images")
    print(f"Val     : {len(val_ds)} images")
    print(f"Test    : {len(test_ds) if test_ds else 0} images")
    print("Per class (train): "
          + ", ".join(f"{c}={n}" for c, n in zip(train_ds.classes, counts)))
    print("-" * 70)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers)

    model = timm.create_model(args.model, pretrained=True,
                              num_classes=len(train_ds.classes))
    model = model.to(DEVICE)

    def freeze_backbone(m):
        for p in m.parameters():
            p.requires_grad = False
        for p in m.get_classifier().parameters():
            p.requires_grad = True

    def unfreeze_all(m):
        for p in m.parameters():
            p.requires_grad = True

    freeze_backbone(model)

    # Inverse-frequency class weights (mean 1) so small classes like defects
    # aren't drowned out by the larger varieties
    weights = counts.sum() / (len(counts) * counts)
    print("Class weights: " + ", ".join(
        f"{c}={w:.2f}" for c, w in zip(train_ds.classes, weights)))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE),
        label_smoothing=0.05)

    def make_optimizer(lr):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4,
        )

    optimizer = make_optimizer(args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.unfreeze_at)



    def train_one_epoch(epoch):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"E{epoch:02d} train",
                    leave=False, ncols=90)
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            loss_sum += loss.item() * bs
            correct  += (out.argmax(1) == y).sum().item()
            n        += bs
            pbar.set_postfix(loss=f"{loss_sum/n:.3f}",
                             acc=f"{correct/n:.3f}")
        return loss_sum / n, correct / n

    @torch.no_grad()
    def validate(epoch):
        model.eval()
        loss_sum, correct, n = 0.0, 0, 0
        ys, ps = [], []
        pbar = tqdm(val_loader, desc=f"E{epoch:02d} val  ",
                    leave=False, ncols=90)
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            bs = x.size(0)
            loss_sum += loss.item() * bs
            correct  += (out.argmax(1) == y).sum().item()
            n        += bs
            ys.extend(y.tolist())
            ps.extend(out.argmax(1).tolist())
            pbar.set_postfix(loss=f"{loss_sum/n:.3f}",
                             acc=f"{correct/n:.3f}")
        return loss_sum / n, correct / n, f1_score(ys, ps, average="macro")

    # Best checkpoint = highest val macro-F1 (every class counts equally),
    # not plain accuracy, which the largest classes dominate
    best_f1 = -1.0
    best_path = f"{args.out}/{tag}_best.pt"
    # full training state, saved after every epoch so --resume can pick up
    # after an interruption (sleep, crash, closed session)
    last_path = f"{args.out}/{tag}_last.pt"
    start_ep = 1
    if args.resume:
        if not os.path.exists(last_path):
            raise FileNotFoundError(f"--resume: {last_path} not found")
        ck = torch.load(last_path, map_location=DEVICE)
        done = ck["epoch"]
        if done > args.unfreeze_at:
            # checkpoint is from the fine-tuning phase
            unfreeze_all(model)
            optimizer = make_optimizer(args.lr / 10)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.unfreeze_at)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        best_f1 = ck["best_f1"]
        start_ep = done + 1
        print(f">>> Resuming after epoch {done} (best val F1 so far "
              f"{best_f1:.4f})")
    t0 = time.time()

    print(f"Starting training on {DEVICE} ...")
    print("=" * 70)

    for ep in range(start_ep, args.epochs + 1):
        if ep == args.unfreeze_at + 1:
            print("\n>>> Unfreezing backbone — fine-tuning with lr/10")
            unfreeze_all(model)
            optimizer = make_optimizer(args.lr / 10)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.unfreeze_at)
            print(">>> Trainable params:",
                  sum(p.numel() for p in model.parameters() if p.requires_grad))
            print("-" * 70)

        ep_t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(ep)
        vl_loss, vl_acc, vl_f1 = validate(ep)
        scheduler.step()
        ep_dt = time.time() - ep_t0

        is_best = vl_f1 > best_f1
        if is_best:
            best_f1 = vl_f1
            torch.save(model.state_dict(), best_path)
        torch.save({"epoch": ep, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_f1": best_f1}, last_path)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {ep:02d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {vl_loss:.4f} acc {vl_acc:.4f} f1 {vl_f1:.4f} | "
            f"lr {lr_now:.1e} | {ep_dt:.1f}s"
            + ("  *best*" if is_best else "")
        )

    total = time.time() - t0
    print("=" * 70)
    print(f"Training complete in {total/60:.1f} min")
    print(f"Best val F1  : {best_f1:.4f} (macro)")
    print(f"Saved        : {best_path}")
    print(f"Classes      : {train_ds.classes}")


    print("\nFinal evaluation on best checkpoint ...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    splits = [("VAL", val_ds)]
    if test_ds is not None:
        # test was never used for training or checkpoint selection: this is
        # the number to report
        splits.append(("TEST", test_ds))
    for name, ds in splits:
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers)
        ys, ps = [], []
        with torch.no_grad():
            for x, y in loader:
                ps.extend(model(x.to(DEVICE)).argmax(1).cpu().tolist())
                ys.extend(y.tolist())

        labels = list(range(len(train_ds.classes)))
        print(f"\n===== {name} ({len(ds)} images) =====")
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(ys, ps, labels=labels))
        print("\nClassification report:")
        print(classification_report(ys, ps, labels=labels,
                                    target_names=train_ds.classes,
                                    digits=4, zero_division=0))


if __name__ == "__main__":
    main()