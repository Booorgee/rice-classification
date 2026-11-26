# export_onnx.py
#
# Converts a trained model (runs/<model>[_color]_best.pt) to ONNX for the
# Raspberry Pi, optionally also an int8-quantized version (smaller, ~2-3x
# faster on the Pi's CPU), then checks both against the PyTorch model on the
# grains/test crops so you can see whether the conversion changed anything.
#
# Run this on your laptop / training machine, then copy the .onnx file to the
# Pi. Class names, input size and normalization are stored inside the .onnx.
#
# Usage:
#   .venv/bin/python export_onnx.py --model densenet121
#   .venv/bin/python export_onnx.py --model densenet121 --color on --int8
#
# Writes:
#   runs/<model>[_color].onnx          float32
#   runs/<model>[_color]_int8.onnx     with --int8

import argparse
import glob
import json
import os
import random
import time

import numpy as np
import onnx
import timm
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torchvision import datasets, transforms

from grain_classifier import OnnxGrainClassifier


def add_metadata(path, meta):
    m = onnx.load(path)
    del m.metadata_props[:]
    p = m.metadata_props.add()
    p.key, p.value = "rice_meta", json.dumps(meta)
    onnx.save(m, path)


def export(model, img, path):
    x = torch.randn(1, 3, img, img)
    try:
        torch.onnx.export(
            model, (x,), path, input_names=["input"], output_names=["logits"],
            dynamic_shapes={"x": {0: torch.export.Dim("batch", max=256)}},
            opset_version=17, dynamo=True)
    except Exception as e:  # older torch / unsupported op: legacy exporter
        print(f"[info] dynamo export failed ({type(e).__name__}); "
              f"using legacy exporter")
        torch.onnx.export(
            model, (x,), path, input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17, dynamo=False)
    # dynamo may write weights to a side file: fold everything into one file
    m = onnx.load(path)
    onnx.save(m, path, save_as_external_data=False)
    for f in glob.glob(path + ".data"):
        os.remove(f)


def quantize_int8(fp32_path, int8_path, calib_crops, clf):
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.batches = iter([
                {clf.input: clf.preprocess(calib_crops[i:i + 8])}
                for i in range(0, len(calib_crops), 8)])

        def get_next(self):
            return next(self.batches, None)

    prep = fp32_path.replace(".onnx", "_prep.onnx")
    quant_pre_process(fp32_path, prep)
    quantize_static(prep, int8_path, Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8)
    os.remove(prep)


def load_crops(folder, per_class=None, seed=0):
    """BGR crops + labels from an ImageFolder-style folder."""
    import cv2
    ds = datasets.ImageFolder(folder)
    items = ds.samples
    if per_class:
        rng = random.Random(seed)
        by = {}
        for p, y in items:
            by.setdefault(y, []).append((p, y))
        items = [s for v in by.values()
                 for s in rng.sample(v, min(per_class, len(v)))]
    return [cv2.imread(p) for p, _ in items], np.array([y for _, y in items])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="tf_efficientnetv2_s.in1k or densenet121")
    ap.add_argument("--color", choices=["off", "on"], default="off")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--grains", default="grains",
                    help="crops used for int8 calibration (train) and the "
                         "accuracy check (test)")
    ap.add_argument("--int8", action="store_true",
                    help="also write an int8-quantized model")
    ap.add_argument("--calib-per-class", type=int, default=50)
    args = ap.parse_args()

    tag = args.model.replace("/", "_") + ("_color" if args.color == "on" else "")
    with open(f"{args.runs}/{tag}_meta.json") as f:
        meta = json.load(f)
    meta.setdefault("color", args.color)

    model = timm.create_model(meta["model"], pretrained=False,
                              num_classes=len(meta["classes"]))
    model.load_state_dict(torch.load(f"{args.runs}/{tag}_best.pt",
                                     map_location="cpu"))
    model.eval()

    fp32_path = f"{args.runs}/{tag}.onnx"
    print(f"Exporting {tag} -> {fp32_path}")
    export(model, meta["img"], fp32_path)
    add_metadata(fp32_path, meta)
    outputs = [("onnx fp32", fp32_path)]

    if args.int8:
        int8_path = f"{args.runs}/{tag}_int8.onnx"
        print(f"Quantizing -> {int8_path} "
              f"(calibrating on {args.calib_per_class} train crops per class)")
        calib, _ = load_crops(f"{args.grains}/train", args.calib_per_class)
        quantize_int8(fp32_path, int8_path, calib,
                      OnnxGrainClassifier(fp32_path))
        add_metadata(int8_path, meta)
        outputs.append(("onnx int8", int8_path))

    # ---- check against PyTorch on the test crops ----
    test_dir = f"{args.grains}/test"
    if not os.path.isdir(test_dir):
        print(f"[skip check] {test_dir} not found")
        return
    crops, ys = load_crops(test_dir)
    tf = transforms.Compose([
        transforms.Resize((meta["img"], meta["img"])),
        transforms.ToTensor(),
        transforms.Normalize(meta["mean"], meta["std"]),
    ])
    import cv2
    with torch.no_grad():
        ref = np.concatenate([
            model(torch.stack([
                tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                for c in crops[i:i + 32]])).argmax(1).numpy()
            for i in range(0, len(crops), 32)])

    print(f"\nCheck on {test_dir} ({len(crops)} crops):")
    print(f"  {'':10s} {'acc':>7s} {'macroF1':>8s} {'same as torch':>14s} "
          f"{'size MB':>8s} {'ms/crop*':>9s}")
    print(f"  {'pytorch':10s} {np.mean(ref == ys):7.4f} "
          f"{f1_score(ys, ref, average='macro'):8.4f} {'-':>14s}")
    keep = []
    for name, path in outputs:
        clf = OnnxGrainClassifier(path)
        t = time.time()
        pred = clf.predict(crops).argmax(1)
        ms = (time.time() - t) / len(crops) * 1000
        agree = np.mean(pred == ref)
        print(f"  {name:10s} {np.mean(pred == ys):7.4f} "
              f"{f1_score(ys, pred, average='macro'):8.4f} "
              f"{agree:13.2%} "
              f"{os.path.getsize(path) / 1e6:8.1f} {ms:9.1f}")
        if agree < 0.98:
            # int8 can break some architectures (DenseNet121 in testing
            # predicted nearly everything as one class): never ship that
            os.remove(path)
            print(f"  !! {name} disagrees with PyTorch on {1 - agree:.0%} of "
                  f"crops — deleted {path}, use the fp32 model")
        else:
            keep.append(path)
    print("  * speed on THIS machine; the Pi 4 will be several times slower")
    print(f"\nCopy to the Pi: {', '.join(keep)}")


if __name__ == "__main__":
    main()
