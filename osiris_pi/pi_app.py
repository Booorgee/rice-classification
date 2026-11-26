# pi_app.py
#
# Raspberry Pi rice inspection: shoot -> segment -> classify -> output image.
#
# Takes a full-resolution still with the Pi camera (v2, 3280x2464), segments
# the grains with the same code used to build the training crops, classifies
# every grain with an exported ONNX model (export_onnx.py), and saves:
#   results/<timestamp>/annotated.jpg   every grain boxed + label + confidence,
#                                       with a summary panel (class mix in %)
#   results/<timestamp>/grains.csv      one row per grain
#   results/<timestamp>/summary.json
#   results/<timestamp>/raw.jpg         the original photo
#
# Files needed on the Pi (same folder): pi_app.py, segment_grains.py,
# grain_classifier.py, the .onnx model, and grains/seg_params.json (optional,
# copied as seg_params.json).
#
# Usage:
#   python pi_app.py --model densenet121_int8.onnx                 # one shot
#   python pi_app.py --model densenet121_int8.onnx --loop --show   # Enter = shoot
#   python pi_app.py --model densenet121_int8.onnx --image photo.jpg  # no camera
#
# First-time rig setup (see PI_SETUP.md):
#   python pi_app.py --setup        # auto exposure/WB once, prints the values
#                                   # to lock and the px/mm from the coin

import argparse
import csv
import json
import os
import time
from types import SimpleNamespace

import cv2
import numpy as np

from grain_classifier import OnnxGrainClassifier
from segment_grains import build_parser, class_limits, detect_coin, \
    foreground_mask, process_image

# BGR colours by class name: defects in red so they stand out
CLASS_COLORS = {"defects": (40, 40, 255), "dinorado": (0, 170, 255),
                "jasmine": (255, 170, 60), "malagkit": (80, 220, 80)}
FALLBACK = [(200, 80, 200), (200, 200, 60), (180, 180, 180)]


def color_of(classes, k):
    return CLASS_COLORS.get(classes[k], FALLBACK[k % len(FALLBACK)])


STILL_SIZE = (3280, 2464)  # Camera Module v2 full resolution
# Grain brightness (0-255 grey level of grain pixels) in the training photos:
# 97-171 per photo, with essentially no pure-white pixels (<= 0.01%).
# Over-exposed, clipped grains look nothing like the training data (they come
# out as "broken", the brightest class), so the check below flags them.
BRIGHT_RANGE = (90, 175)
TARGET_BRIGHTNESS = 125


def grain_brightness(res):
    """Mean grey level of the grain pixels and the share that is clipped."""
    vals = [cv2.cvtColor(g["crop"], cv2.COLOR_BGR2GRAY)[g["crop"].max(2) > 15]
            for g in res["grains"]]
    vals = np.concatenate(vals) if vals else np.array([])
    if not len(vals):
        return None, None
    return float(vals.mean()), float(np.mean(vals >= 250))


def exposure_advice(mean, clipped, exposure_us):
    """One line of advice, or None when the grains look like the training data."""
    if mean is None:
        return None
    if clipped > 0.01:
        # clipped: the true brightness is unknown, so step down by half
        fix = (f" — try --exposure-us {int(exposure_us * 0.5)}"
               if exposure_us else " — lower the exposure")
        return (f"grains over-exposed: {clipped:.0%} of grain pixels are pure "
                f"white (training photos: ~0%){fix}")
    lo, hi = BRIGHT_RANGE
    if not lo <= mean <= hi:
        fix = (f" — try --exposure-us {int(exposure_us * TARGET_BRIGHTNESS / mean)}"
               if exposure_us else "")
        return (f"grain brightness {mean:.0f} is outside the training range "
                f"{lo}-{hi}{fix}")
    return None


# ---------------------------------------------------------------
# Camera
# ---------------------------------------------------------------
class Camera:
    def __init__(self, args):
        from picamera2 import Picamera2  # only needed on the Pi
        self.cam = Picamera2()
        # "RGB888" in picamera2 is BGR byte order: what OpenCV expects
        cfg = self.cam.create_still_configuration(
            main={"size": STILL_SIZE, "format": "RGB888"})
        self.cam.configure(cfg)
        controls = {}
        if not args.auto:
            # Locked settings: every photo gets the same brightness and
            # colour, which the classifier (especially --color on) relies on
            controls.update(AeEnable=False, AwbEnable=False,
                            ExposureTime=args.exposure_us,
                            AnalogueGain=args.gain,
                            ColourGains=tuple(args.awb_gains))
        self.cam.start()
        if controls:
            self.cam.set_controls(controls)
        time.sleep(2)  # let exposure / white balance settle

    def capture(self):
        req = self.cam.capture_request()
        try:
            img = req.make_array("main").copy()
            meta = req.get_metadata()
        finally:
            req.release()
        return img, meta

    def close(self):
        self.cam.stop()
        self.cam.close()


# ---------------------------------------------------------------
# Output image
# ---------------------------------------------------------------
def annotate(img, res, preds, confs, classes, low_conf, summary):
    vis = img.copy()
    s = img.shape[1] / STILL_SIZE[0]          # scale drawing to image size
    th, fs = max(2, int(3 * s)), 0.9 * s
    lth, lfs = max(1, int(2 * s)), 0.6 * s    # grain labels: small
    for c, _ in res["rejects"]:
        cv2.drawContours(vis, [c], -1, (120, 120, 120), max(1, int(2 * s)))
    boxes = [cv2.boundingRect(g["contour"]) for g in res["grains"]]
    for (x, y, w, h), k in zip(boxes, preds):
        cv2.rectangle(vis, (x, y), (x + w, y + h), color_of(classes, k), lth)
    # labels after all boxes; black outline instead of a filled background so
    # they never hide neighbouring grains
    for (x, y, w, h), k, conf in zip(boxes, preds, confs):
        txt = f"{classes[k][:3].upper()} {conf * 100:.0f}%"
        if conf < low_conf:
            txt += "?"
        org = (x, max(int(20 * s), y - int(6 * s)))
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, lfs, (0, 0, 0),
                    lth + max(2, int(3 * s)), cv2.LINE_AA)
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, lfs,
                    color_of(classes, k), lth, cv2.LINE_AA)

    # summary panel on the right
    pw = int(900 * s)
    panel = np.full((vis.shape[0], pw, 3), 25, np.uint8)
    lines = [("RICE INSPECTION", (255, 255, 255), 1.6),
             (time.strftime("%Y-%m-%d %H:%M:%S"), (180, 180, 180), 1.0),
             ("", None, 1),
             (f"Grains classified: {summary['n_grains']}", (255, 255, 255), 1.2),
             ("", None, 1)]
    for j, c in enumerate(classes):
        n = summary["counts"][c]
        lines.append((f"{c:<9s} {summary['percent'][c]:5.1f}%  ({n})",
                      color_of(classes, j), 1.4))
    lines += [("", None, 1),
              (f"Low confidence (<{low_conf:.0%}): {summary['n_low_conf']}",
               (200, 200, 200), 1.0),
              (f"Not segmented: {summary['n_not_segmented']}",
               (200, 200, 200), 1.0),
              (f"Scale: {summary['px_per_mm']:.1f} px/mm", (200, 200, 200), 1.0),
              (f"Model: {summary['model']}", (200, 200, 200), 0.9),
              (f"Time: {summary['seconds_total']:.1f} s", (200, 200, 200), 1.0)]
    y = int(80 * s)
    for txt, col, k in lines:
        if txt:
            cv2.putText(panel, txt, (int(30 * s), y), cv2.FONT_HERSHEY_SIMPLEX,
                        fs * k, col, th, cv2.LINE_AA)
        y += int(70 * s * max(k, 1))
    return np.hstack([vis, panel])


# ---------------------------------------------------------------
# One inspection
# ---------------------------------------------------------------
def inspect(img, clf, seg, lim, args, out_dir, cam_meta=None):
    t0 = time.time()
    res, err = process_image(img, seg, lim, px_per_mm=args.px_per_mm)
    if err:
        print(f"  !! {err}: put the coin in view, or pass --px-per-mm "
              f"(run --setup once to measure it)")
        return None
    t1 = time.time()
    probs = clf.predict([g["crop"] for g in res["grains"]])
    t2 = time.time()
    preds = probs.argmax(1) if len(probs) else np.array([], int)
    confs = probs.max(1) if len(probs) else np.array([])

    n = len(preds)
    counts = {c: int(np.sum(preds == j)) for j, c in enumerate(clf.classes)}
    summary = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": os.path.basename(args.model),
        "n_grains": n,
        "counts": counts,
        "percent": {c: (100.0 * v / n if n else 0.0) for c, v in counts.items()},
        "n_low_conf": int(np.sum(confs < args.low_confidence)),
        "n_not_segmented": len(res["rejects"]),
        "px_per_mm": float(res["px_per_mm"]),
        "seconds_segment": round(t1 - t0, 2),
        "seconds_classify": round(t2 - t1, 2),
        "seconds_total": round(t2 - t0, 2),
    }
    bright, clipped = grain_brightness(res)
    summary["grain_brightness"] = round(bright, 1) if bright is not None else None
    summary["grain_clipped_fraction"] = (round(clipped, 4)
                                         if clipped is not None else None)
    if cam_meta:
        summary["camera"] = {k: cam_meta.get(k) for k in
                             ("ExposureTime", "AnalogueGain", "ColourGains",
                              "Lux", "ColourTemperature")}

    os.makedirs(out_dir, exist_ok=True)
    if args.save_crops:
        # exactly what the model sees, named by prediction, for checking
        # scale/focus/colour against the training crops in grains/
        os.makedirs(f"{out_dir}/crops", exist_ok=True)
        for i, (g, k, conf) in enumerate(zip(res["grains"], preds, confs)):
            cv2.imwrite(f"{out_dir}/crops/{i:03d}_{clf.classes[k]}_"
                        f"{conf * 100:.0f}.png", g["crop"])
    cv2.imwrite(f"{out_dir}/raw.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    vis = annotate(img, res, preds, confs, clf.classes, args.low_confidence,
                   summary)
    cv2.imwrite(f"{out_dir}/annotated.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(f"{out_dir}/grains.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grain", "pred", "confidence",
                    *[f"p_{c}" for c in clf.classes],
                    "length_mm", "width_mm", "area_mm2", "x", "y", "w", "h"])
        for i, (g, k, conf) in enumerate(zip(res["grains"], preds, confs)):
            x, y, bw, bh = cv2.boundingRect(g["contour"])
            w.writerow([i, clf.classes[k], f"{conf:.4f}",
                        *[f"{p:.4f}" for p in probs[i]],
                        f"{g['length_mm']:.3f}", f"{g['width_mm']:.3f}",
                        f"{g['area_mm2']:.3f}", x, y, bw, bh])
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    mix = ", ".join(f"{c} {summary['percent'][c]:.1f}%"
                    for c in clf.classes if counts[c])
    print(f"  {n} grains -> {mix or '-'}")
    print(f"  low confidence: {summary['n_low_conf']}, not segmented: "
          f"{summary['n_not_segmented']}")
    print(f"  time: segment {summary['seconds_segment']}s + classify "
          f"{summary['seconds_classify']}s "
          f"({1000 * (t2 - t1) / max(n, 1):.0f} ms/grain)")
    advice = exposure_advice(bright, clipped,
                             (cam_meta or {}).get("ExposureTime"))
    if bright is not None:
        print(f"  grain brightness: {bright:.0f}, {clipped:.1%} pure white "
              f"(training: {BRIGHT_RANGE[0]}-{BRIGHT_RANGE[1]}, ~0% pure white)")
    if advice:
        print(f"  !! {advice}. Results are unreliable until this is fixed.")
    print(f"  saved: {out_dir}/annotated.jpg")
    return vis


def show(vis, args):
    if not args.show:
        return
    h, w = vis.shape[:2]
    k = min(1.0, args.screen_w / w, args.screen_h / h)
    cv2.imshow("rice", cv2.resize(vis, (int(w * k), int(h * k)),
                                  interpolation=cv2.INTER_AREA))
    print("  (press any key in the image window to continue)")
    cv2.waitKey(0)


# ---------------------------------------------------------------
# First-time rig setup
# ---------------------------------------------------------------
def setup(args, seg):
    args.auto = True
    cam = Camera(args)
    img, meta = cam.capture()
    cam.close()
    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/setup.jpg"
    cv2.imwrite(path, img)
    exp, gain = meta.get("ExposureTime"), meta.get("AnalogueGain")
    rg, bg = meta.get("ColourGains", (None, None))
    print(f"Saved {path}  ({img.shape[1]}x{img.shape[0]})")
    print("\nAuto settings the camera chose — lock them with:")
    print(f"  --exposure-us {exp} --gain {gain:.2f} --awb-gains {rg:.3f} {bg:.3f}")

    work = 0.5
    small = cv2.resize(img, None, fx=work, fy=work, interpolation=cv2.INTER_AREA)
    mask = foreground_mask(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), work)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    coin = detect_coin(cnts)
    if coin is None:
        print("\nNo coin found: place the coin flat in view and run --setup again")
        return
    ppm = 2 * coin[2] / work / seg.coin_mm
    print(f"\nCoin found: {ppm:.2f} px/mm  (field of view "
          f"{img.shape[1] / ppm:.0f} x {img.shape[0] / ppm:.0f} mm)")
    print(f"  fixed rig -> add  --px-per-mm {ppm:.2f}  (then the coin is optional)")
    if ppm < 12:
        print("  !! below ~12 px/mm the grains have less detail than the "
              "training photos (~13 px/mm): lower the camera / refocus closer")
    elif ppm > 20:
        print("  note: the training photos were ~13 px/mm; raising the camera "
              "to ~13-16 px/mm makes images closer to the training data")

    # exposure check on the grains themselves
    res, err = process_image(img, seg, class_limits(seg, loose=True),
                             px_per_mm=ppm)
    if err or not res["grains"]:
        print("\nNo grains found: put some spread-out rice in view to check "
              "the exposure")
        return
    bright, clipped = grain_brightness(res)
    print(f"\nGrain brightness: {bright:.0f}, {clipped:.1%} pure white  "
          f"(training: {BRIGHT_RANGE[0]}-{BRIGHT_RANGE[1]}, ~0% pure white)")
    advice = exposure_advice(bright, clipped, exp)
    if advice:
        print(f"  !! {advice}")
    else:
        print("  OK: grain brightness matches the training images")
    if gain and gain > 2.5:
        print(f"  !! gain {gain:.1f} is high (dim light -> noisy image): turn "
              f"on / move the ring light closer, then run --setup again")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help=".onnx file from export_onnx.py")
    ap.add_argument("--image", nargs="+",
                    help="classify these photos instead of using the camera")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seg-params", default="seg_params.json",
                    help="segment_grains settings used for the training crops")
    ap.add_argument("--px-per-mm", type=float, default=None,
                    help="fixed rig scale (from --setup); without it the coin "
                         "must be in every photo")
    ap.add_argument("--link-mm", type=float, default=0,
                    help="drop grains further than this from the main spread "
                         "(0 = keep all grains on the tray)")
    ap.add_argument("--strict", action="store_true",
                    help="whole-grain size limits; default keeps broken grains "
                         "so defects are detected")
    ap.add_argument("--low-confidence", type=float, default=0.70)
    ap.add_argument("--save-crops", action="store_true",
                    help="also save every grain image the model classified "
                         "(results/<time>/crops/) for troubleshooting")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--loop", action="store_true",
                    help="keep the camera open: Enter = new photo, q = quit")
    ap.add_argument("--show", action="store_true",
                    help="display the result (needs a desktop session)")
    ap.add_argument("--screen-w", type=int, default=1600)
    ap.add_argument("--screen-h", type=int, default=900)
    # camera
    ap.add_argument("--setup", action="store_true",
                    help="one auto-exposed shot: prints settings to lock and "
                         "px/mm from the coin")
    ap.add_argument("--auto", action="store_true",
                    help="auto exposure / white balance (not recommended: "
                         "colour changes between photos)")
    ap.add_argument("--exposure-us", type=int, default=20000)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--awb-gains", type=float, nargs=2, default=[1.8, 1.6],
                    metavar=("RED", "BLUE"))
    args = ap.parse_args()

    # segmentation settings: same as the training crops
    seg = vars(build_parser().parse_args([]))
    if os.path.exists(args.seg_params):
        with open(args.seg_params) as f:
            seg.update(json.load(f))
    else:
        print(f"[info] {args.seg_params} not found, using segment_grains "
              f"defaults")
    seg["link_mm"] = args.link_mm
    seg = SimpleNamespace(**seg)

    if args.setup:
        setup(args, seg)
        return
    if not args.model:
        ap.error("--model is required (unless --setup)")

    lim = class_limits(seg, loose=not args.strict)
    clf = OnnxGrainClassifier(args.model, threads=args.threads)
    print(f"Model: {args.model}  classes={clf.classes}  color={clf.color}")

    stamp = lambda: time.strftime("%Y%m%d_%H%M%S")
    if args.image:
        for p in args.image:
            print(f"\n{p}")
            img = cv2.imread(p)
            if img is None:
                print("  !! unreadable")
                continue
            name = os.path.splitext(os.path.basename(p))[0]
            vis = inspect(img, clf, seg, lim, args, f"{args.out}/{name}")
            if vis is not None:
                show(vis, args)
        return

    cam = Camera(args)
    try:
        while True:
            print(f"\n[{stamp()}] capturing ...")
            img, meta = cam.capture()
            vis = inspect(img, clf, seg, lim, args, f"{args.out}/{stamp()}",
                          meta)
            if vis is not None:
                show(vis, args)
            if not args.loop:
                break
            if input("\nEnter = next photo, q = quit: ").strip().lower() == "q":
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
