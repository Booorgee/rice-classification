# osiris_gui.py
#
# Simple OSIRIS window: "Take Picture" (Pi camera) or "Upload Image" (any
# photo), then shows the annotated result and the class mix of the sample.
#
# Uses the same code as pi_app.py, so results are identical to ./run.sh, and
# reads the model and camera settings from run.sh (the values from --setup).
#
# Usage (on the Pi's desktop):
#   ~/rice-env/bin/python osiris_gui.py
# Start automatically at boot:  ./install_autostart.sh  (see README.md)

import os
import queue
import re
import threading
import time
from types import SimpleNamespace

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------
# Processing (no GUI code here, so it can be tested without a screen)
# ---------------------------------------------------------------
def read_run_sh(path):
    """MODEL / PX_PER_MM / EXPOSURE_US / GAIN / AWB_RED / AWB_BLUE from run.sh,
    overridden by settings.sh next to it (this Pi's calibration), like run.sh."""
    vals = {}
    for p in (path, os.path.join(os.path.dirname(path), "settings.sh")):
        if os.path.exists(p):
            for line in open(p):
                m = re.match(r"\s*([A-Z_]+)=([^\s#]*)", line)
                if m:
                    vals[m.group(1)] = m.group(2)
    return vals


class Inspector:
    def __init__(self, folder=HERE):
        from grain_classifier import OnnxGrainClassifier
        from segment_grains import build_parser, class_limits
        import json

        self.folder = folder
        cfg = read_run_sh(os.path.join(folder, "run.sh"))
        num = lambda k, d: float(cfg[k]) if cfg.get(k) else d
        self.args = SimpleNamespace(
            model=os.path.join(folder, cfg.get("MODEL", "densenet121_color.onnx")),
            px_per_mm=num("PX_PER_MM", None),
            exposure_us=int(num("EXPOSURE_US", 20000)),
            gain=num("GAIN", 1.0),
            awb_gains=[num("AWB_RED", 1.8), num("AWB_BLUE", 1.6)],
            auto=False, low_confidence=0.70, save_crops=False)
        seg = vars(build_parser().parse_args([]))
        params = os.path.join(folder, "seg_params.json")
        if os.path.exists(params):
            seg.update(json.load(open(params)))
        seg["link_mm"] = 0              # keep every grain on the tray
        self.seg = SimpleNamespace(**seg)
        self.lim = class_limits(self.seg, loose=True)
        self.clf = OnnxGrainClassifier(self.args.model, threads=4)
        self.camera = None
        self.results = os.path.join(folder, "results")

    def capture(self):
        from pi_app import Camera
        if self.camera is None:          # open once, keep it for next shots
            self.camera = Camera(self.args)
        return self.camera.capture()

    def run(self, img, name, cam_meta=None):
        """Classify one image; returns (annotated image, summary dict, out dir)."""
        import json
        from pi_app import exposure_advice, inspect
        out_dir = os.path.join(self.results, name)
        vis = inspect(img, self.clf, self.seg, self.lim, self.args, out_dir,
                      cam_meta)
        if vis is None:
            raise RuntimeError("No coin found. Put the coin in view, or set "
                               "PX_PER_MM in run.sh (run --setup once).")
        summary = json.load(open(os.path.join(out_dir, "summary.json")))
        summary["advice"] = exposure_advice(
            summary.get("grain_brightness"), summary.get("grain_clipped_fraction"),
            (cam_meta or {}).get("ExposureTime"))
        return vis, summary, out_dir

    def close(self):
        if self.camera is not None:
            self.camera.close()


def summary_text(s):
    lines = [f"Grains classified: {s['n_grains']}", ""]
    for c, pct in s["percent"].items():
        name = "Broken" if c == "defects" else c.capitalize()
        lines.append(f"{name:<10} {pct:5.1f}%  ({s['counts'][c]})")
    lines += ["",
              f"Low confidence: {s['n_low_conf']}",
              f"Not segmented: {s['n_not_segmented']}",
              f"Time: {s['seconds_total']:.1f} s"]
    if s.get("advice"):
        lines += ["", "WARNING: " + s["advice"]]
    return "\n".join(lines)


# ---------------------------------------------------------------
# Window
# ---------------------------------------------------------------
def main():
    import base64
    import tkinter as tk
    from tkinter import filedialog, messagebox

    BG, PANEL, INK, MUTED, ACCENT = "#1a1a19", "#262624", "#f0efec", "#a3a29c", "#2a78d6"

    root = tk.Tk()
    root.title("OSIRIS — Rice Inspection")
    root.configure(bg=BG)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{sw}x{sh}+0+0")

    # right-hand panel
    side = tk.Frame(root, bg=PANEL, width=340)
    side.pack(side="right", fill="y")
    side.pack_propagate(False)
    tk.Label(side, text="OSIRIS", font=("DejaVu Sans", 22, "bold"),
             fg=INK, bg=PANEL).pack(pady=(24, 2))
    tk.Label(side, text="Rice grain inspection", font=("DejaVu Sans", 11),
             fg=MUTED, bg=PANEL).pack(pady=(0, 20))

    btn_opts = dict(font=("DejaVu Sans", 14, "bold"), fg="white", bg=ACCENT,
                    activebackground="#256abf", activeforeground="white",
                    relief="flat", height=2, cursor="hand2")
    take_btn = tk.Button(side, text="Take Picture", **btn_opts)
    take_btn.pack(fill="x", padx=24, pady=6)
    upload_btn = tk.Button(side, text="Upload Image", **btn_opts)
    upload_btn.pack(fill="x", padx=24, pady=6)

    status = tk.Label(side, text="Loading model...", font=("DejaVu Sans", 11),
                      fg=MUTED, bg=PANEL, wraplength=290, justify="left")
    status.pack(fill="x", padx=24, pady=(18, 6))
    result = tk.Label(side, text="", font=("DejaVu Sans Mono", 12), fg=INK,
                      bg=PANEL, justify="left", anchor="nw", wraplength=290)
    result.pack(fill="both", expand=True, padx=24, pady=6)

    # image area
    view = tk.Label(root, bg=BG, fg=MUTED, font=("DejaVu Sans", 14),
                    text="Take a picture or upload an image")
    view.pack(side="left", fill="both", expand=True)

    state = {"inspector": None, "photo": None, "busy": True}
    jobs = queue.Queue()

    def set_busy(busy, msg=None):
        state["busy"] = busy
        for b in (take_btn, upload_btn):
            b.configure(state="disabled" if busy else "normal")
        if msg is not None:
            status.configure(text=msg)

    def show_image(img):
        w = max(view.winfo_width(), 200)
        h = max(view.winfo_height(), 200)
        k = min(w / img.shape[1], h / img.shape[0])
        small = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)),
                           interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", small)
        state["photo"] = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        view.configure(image=state["photo"], text="")

    def worker(task, arg=None):
        try:
            if task == "load":
                jobs.put(("loaded", Inspector()))
                return
            insp = state["inspector"]
            if task == "camera":
                img, meta = insp.capture()
                name = time.strftime("%Y%m%d_%H%M%S")
            else:
                img, meta = cv2.imread(arg), None
                if img is None:
                    raise RuntimeError(f"Cannot read image: {arg}")
                name = os.path.splitext(os.path.basename(arg))[0]
            jobs.put(("done", insp.run(img, name, meta)))
        except Exception as e:  # shown in the window instead of crashing
            jobs.put(("error", e))

    def start(task, arg=None, msg="Working..."):
        set_busy(True, msg)
        threading.Thread(target=worker, args=(task, arg), daemon=True).start()

    def poll():
        try:
            kind, payload = jobs.get_nowait()
        except queue.Empty:
            root.after(150, poll)
            return
        if kind == "loaded":
            state["inspector"] = payload
            set_busy(False, "Ready. Spread the grains so they don't touch.")
        elif kind == "done":
            vis, summary, out_dir = payload
            show_image(vis)
            result.configure(text=summary_text(summary))
            set_busy(False, f"Saved to {os.path.relpath(out_dir, HERE)}")
        else:
            set_busy(state["inspector"] is None, "Error")
            messagebox.showerror("OSIRIS", str(payload))
        root.after(150, poll)

    def on_take():
        start("camera", msg="Taking picture and classifying...")

    def on_upload():
        path = filedialog.askopenfilename(
            title="Choose a rice photo", initialdir=os.path.expanduser("~"),
            filetypes=[("Images", "*.jpg *.jpeg *.JPG *.JPEG *.png *.PNG"),
                       ("All files", "*")])
        if path:
            start("file", path, msg="Classifying...")

    def on_close():
        if state["inspector"] is not None:
            state["inspector"].close()
        root.destroy()

    take_btn.configure(command=on_take)
    upload_btn.configure(command=on_upload)
    root.protocol("WM_DELETE_WINDOW", on_close)
    set_busy(True)
    start("load", msg="Loading model...")
    root.after(150, poll)
    root.mainloop()


if __name__ == "__main__":
    main()
