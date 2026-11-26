# grain_classifier.py
#
# Runs an exported ONNX grain classifier (see export_onnx.py) with
# onnxruntime. Used by pi_app.py on the Raspberry Pi and by export_onnx.py to
# check the export against PyTorch. Needs only numpy + opencv + onnxruntime.
#
# The class names, input size and normalization are stored inside the .onnx
# file, so the model file is all the Pi needs.

import json
import os

import cv2
import numpy as np
import onnxruntime as ort


class OnnxGrainClassifier:
    def __init__(self, path, threads=None):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or os.cpu_count() or 4
        self.sess = ort.InferenceSession(path, so,
                                         providers=["CPUExecutionProvider"])
        meta = json.loads(
            self.sess.get_modelmeta().custom_metadata_map["rice_meta"])
        self.classes = meta["classes"]
        self.img = meta["img"]
        self.color = meta.get("color", "off")
        self.model_name = meta["model"]
        self.mean = np.array(meta["mean"], np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(meta["std"], np.float32).reshape(1, 3, 1, 1)
        self.input = self.sess.get_inputs()[0].name

    def preprocess(self, crops_bgr):
        """Same as train.py's val transform: resize, RGB, /255, normalize."""
        x = np.stack([
            cv2.cvtColor(
                c if c.shape[:2] == (self.img, self.img)
                else cv2.resize(c, (self.img, self.img),
                                interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB)
            for c in crops_bgr
        ]).astype(np.float32) / 255.0
        x = x.transpose(0, 3, 1, 2)
        return (x - self.mean) / self.std

    def predict(self, crops_bgr, batch=16):
        """Returns softmax probabilities, shape (n_crops, n_classes)."""
        out = []
        for i in range(0, len(crops_bgr), batch):
            logits = self.sess.run(
                None, {self.input: self.preprocess(crops_bgr[i:i + batch])})[0]
            e = np.exp(logits - logits.max(1, keepdims=True))
            out.append(e / e.sum(1, keepdims=True))
        return (np.concatenate(out) if out
                else np.zeros((0, len(self.classes)), np.float32))
