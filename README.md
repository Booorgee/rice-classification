# OSIRIS — Rice grain classification

Grain-level classification of Philippine rice (Dinorado, Jasmine, Malagkit)
and broken grains, with a Raspberry Pi 4B + Camera Module v2 device (OSIRIS).

## Folders

| Path | Contents |
|---|---|
| `osiris_pi/` | **Everything the Raspberry Pi needs** — app, GUI, setup guide (`osiris_pi/README.md`). The model (`densenet121_color.onnx`, 29 MB) is attached to the **v1.0 Release** |
| `figures/`, `tables/`, `eval/` | Paper figures, tables and evaluation results |
| `grains/grains.csv`, `grains/seg_params.json` | Per-grain measurements and segmentation settings |
| `runs/*_meta.json` | Class names / preprocessing of each trained model |

Not in Git (too large — kept on Google Drive): `DATASET/` (photos),
`grains/train|val|test/` (crops), `runs/*.pt` (trained PyTorch models).

## Pipeline (laptop)

| Script | Purpose |
|---|---|
| `segment_grains.py` | Segment photos in `DATASET/<CLASS>/` into grain crops, split 70/15/15 by photo |
| `train.py` | Train DenseNet121 / EfficientNetV2-S (`--color on` = color-preserving) |
| `evaluate.py`, `analyze.py` | Metrics, confidence intervals, McNemar tests, confusion matrices |
| `export_onnx.py` | Convert a trained model to ONNX for the Pi |
| `test_photos.py` | Test a model on a new batch of whole photos |
| `make_figure.py` | Segmentation pipeline figure |

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Raspberry Pi

See [`osiris_pi/README.md`](osiris_pi/README.md).
