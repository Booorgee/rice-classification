#!/bin/bash
# OSIRIS launcher:  ./run.sh
# Defaults are below. Put THIS Pi's values from `pi_app.py --setup` in
# settings.sh (same names, e.g. EXPOSURE_US=104000) — see README Step 9.
# settings.sh is not in Git, so `git pull` never overwrites your calibration.
cd "$(dirname "$0")"

MODEL=densenet121_color.onnx
PX_PER_MM=          # empty = measure from the coin in every photo
EXPOSURE_US=20000
GAIN=1.0
AWB_RED=1.8
AWB_BLUE=1.6
[ -f settings.sh ] && . ./settings.sh

ARGS=(--model "$MODEL" --exposure-us "$EXPOSURE_US" --gain "$GAIN"
      --awb-gains "$AWB_RED" "$AWB_BLUE" --loop)
[ -n "$PX_PER_MM" ] && ARGS+=(--px-per-mm "$PX_PER_MM")
[ -n "$DISPLAY" ] && ARGS+=(--show)     # show the result window on the desktop

exec ~/rice-env/bin/python pi_app.py "${ARGS[@]}" "$@"
