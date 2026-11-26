# OSIRIS — Raspberry Pi setup, step by step

From a **fresh** Raspberry Pi 4B to classifying rice with the best model
(DenseNet121-CP). Follow the steps in order. Commands in grey boxes are typed
into a **terminal** (on the Pi unless the step says *laptop*).

Everything the Pi needs is in the `osiris_pi/` folder of the project:

| File | What it is |
|---|---|
| `pi_app.py` | the OSIRIS program: shoot → segment → classify → result image |
| `segment_grains.py` | grain segmentation (same code that built the training data) |
| `grain_classifier.py` | runs the model |
| `densenet121_color.onnx` | the best model (DenseNet121-CP) |
| `seg_params.json` | segmentation settings used for the training crops |
| `sample.jpg` | a test photo, to check everything works before the camera |
| `run.sh` | one-command launcher (terminal version) |
| `osiris_gui.py` | the OSIRIS window: Take Picture / Upload Image (Step 11) |
| `install_autostart.sh` | opens the OSIRIS window automatically at boot (Step 11) |
| `settings.sh` | **created by you in Step 9**: this Pi's camera calibration (not in Git) |

---

## What you need

- Raspberry Pi 4B (4 GB), official **5.1 V 3 A USB-C** power supply
- microSD card, **16 GB or larger** (32 GB recommended), and a card reader
- Raspberry Pi **Camera Module v2** with its ribbon cable
- Your stand/rig, ring light, black background, and the **1-peso coin (23 mm)**
- For the first setup: monitor (micro-HDMI cable), USB keyboard and mouse —
  or just a laptop on the same Wi-Fi (SSH)
- Internet on the Pi (Wi-Fi or phone hotspot), to download OSIRIS from GitHub

---

## Step 1 — Install Raspberry Pi OS on the microSD card (*laptop*)

1. Download and install **Raspberry Pi Imager** from
   https://www.raspberrypi.com/software/
2. Insert the microSD card into your laptop and open Raspberry Pi Imager.
3. **Device:** Raspberry Pi 4.
   **OS:** *Raspberry Pi OS (64-bit)* — the version **with desktop**.
   **Storage:** your microSD card.
4. Click **Next** → **Edit settings** (OS customisation) and set:
   - **Hostname:** `osiris`
   - **Username and password:** choose your own (remember them; this guide
     writes the username as `pi` — replace it with yours)
   - **Wi-Fi:** your network name and password, and your country
   - **Services tab:** tick **Enable SSH** (use password authentication)
5. **Save** → **Yes** → wait until it finishes writing and verifying.

## Step 2 — Connect the camera (Pi switched OFF)

1. Make sure the Pi is **unplugged**.
2. Find the **CAMERA** connector (between the HDMI ports and the audio jack —
   *not* the DISPLAY connector near the SD card).
3. Gently pull up the connector's plastic latch.
4. Insert the ribbon cable with the **silver contacts facing the HDMI ports**
   (the blue strip faces the USB/Ethernet ports). Push it in straight.
5. Press the latch back down. The cable should not come out when lightly pulled.

## Step 3 — First boot and update

1. Insert the microSD card into the Pi, connect monitor/keyboard/mouse (if
   using), then plug in the power.
2. First boot takes a few minutes. You'll reach the desktop (or, headless,
   wait ~3 minutes).
3. Open a terminal:
   - **On the Pi's desktop:** click the terminal icon (top bar).
   - **From your laptop (SSH):** `ssh pi@osiris.local`
     (type `yes` the first time, then your password).
4. Update everything (takes 5–15 minutes):

   ```bash
   sudo apt update && sudo apt full-upgrade -y
   ```
   ```bash
   sudo reboot
   ```

## Step 4 — Check the camera

After the reboot, open a terminal again and run:

```bash
rpicam-hello --list-cameras
```

You should see one camera listed as **imx219** (that's the v2 sensor).
If you get `command not found`, try `libcamera-hello --list-cameras` instead.

**"No cameras available"?** Power off, re-seat the ribbon cable (Step 2,
check the direction), and try again.

On the Pi's desktop you can also see a live picture (close with Ctrl+C):

```bash
rpicam-hello -t 0
```

## Step 5 — Install the software

```bash
sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-venv
```

Create a Python environment for OSIRIS and install the model runtime:

```bash
python3 -m venv --system-site-packages ~/rice-env
```
```bash
~/rice-env/bin/pip install onnxruntime
```

`--system-site-packages` lets OSIRIS use the camera library and OpenCV
installed above. **Don't** `pip install numpy` or `opencv-python` into this
environment — a second copy of numpy can break the camera library.

Check that everything loads (should print `OK`):

```bash
~/rice-env/bin/python -c "import cv2, numpy, onnxruntime, picamera2; print('OK')"
```

## Step 6 — Download OSIRIS from GitHub

On the **Pi** (replace `USERNAME/REPO` with your GitHub repository):

```bash
git clone https://github.com/USERNAME/REPO.git ~/osiris
```
```bash
ln -s ~/osiris/osiris_pi ~/osiris_pi
```

The first command downloads the whole project to `~/osiris`; the second makes
`~/osiris_pi` point to its Pi folder, so every command in this guide works
as written. Check:

```bash
ls ~/osiris_pi
```

You should see `pi_app.py`, `osiris_gui.py`, `run.sh`, … .

**Download the model.** It is too large for the repository itself (GitHub's
website limit is 25 MB), so it is attached to the repository's **Release**:

```bash
wget -O ~/osiris_pi/densenet121_color.onnx https://github.com/USERNAME/REPO/releases/download/v1.0/densenet121_color.onnx
```

Check it arrived (about 29 MB):

```bash
ls -lh ~/osiris_pi/densenet121_color.onnx
```

*Private repository:* `wget` can't log in, so open the repository's
**Releases** page in the Pi's web browser (signed in to GitHub), download
`densenet121_color.onnx`, and move it into `~/osiris_pi/`.

**Private repository?** GitHub asks for a username and password: the
"password" must be a **personal access token** (GitHub → Settings → Developer
settings → Personal access tokens → *Fine-grained*, read-only access to this
repository), not your GitHub password.

**Updating later** (after changes are pushed to GitHub):

```bash
cd ~/osiris && git pull
```

Your calibration (`settings.sh`) and results are not in Git, so updates never
overwrite them.

*No internet on the Pi?* Copy the `osiris_pi` folder from your laptop with
`scp -r osiris_pi USERNAME@osiris.local:~/` (same network) or a USB stick.

## Step 7 — Test without the camera

Back on the **Pi**:

```bash
cd ~/osiris_pi
```
```bash
~/rice-env/bin/python pi_app.py --model densenet121_color.onnx --image sample.jpg
```

It prints the number of grains, the class mix, and the time taken. For this
sample photo (a Jasmine test photo) you should see:

```
  51 grains -> jasmine 100.0%
  time: segment __s + classify __s (___ ms/grain)
  saved: results/sample/annotated.jpg
```

(The times are your Pi's; they'll be slower than a laptop.)

Open `~/osiris_pi/results/sample/annotated.jpg` (double-click it in the file
manager) — every grain should have a box and a label. **Write down the
ms/grain:** that's the Pi's speed for your paper.

If this works, the software is fully set up. The rest is the camera rig.

## Step 8 — Set up the rig

1. Mount the camera **pointing straight down**, about **21 cm** above the
   black background. The v2 sees an area about 1.2 × its height wide (≈25 ×
   19 cm at 21 cm).
2. Turn on the ring light. Keep room lighting the same every time (no window
   sunlight).
3. **Focus:** the v2 lens comes focused far away. Start the live preview
   (Pi desktop):

   ```bash
   rpicam-hello -t 0
   ```

   Place a few grains under the camera and turn the lens slowly (with the small
   white focus tool that came with the camera, or carefully with tweezers)
   until the grain edges are sharp. Stop the preview with Ctrl+C.

## Step 9 — Calibrate (once)

Put the **coin** and some spread-out rice under the camera, then:

```bash
cd ~/osiris_pi
```
```bash
~/rice-env/bin/python pi_app.py --setup
```

It saves `results/setup.jpg` and prints two things:

```
Auto settings the camera chose — lock them with:
  --exposure-us 18500 --gain 1.12 --awb-gains 1.854 1.512

Coin found: 13.62 px/mm  (field of view 241 x 181 mm)
  fixed rig -> add  --px-per-mm 13.62
```

(Your numbers will differ.) If it warns that px/mm is **below 12**, lower the
camera a little, refocus (Step 8), and run `--setup` again.

It also checks the **grain brightness**:

```
Grain brightness: 131, 0.0% pure white  (training: 90-175, ~0% pure white)
  OK: grain brightness matches the training images
```

**This must say OK.** Over-exposed grains (pure white, no visible texture)
look nothing like the training images, and the model then calls *every* grain
"broken". If it warns, use the `--exposure-us` value it suggests in `settings.sh`,
take a test shot, and repeat until OSIRIS stops warning (every inspection
prints the same brightness check). If it also says the **gain is high**, the
scene is too dark: turn on the ring light or move it closer, then run
`--setup` again.

**Tip — less noise:** the rice doesn't move, so prefer **`GAIN=1.0`** with a
longer exposure (exposure × gain gives the same brightness). For example, if
`--setup` says `--exposure-us 16000 --gain 6.5`, use `EXPOSURE_US=104000` and
`GAIN=1.0`.

Now save your numbers in **`settings.sh`** (this Pi's own settings file):

```bash
nano ~/osiris_pi/settings.sh
```

Type your values, one per line, for example:

```
PX_PER_MM=13.62
EXPOSURE_US=104000
GAIN=1.0
AWB_RED=1.854
AWB_BLUE=1.512
```

Save with **Ctrl+O**, Enter, and exit with **Ctrl+X**. Both `./run.sh` and
the OSIRIS window read this file. To change a value later, edit it again.

**Locked settings matter:** the model uses the grains' color, so every photo
must have the same brightness and white balance.

## Step 10 — Run OSIRIS

1. Spread the rice on the background so **grains don't touch** each other or
   the edge of the picture (touching grains are skipped, shown in grey).
2. Run:

   ```bash
   cd ~/osiris_pi && ./run.sh
   ```

3. Press **Enter** to take a photo. After processing, the result appears on
   screen (on the Pi's desktop); press any key in the image window, then
   **Enter** for the next sample or **q** + Enter to quit.

Each inspection is saved in `~/osiris_pi/results/<date_time>/`:

| File | Contents |
|---|---|
| `annotated.jpg` | each grain boxed with its class and confidence, plus a summary panel (class mix in %) |
| `grains.csv` | one row per grain: class, confidence, length/width/area in mm |
| `summary.json` | counts, percentages, processing time, camera settings |
| `raw.jpg` | the original photo |

A **`?`** after a label means confidence below 70%. In testing, 79% of the
model's mistakes had confidence below 70% — treat those grains as "check
manually".

## Step 11 — The OSIRIS window (GUI)

Instead of the terminal, you can use a simple window with two buttons:

- **Take Picture** — captures with the camera and classifies the grains
- **Upload Image** — classifies an existing photo (e.g. from a USB stick)

The result image and the class mix appear in the window, and everything is
saved in `results/` exactly like `./run.sh`. The window uses your
`settings.sh`, so finish Step 9 first.

Start it from a terminal on the Pi's desktop:

```bash
cd ~/osiris_pi && ~/rice-env/bin/python osiris_gui.py
```

**Open it automatically every time the Pi starts:**

```bash
cd ~/osiris_pi && ./install_autostart.sh
```

Then make the Pi log in to the desktop by itself: run `sudo raspi-config`,
choose **System Options → Boot / Auto Login → Desktop Autologin** (pick it by
its name), then **Finish** and reboot.

After the reboot the Pi logs in to the desktop by itself and OSIRIS opens
after a few seconds.

**If the Pi boots to a black text screen instead of the desktop**, log in
there and run `sudo systemctl set-default graphical.target && sudo reboot`. It is also in the applications menu (Raspberry menu →
Accessories → OSIRIS). If it doesn't open, check `~/osiris_pi/osiris_gui.log`.

To stop it from starting at boot: `./install_autostart.sh --remove`.

Note: automatic login means anyone who powers on the Pi gets the desktop
without a password — fine for a lab prototype.

## Copying results back to your laptop (*laptop*)

```bash
scp -r pi@osiris.local:~/osiris_pi/results ./osiris_results
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ssh: Could not resolve hostname osiris.local` | Use the Pi's IP address instead (on the Pi: `hostname -I`) |
| `No cameras available` | Pi off → re-seat ribbon cable, contacts toward HDMI ports (Step 2) |
| `!! no coin` | Put the coin in view, or set `PX_PER_MM` in `settings.sh` (Step 9) |
| Very few grains found | Grains touching each other/the edge, or image out of focus (Step 8) |
| Photo too dark / too bright | Run `--setup` again with the ring light on, update `settings.sh` |
| `cannot open display` / no window | You're on SSH — results are still saved; `run.sh` skips the window automatically |
| `ModuleNotFoundError: picamera2` | Step 5: venv must be created with `--system-site-packages` |
| numpy error when importing picamera2 | You installed numpy with pip — `~/rice-env/bin/pip uninstall numpy` |

## Before trusting results on new rice

The model was trained on DSLR photos; the Pi camera's colors and sharpness
differ. Check accuracy on Pi photos of known varieties first (Section 1.1 of
the paper). If accuracy drops, take 10–20 Pi photos per class on the rig, add
them to `DATASET/`, re-run `segment_grains.py` and retrain.
