import os, random, shutil

random.seed(0)

SRC = "Rice_Image_Dataset"
DST = "dataset"
VAL_FRAC = 0.15 

for cls in ["Jasmine", "Basmati"]:
    files = sorted(os.listdir(f"{SRC}/{cls}"))
    random.shuffle(files)
    n_val = int(len(files) * VAL_FRAC)
    for split, chunk in [("val", files[:n_val]), ("train", files[n_val:])]:
        dst_dir = f"{DST}/{split}/{cls.lower()}"
        os.makedirs(dst_dir, exist_ok=True)
        for f in chunk:
            shutil.copy(f"{SRC}/{cls}/{f}", f"{dst_dir}/{f}")
    print(f"{cls}: train={len(files)-n_val}, val={n_val}")