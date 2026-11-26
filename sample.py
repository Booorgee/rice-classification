import os, random, shutil
random.seed(0)
for split in ["train"]:
    for cls in ["jasmine", "basmati"]:
        src = f"dataset/{split}/{cls}"
        files = sorted(os.listdir(src)); random.shuffle(files)
        os.makedirs(f"subset/{split}/{cls}", exist_ok=True)
        for f in files[:2000]:
            shutil.copy(f"{src}/{f}", f"subset/{split}/{cls}/{f}")
# copy val as-is
shutil.copytree("dataset/val", "subset/val", dirs_exist_ok=True)