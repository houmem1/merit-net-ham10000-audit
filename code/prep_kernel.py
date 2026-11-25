# ================= derm-prep : cache HAM10000 + PH2 =================
# Resize uint8 256px caches so training kernels skip JPEG decoding.
import os
import re
import time

import cv2
import numpy as np
import pandas as pd

t0 = time.time()
INP = "/kaggle/input"
OUT = "/kaggle/working"
SIZE = 256

# ---------------- HAM10000 ----------------
meta = None
imgdirs = []
ph2_txt = None
ph2_imgs = {}
for base, dirs, files in os.walk(INP):
    for f in files:
        if f == "HAM10000_metadata.csv" and meta is None:
            meta = os.path.join(base, f)
        if f == "PH2_dataset.txt" and ph2_txt is None:
            ph2_txt = os.path.join(base, f)
        if f.endswith(".bmp") and "Dermoscopic_Image" in base:
            ph2_imgs[f.split(".")[0]] = os.path.join(base, f)
    if os.path.basename(base).startswith("HAM10000_images_part"):
        imgdirs.append(base)
print("meta:", meta, "| imgdirs:", imgdirs, "| ph2 txt:", ph2_txt,
      "| ph2 imgs:", len(ph2_imgs), flush=True)

df = pd.read_csv(meta).sort_values("image_id").reset_index(drop=True)  # frozen
path = {}
for d in imgdirs:
    for f in os.listdir(d):
        if f.endswith(".jpg"):
            path[f[:-4]] = os.path.join(d, f)
df["path"] = df["image_id"].map(path)
assert df["path"].notna().all(), "missing images"
CLASSES = sorted(df["dx"].unique())
df["y"] = df["dx"].map({c: i for i, c in enumerate(CLASSES)})
print("classes:", CLASSES, "| images:", len(df), flush=True)

X = np.empty((len(df), SIZE, SIZE, 3), dtype=np.uint8)
for i, p in enumerate(df["path"].values):
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    X[i] = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    if (i + 1) % 1000 == 0:
        print("  ham", i + 1, flush=True)
np.save(os.path.join(OUT, f"ham_{SIZE}.npy"), X)
df[["image_id", "lesion_id", "dx", "y"]].to_csv(
    os.path.join(OUT, "ham_labels.csv"), index=False)

# ---------------- PH2 (external zero-shot) ----------------
rows = []
for line in open(ph2_txt, encoding="utf-8", errors="replace"):
    m = re.match(r"\s*\|\|\s*(IMD\d+)\s*\|\|[^|]*\|\|\s*(\d)\s*\|\|", line)
    if m:
        rows.append((m.group(1), int(m.group(2))))
ph2 = pd.DataFrame(rows, columns=["name", "clin"]).sort_values(
    "name").reset_index(drop=True)
print("ph2 labelled:", len(ph2), "| counts:",
      ph2["clin"].value_counts().to_dict(), flush=True)
ph2["found"] = ph2["name"].map(lambda n: n in ph2_imgs)
assert ph2["found"].all(), ph2[~ph2.found]
P = np.empty((len(ph2), SIZE, SIZE, 3), dtype=np.uint8)
for i, n in enumerate(ph2["name"].values):
    img = cv2.cvtColor(cv2.imread(ph2_imgs[n]), cv2.COLOR_BGR2RGB)
    P[i] = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
np.save(os.path.join(OUT, f"ph2_{SIZE}.npy"), P)
ph2[["name", "clin"]].to_csv(os.path.join(OUT, "ph2_labels.csv"), index=False)
print("DONE %.1f min" % ((time.time() - t0) / 60), flush=True)
