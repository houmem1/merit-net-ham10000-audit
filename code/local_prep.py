# -*- coding: utf-8 -*-
"""Prep LOCAL : telecharge HAM10000 original (si absent), construit les
caches 320/384 + copie PH2/labels dans results/local_input/ (arborescence
compatible kernel_body via DERM_INPUT)."""
import os
import subprocess
import sys
import zipfile

import cv2
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
RAW = os.path.join(RES, "raw_ham")
INP = os.path.join(RES, "local_input")
os.makedirs(RAW, exist_ok=True)
os.makedirs(INP, exist_ok=True)

# ---------- telechargement (une fois) ----------
marker = os.path.join(RAW, "_done")
if not os.path.exists(marker):
    z = os.path.join(RAW, "skin-cancer-mnist-ham10000.zip")
    if not os.path.exists(z):
        print("telechargement HAM10000 (5.5 GB)...", flush=True)
        subprocess.run([sys.executable, "-m", "kaggle", "datasets",
                        "download", "kmader/skin-cancer-mnist-ham10000",
                        "-p", RAW], check=True)
    print("extraction...", flush=True)
    with zipfile.ZipFile(z) as f:
        names = [n for n in f.namelist()
                 if "HAM10000_images_part" in n and n.endswith(".jpg")]
        f.extractall(RAW, members=names)
    open(marker, "w").write("ok")
    os.remove(z)

# ---------- index des images ----------
path = {}
for base, dirs, files in os.walk(RAW):
    for fn in files:
        if fn.endswith(".jpg"):
            path[fn[:-4]] = os.path.join(base, fn)
print("images locales:", len(path), flush=True)

lab = pd.read_csv(os.path.join(RES, "prep_log", "ham_labels.csv"))
assert lab["image_id"].map(lambda i: i in path).all()

for SIZE in (320, 384):
    out = os.path.join(INP, f"ham_{SIZE}.npy")
    if os.path.exists(out):
        print(out, "existe deja")
        continue
    X = np.empty((len(lab), SIZE, SIZE, 3), dtype=np.uint8)
    for i, iid in enumerate(lab["image_id"].values):
        img = cv2.cvtColor(cv2.imread(path[iid]), cv2.COLOR_BGR2RGB)
        X[i] = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if (i + 1) % 2000 == 0:
            print(f"  {SIZE}px {i + 1}/{len(lab)}", flush=True)
    np.save(out, X)
    print("ecrit", out, flush=True)

# ---------- PH2 natif 320/384 (comme les prep Kaggle) ----------
RAWP = os.path.join(RES, "raw_ph2")
os.makedirs(RAWP, exist_ok=True)
pmark = os.path.join(RAWP, "_done")
if not os.path.exists(pmark):
    z = os.path.join(RAWP, "ph2-dataset.zip")
    if not os.path.exists(z):
        print("telechargement PH2 (212 MB)...", flush=True)
        subprocess.run([sys.executable, "-m", "kaggle", "datasets",
                        "download", "spacesurfer/ph2-dataset", "-p", RAWP],
                       check=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(RAWP, members=[n for n in f.namelist()
                                    if "Dermoscopic_Image" in n])
    open(pmark, "w").write("ok")
    os.remove(z)
ph2_imgs = {}
for base, dirs, files in os.walk(RAWP):
    for fn in files:
        if fn.endswith(".bmp") and "Dermoscopic_Image" in base:
            ph2_imgs[fn.split(".")[0]] = os.path.join(base, fn)
ph2 = pd.read_csv(os.path.join(RES, "prep_log", "ph2_labels.csv"))
assert ph2["name"].map(lambda n: n in ph2_imgs).all()
for SIZE in (320, 384):
    out = os.path.join(INP, f"ph2_{SIZE}.npy")
    if os.path.exists(out):
        continue
    P = np.empty((len(ph2), SIZE, SIZE, 3), dtype=np.uint8)
    for i, n in enumerate(ph2["name"].values):
        img = cv2.cvtColor(cv2.imread(ph2_imgs[n]), cv2.COLOR_BGR2RGB)
        P[i] = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    np.save(out, P)
    print("ecrit", out, flush=True)

import shutil
for f in ("ham_labels.csv", "ph2_256.npy", "ph2_labels.csv", "ham_256.npy"):
    src = os.path.join(RES, "prep_log", f)
    dst = os.path.join(INP, f)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
print("local_input pret:", sorted(os.listdir(INP)))
