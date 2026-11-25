# -*- coding: utf-8 -*-
"""Revision experiments (BMC Medical Imaging, round 1) - run LOCALLY on the
RTX 4070 with the campaign code (kernel_body.py) and the campaign split.
  ctl-imgsplit    : ConvNeXt-T + CE on an IMAGE-level split (leakage control)
  m-edl-lam01     : MERIT, unweighted EDL, lambda_max=0.1 (weights x KL cell)
  cnx-cbedl-lam01 : plain ConvNeXt-T, CB-EDL, lambda_max=0.1 (replication)
  *-s43           : seed-43 replicates of main / m-lam01 / merit-ce
All 3 folds, matched recipe. Sequential; skips finished tags."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
SPLIT = os.path.join(RES, "kaggle_split.json").replace("\\", "/")
IMG_SPLIT = os.path.join(RES, "split_imagelevel.json").replace("\\", "/")

# ---- image-level control split: same sizes/procedure, NO lesion grouping
if not os.path.exists(IMG_SPLIT):
    lab = pd.read_csv(os.path.join(RES, "local_input", "ham_labels.csv"))
    y = lab["y"].values
    sk1 = StratifiedKFold(5, shuffle=True, random_state=42)
    trv, te = next(iter(sk1.split(np.zeros(len(y)), y)))
    trv, te = np.sort(trv), np.sort(te)
    sk2 = StratifiedKFold(5, shuffle=True, random_state=43)
    folds = [sorted(int(i) for i in va)
             for _, va in sk2.split(np.zeros(len(trv)), y[trv])]
    json.dump(dict(te_idx=[int(i) for i in te], folds_va=folds),
              open(IMG_SPLIT, "w"))
    print("image-level split written", len(te), [len(f) for f in folds])

F3 = {"folds_run": 3, "split_file": SPLIT}
QUEUE = [
    ("ctl-imgsplit", {"folds_run": 3, "split_file": IMG_SPLIT,
                      "image_level_control": True, "arch": "plain",
                      "backbone": "convnext_tiny", "head": "ce",
                      "loss": "ce"}),
    ("m-edl-lam01", {**F3, "loss": "edl", "lam_max": 0.1}),
    ("cnx-cbedl-lam01", {**F3, "arch": "plain",
                         "backbone": "convnext_tiny", "head": "edl",
                         "loss": "cbedl", "lam_max": 0.1}),
    ("m-lam01-s43", {**F3, "lam_max": 0.1, "seed": 43}),
    ("main-s43", {**F3, "seed": 43}),
    ("merit-ce-s43", {**F3, "head": "ce", "loss": "ce", "seed": 43}),
]
only = sys.argv[1:]
for tag, cfg in QUEUE:
    if only and tag not in only:
        continue
    out = os.path.join(RES, tag, f"derm_{tag}.json")
    if os.path.exists(out):
        print("skip", tag, flush=True)
        continue
    t0 = time.time()
    print("RUN", tag, json.dumps(cfg), flush=True)
    os.makedirs(os.path.join(RES, tag), exist_ok=True)
    with open(os.path.join(RES, tag, f"derm-{tag}.log"), "w",
              encoding="utf-8") as lf:
        r = subprocess.run([sys.executable,
                            os.path.join(HERE, "local_run.py"), tag,
                            json.dumps(cfg)], stdout=lf,
                           stderr=subprocess.STDOUT)
    print("DONE" if r.returncode == 0 else "FAIL", tag,
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
