# -*- coding: utf-8 -*-
"""Second leakage control: SAME image-level test set as ctl-imgsplit, but
every training/validation image whose lesion also appears in that test set
is REMOVED from the pool ('clean' arm). Same recipe, 3 folds.
Leaky arm = ctl-imgsplit ; clean arm = ctl-imgclean."""
import json, os, subprocess, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
src = json.load(open(os.path.join(RES, "split_imagelevel.json")))
lab = pd.read_csv(os.path.join(RES, "local_input", "ham_labels.csv"))
y = lab["y"].values; les = lab["lesion_id"].values
te = np.array(sorted(src["te_idx"]))
pool = np.array(sorted(set(range(len(y))) - set(te.tolist())))
drop = pool[np.isin(les[pool], les[te])]
clean = np.array(sorted(set(pool.tolist()) - set(drop.tolist())))
assert not set(les[clean]) & set(les[te])
sk = StratifiedKFold(5, shuffle=True, random_state=43)
folds = [sorted(int(i) for i in va) for _, va in sk.split(np.zeros(len(clean)), y[clean])]
out = os.path.join(RES, "split_imageclean.json").replace("\\", "/")
json.dump(dict(te_idx=[int(i) for i in te], drop_idx=[int(i) for i in drop], folds_va=folds), open(out, "w"))
print("clean pool", len(clean), "dropped", len(drop), "test", len(te), flush=True)
tag = "ctl-imgclean"
cfg = {"folds_run": 3, "split_file": out, "image_level_control": True, "arch": "plain",
       "backbone": "convnext_tiny", "head": "ce", "loss": "ce"}
if not os.path.exists(os.path.join(RES, tag, f"derm_{tag}.json")):
    os.makedirs(os.path.join(RES, tag), exist_ok=True)
    t0 = time.time()
    with open(os.path.join(RES, tag, f"derm-{tag}.log"), "w", encoding="utf-8") as lf:
        r = subprocess.run([sys.executable, os.path.join(HERE, "local_run.py"), tag, json.dumps(cfg)],
                           stdout=lf, stderr=subprocess.STDOUT)
    print("DONE" if r.returncode == 0 else "FAIL", tag, f"{(time.time() - t0) / 60:.1f} min", flush=True)
