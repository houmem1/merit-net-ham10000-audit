# -*- coding: utf-8 -*-
"""Assemble the public release folder (no images, no checkpoints):
splits, per-run configurations, per-fold predictions, code, analysis
outputs. -> DERM_journal/release/merit-net-ham10000-audit/"""
import json
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "release", "merit-net-ham10000-audit")
if os.path.exists(OUT):
    for d in ("splits", "predictions", "code", "analysis", "configs"):
        shutil.rmtree(os.path.join(OUT, d), ignore_errors=True)
for d in ("splits", "predictions", "code", "analysis", "configs"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

MATCHED = ["main", "merit-ce", "cnx-cbedl", "cnx-ce", "swinv2t-ce",
           "deit3s-ce", "effb3-ce", "dense121-ce", "r50-ce", "cnx-wce",
           "cnx-focal", "cnx-cbce", "cnx-edl", "m-gap", "m-attplain",
           "m-single", "m-fusemean", "m-lam0", "m-lam01", "m-beta9999",
           "m-edl", "m-effb3", "m-224"]
PERF = ["boost", "boost2", "boost3", "boost4", "boost5", "screen1",
        "screen2"]
REVN = ["ctl-imgsplit", "ctl-imgclean", "m-edl-lam01", "cnx-cbedl-lam01",
        "m-lam01-s43", "main-s43", "merit-ce-s43"]
LABEL = {"boost": "P1", "boost2": "P2", "boost3": "P3", "boost4": "P4",
         "boost5": "P5", "screen1": "B1", "screen2": "B2",
         "main": "MERIT-Net (MSAP + CB-EDL)", "merit-ce": "MSAP + CE"}
cfgs, missing = {}, []
for grp, tags in (("matched", MATCHED), ("performance", PERF),
                  ("revision", REVN)):
    for t in tags:
        src = os.path.join(RES, t, f"derm_{t}.json")
        if not os.path.exists(src):
            missing.append(t)
            continue
        d = json.load(open(src))
        c = d["config"]
        for k in ("split_file", "split_inline"):
            if k in c and isinstance(c[k], str):
                c[k] = os.path.basename(c[k])   # no local absolute paths
        json.dump(d, open(os.path.join(OUT, "predictions",
                                       f"derm_{t}.json"), "w"))
        cfgs[t] = dict(group=grp, paper_label=LABEL.get(t, t), config=c)
        log = os.path.join(RES, t, f"derm-{t}.log")
        if os.path.exists(log):
            txt = open(log, encoding="utf-8", errors="replace").read()
            txt = txt.replace(RES.replace(os.sep, "/") + "/", "")
            txt = txt.replace(RES + os.sep, "")
            open(os.path.join(OUT, "predictions", f"derm-{t}.log"), "w",
                 encoding="utf-8").write(txt)
json.dump(cfgs, open(os.path.join(OUT, "configs", "configs.json"), "w"),
          indent=1)
for f in ("kaggle_split.json", "split_imagelevel.json",
          "split_imageclean.json"):
    if os.path.exists(os.path.join(RES, f)):
        shutil.copy2(os.path.join(RES, f), os.path.join(OUT, "splits", f))
shutil.copy2(os.path.join(RES, "revision", "split_assignments.csv"),
             os.path.join(OUT, "splits", "split_assignments.csv"))
for f in ("rev_analysis.json", "rev_analysis_log.txt", "dupscreen.json",
          "abhishek2025_missed_duplicates.csv",
          "abhishek2025_pairs_partition.csv"):
    shutil.copy2(os.path.join(RES, "revision", f),
                 os.path.join(OUT, "analysis", f))
shutil.copy2(os.path.join(RES, "complexity.json"),
             os.path.join(OUT, "analysis", "complexity.json"))
CODE = ["kernel_body.py", "local_run.py", "push_run.py", "derm_campaign.py",
        "derm_campaign_boost.py", "derm_campaign_boost2.py",
        "revision_runs.py", "revision_runs2.py", "prep_kernel.py",
        "prep_kernel320.py", "prep_kernel384.py", "local_prep.py",
        "derm_complexity.py", "rev_analysis.py", "rev_make_assets.py",
        "rev_dupscreen.py", "make_interp_figs.py", "make_data_figs.py",
        "make_release.py"]
for f in CODE:
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        shutil.copy2(p, os.path.join(OUT, "code", f))
print("release assembled:", OUT)
print("runs:", len(cfgs), "| missing:", missing)
tot = 0
for b, _, fs in os.walk(OUT):
    tot += sum(os.path.getsize(os.path.join(b, f)) for f in fs)
print(f"size {tot / 1e6:.1f} MB")
