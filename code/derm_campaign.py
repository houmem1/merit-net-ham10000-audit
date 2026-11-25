#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Orchestrateur de campagne MERIT-Net : pousse les kernels Kaggle (max 2
concurrents), surveille, rapatrie les sorties dans ../results/<tag>/.

Usage:  python derm_campaign.py            # campagne complete
        python derm_campaign.py --list     # affiche la file
        python derm_campaign.py t1 t2 ...  # seulement ces tags
"""
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
USER = "houmemslimi"
MAXRUN = 2
PY = sys.executable

# ----------------------------------------------------------------- configs
F5 = {"folds_run": 5}
F3 = {"folds_run": 3}
CN = {"arch": "plain", "backbone": "convnext_tiny"}
CE = {"head": "ce", "loss": "ce"}
QUEUE = [
    # --- coeur factoriel 2x2 (arch x objectif), 5 folds ---
    ("main", {**F5, "save_extras": True, "save_ckpt": True}),
    ("merit-ce", {**F5, "head": "ce", "loss": "ce"}),
    ("cnx-ce", {**F5, **CN, **CE}),
    ("cnx-cbedl", {**F5, **CN, "head": "edl", "loss": "cbedl"}),
    # --- baselines architecturales (plain + CE), 3 folds ---
    ("r50-ce", {**F3, "arch": "plain", "backbone": "resnet50", **CE}),
    ("effb3-ce", {**F3, "arch": "plain",
                  "backbone": "tf_efficientnet_b3", **CE}),
    ("deit3s-ce", {**F3, "arch": "plain",
                   "backbone": "deit3_small_patch16_224", **CE}),
    ("swinv2t-ce", {**F3, "arch": "plain",
                    "backbone": "swinv2_tiny_window8_256", **CE}),
    ("dense121-ce", {**F3, "arch": "plain",
                     "backbone": "densenet121", **CE}),
    # --- objectifs de desequilibre sur ConvNeXt plain, 3 folds ---
    ("cnx-wce", {**F3, **CN, "head": "ce", "loss": "wce"}),
    ("cnx-focal", {**F3, **CN, "head": "ce", "loss": "focal"}),
    ("cnx-cbce", {**F3, **CN, "head": "ce", "loss": "cbce"}),
    ("cnx-edl", {**F3, **CN, "head": "edl", "loss": "edl"}),
    # --- ablations MERIT, 3 folds ---
    ("m-gap", {**F3, "pool": "mean"}),
    ("m-attplain", {**F3, "pool": "att_plain"}),
    ("m-single", {**F3, "scales": [3]}),
    ("m-fusemean", {**F3, "scale_fuse": "mean"}),
    ("m-lam0", {**F3, "lam_max": 0.0}),
    ("m-lam01", {**F3, "lam_max": 0.1}),
    ("m-beta9999", {**F3, "beta_eff": 0.9999}),
    ("m-edl", {**F3, "loss": "edl"}),
    ("m-effb3", {**F3, "backbone": "tf_efficientnet_b3"}),
    ("m-224", {**F3, "size": 224}),
]

def sh(*args, timeout=180):
    r = subprocess.run([PY, "-m", "kaggle"] + list(args),
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr

def status(tag):
    out = sh("kernels", "status", f"{USER}/derm-{tag}")
    for s in ("COMPLETE", "RUNNING", "QUEUED", "ERROR", "CANCEL"):
        if s.lower() in out.lower():
            return s
    return "UNKNOWN:" + out.strip()[:120]

def push(tag, cfg):
    r = subprocess.run([PY, os.path.join(HERE, "push_run.py"), tag,
                        json.dumps(cfg)], capture_output=True, text=True)
    ok = "successfully" in (r.stdout + r.stderr).lower()
    print(("PUSH " if ok else "PUSH-FAIL ") + tag, flush=True)
    return ok

def fetch(tag):
    d = os.path.join(RES, tag)
    os.makedirs(d, exist_ok=True)
    out = sh("kernels", "output", f"{USER}/derm-{tag}", "-p", d,
             timeout=1200)
    ok = os.path.exists(os.path.join(d, f"derm_{tag}.json"))
    print(("FETCH " if ok else "FETCH-FAIL ") + tag + " | " +
          out.strip().splitlines()[-1][:100] if out.strip() else "", flush=True)
    return ok

def main():
    args = sys.argv[1:]
    if args and args[0] == "--list":
        for t, c in QUEUE:
            print(t, json.dumps(c))
        return
    queue = [(t, c) for t, c in QUEUE if not args or t in args]
    queue = [(t, c) for t, c in queue
             if not os.path.exists(os.path.join(RES, t, f"derm_{t}.json"))]
    print("file d'attente:", [t for t, _ in queue], flush=True)
    running = {}
    failed, done = [], []
    while queue or running:
        for tag in list(running):
            st = status(tag)
            if st == "COMPLETE":
                del running[tag]
                (done if fetch(tag) else failed).append(tag)
            elif st in ("ERROR", "CANCEL") or st.startswith("UNKNOWN"):
                del running[tag]
                failed.append(tag)
                print("FAILED", tag, st, flush=True)
        while queue and len(running) < MAXRUN:
            tag, cfg = queue.pop(0)
            if push(tag, cfg):
                running[tag] = time.time()
                time.sleep(10)
            else:
                failed.append(tag)
        print(time.strftime("%H:%M"), "running:",
              {t: int((time.time() - s) / 60) for t, s in running.items()},
              "| left:", len(queue), "| done:", len(done),
              "| failed:", failed, flush=True)
        if queue or running:
            time.sleep(180)
    print("CAMPAGNE TERMINEE | done:", done, "| failed:", failed, flush=True)

if __name__ == "__main__":
    main()
