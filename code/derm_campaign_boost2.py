#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Boost round 2 (gate >90 %) : 3e membre d'ensemble DeiT3-S@320 + montee
384px du meilleur single (EffNetV2-S) avec mixup. Attend derm-prep384 avant
de pousser la config 384."""
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import derm_campaign as dc

BOOST2 = [
    ("boost3", {"folds_run": 5, "arch": "plain",
                "backbone": "deit3_small_patch16_224", "head": "ce",
                "loss": "ce", "size": 320, "epochs": 16, "bs": 24,
                "tta": "tta4", "mixup": 0.2, "save_ckpt": True}),
    ("boost4", {"folds_run": 5, "arch": "plain",
                "backbone": "tf_efficientnetv2_s", "head": "ce",
                "loss": "ce", "size": 384, "epochs": 16, "bs": 16,
                "tta": "tta4", "mixup": 0.2, "save_ckpt": True}),
]

def prep384_ready():
    out = subprocess.run([sys.executable, "-m", "kaggle", "kernels",
                          "status", f"{dc.USER}/derm-prep384"],
                         capture_output=True, text=True)
    return "complete" in (out.stdout + out.stderr).lower()

def main():
    queue = [(t, c) for t, c in BOOST2 if not os.path.exists(
        os.path.join(dc.RES, t, f"derm_{t}.json"))]
    print("file boost2:", [t for t, _ in queue], flush=True)
    running = {}
    failed, done = [], []
    while queue or running:
        for tag in list(running):
            st = dc.status(tag)
            if st == "COMPLETE":
                del running[tag]
                (done if dc.fetch(tag) else failed).append(tag)
            elif st in ("ERROR", "CANCEL") or st.startswith("UNKNOWN"):
                del running[tag]
                failed.append(tag)
                print("FAILED", tag, st, flush=True)
        while queue and len(running) < dc.MAXRUN:
            tag, cfg = queue[0]
            if cfg.get("size") == 384 and not prep384_ready():
                print("attente derm-prep384...", flush=True)
                break
            queue.pop(0)
            if dc.push(tag, cfg):
                running[tag] = time.time()
                time.sleep(10)
            else:
                failed.append(tag)
        print(time.strftime("%H:%M"), "boost2 running:",
              {t: int((time.time() - s) / 60)
               for t, s in running.items()},
              "| left:", len(queue), "| done:", done,
              "| failed:", failed, flush=True)
        if queue or running:
            time.sleep(180)
    print("BOOST2 TERMINE | done:", done, "| failed:", failed, flush=True)

if __name__ == "__main__":
    main()
