#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase BOOST (objectif user : >90 %) — recette renforcee du meilleur
modele apparie, memes splits groupes par lesion. Lancee apres la campagne.
Usage: python derm_campaign_boost.py [tags...]"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import derm_campaign as dc

BOOST = [
    # vaisseau amiral boost : MSAP + CE, ConvNeXt-Small, 320px, 16 ep, TTA4
    ("boost", {"folds_run": 5, "arch": "merit", "head": "ce", "loss": "ce",
               "backbone": "convnext_small", "size": 320, "epochs": 16,
               "bs": 24, "tta": "tta4", "save_ckpt": True}),
    # second membre d'ensemble heterogene (si besoin pour franchir 90)
    ("boost2", {"folds_run": 5, "arch": "plain",
                "backbone": "tf_efficientnetv2_s", "head": "ce",
                "loss": "ce", "size": 320, "epochs": 16, "bs": 24,
                "tta": "tta4", "save_ckpt": True}),
]

def main():
    args = sys.argv[1:]
    queue = [(t, c) for t, c in BOOST if not args or t in args]
    queue = [(t, c) for t, c in queue if not os.path.exists(
        os.path.join(dc.RES, t, f"derm_{t}.json"))]
    print("file boost:", [t for t, _ in queue], flush=True)
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
            tag, cfg = queue.pop(0)
            if dc.push(tag, cfg):
                running[tag] = time.time()
                time.sleep(10)
            else:
                failed.append(tag)
        print(time.strftime("%H:%M"), "boost running:",
              {t: int((time.time() - s) / 60)
               for t, s in running.items()},
              "| left:", len(queue), "| done:", done,
              "| failed:", failed, flush=True)
        if queue or running:
            time.sleep(180)
    print("BOOST TERMINE | done:", done, "| failed:", failed, flush=True)

if __name__ == "__main__":
    main()
