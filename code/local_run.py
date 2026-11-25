# -*- coding: utf-8 -*-
"""Runner LOCAL : execute kernel_body.py sur la RTX 4070 avec le cache de
results/local_input/, sort dans results/<tag>/ (meme format que Kaggle).
Usage: python local_run.py <tag> '<json-config>'"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
tag = sys.argv[1]
cfg = json.loads(sys.argv[2])
cfg["tag"] = tag
outdir = os.path.join(RES, tag)
os.makedirs(outdir, exist_ok=True)
os.environ["DERM_INPUT"] = os.path.join(RES, "local_input")
os.environ["DERM_OUT"] = outdir
os.environ["DERM_WORKERS"] = "0"   # Windows spawn + exec => pas de workers

body = open(os.path.join(HERE, "kernel_body.py"), encoding="utf-8").read()
src = "CFG = " + repr(cfg) + "\n" + body
code = compile(src, f"kernel_local_{tag}", "exec")
exec(code, {"__name__": "__main__"})
