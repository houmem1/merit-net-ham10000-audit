#!/usr/bin/env python
"""Push a MERIT-Net run (or the prep) to Kaggle as a GPU script kernel.

Usage:
    python push_run.py prep
    python push_run.py <tag> '<json-config-overrides>'
e.g.
    python push_run.py smoke '{"folds_run":2,"epochs":2,"max_epsamples":640}'
    python push_run.py main  '{"folds_run":5,"save_extras":true}'

Then:  python -m kaggle kernels status houmemslimi/derm-<tag>
       python -m kaggle kernels output houmemslimi/derm-<tag> -p ../results/<tag>
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
USER = "houmemslimi"

def main():
    tag = sys.argv[1]
    d = tempfile.mkdtemp()
    if tag in ("prep", "prep320", "prep384"):
        fn = {"prep": "prep_kernel.py", "prep320": "prep_kernel320.py", "prep384": "prep_kernel384.py"}[tag]
        src = open(os.path.join(HERE, fn), encoding="utf-8").read()
        meta = {
            "id": f"{USER}/derm-{tag}",
            "title": f"derm-{tag}",
            "code_file": f"derm-{tag}.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_internet": "false",
            "dataset_sources": ["kmader/skin-cancer-mnist-ham10000",
                                "spacesurfer/ph2-dataset"],
            "competition_sources": [],
            "kernel_sources": [],
        }
        fname = f"derm-{tag}.py"
    else:
        raw = sys.argv[2] if len(sys.argv) > 2 else "{}"
        if raw.startswith("@"):            # @fichier.json (gros configs)
            raw = open(raw[1:], encoding="utf-8").read()
        over = json.loads(raw)
        over["tag"] = tag
        body = open(os.path.join(HERE, "kernel_body.py"),
                    encoding="utf-8").read()
        src = "CFG = " + repr(over) + "\n" + body   # repr => Python literals
        meta = {
            "id": f"{USER}/derm-{tag}",
            "title": f"derm-{tag}",
            "code_file": f"derm-{tag}.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [f"{USER}/derm-prep"]
            + ([f"{USER}/derm-prep320"] if over.get("size", 256) == 320
               else [])
            + ([f"{USER}/derm-prep384"] if over.get("size", 256) == 384
               else []),
        }
        fname = f"derm-{tag}.py"
    with open(os.path.join(d, fname), "w", encoding="utf-8") as f:
        f.write(src)
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f)
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                        "-p", d], capture_output=True, text=True)
    print(r.stdout, r.stderr)
    shutil.rmtree(d, ignore_errors=True)
    sys.exit(0 if "successfully" in (r.stdout + r.stderr).lower() else 1)

if __name__ == "__main__":
    main()
