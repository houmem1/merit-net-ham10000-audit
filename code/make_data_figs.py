# -*- coding: utf-8 -*-
"""Figures independantes des resultats : vignettes pour le diagramme,
distribution des classes, grille d'exemples. Donnees reelles (cache 256px)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(HERE, "..", "results", "prep_log")
FIG = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

lab = pd.read_csv(os.path.join(PREP, "ham_labels.csv"))
X = np.load(os.path.join(PREP, "ham_256.npy"), mmap_mode="r")
CLASSES = sorted(lab["dx"].unique())
LONG = dict(akiec="Actinic keratosis", bcc="Basal cell carcinoma",
            bkl="Benign keratosis", df="Dermatofibroma", mel="Melanoma",
            nv="Melanocytic nevus", vasc="Vascular lesion")

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})

# ---- vignettes pour le diagramme (1 nevus, 1 melanome) ----
rng = np.random.RandomState(0)
for dx, name in (("mel", "arch_input"), ("nv", "arch_input2")):
    i = lab.index[lab.dx == dx][7]
    plt.imsave(os.path.join(FIG, f"{name}.png"), X[i])

# ---- distribution des classes (images + lesions) ----
img_c = lab.dx.value_counts().reindex(CLASSES)
les_c = lab.groupby("dx").lesion_id.nunique().reindex(CLASSES)
order = img_c.sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(3.5, 2.3))
x = np.arange(len(order))
ax.bar(x - 0.21, img_c[order], width=0.42, label="images",
       color="#4878a8")
ax.bar(x + 0.21, les_c[order], width=0.42, label="lesions",
       color="#c44e52")
for xi, v in zip(x, img_c[order]):
    ax.text(xi - 0.21, v * 1.12, str(v), ha="center", fontsize=6)
ax.set_yscale("log")
ax.set_xticks(x, order)
ax.set_ylabel("count (log)")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_classdist.pdf"))
plt.close(fig)

# ---- grille d'exemples 7 x 3 ----
fig, axes = plt.subplots(3, 7, figsize=(7.0, 3.3))
for c, dx in enumerate(order):
    idx = lab.index[lab.dx == dx].values
    pick = rng.choice(idx, 3, replace=False)
    for r in range(3):
        ax = axes[r, c]
        ax.imshow(X[pick[r]])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if r == 0:
            ax.set_title(f"{dx}\n({LONG[dx]})", fontsize=6.5)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "fig_examples.pdf"), bbox_inches="tight")
plt.close(fig)
print("figures data OK:", sorted(os.listdir(FIG)))
