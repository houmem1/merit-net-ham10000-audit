# -*- coding: utf-8 -*-
"""Figures d'interpretabilite depuis les artefacts REELS du run main :
  - fig_attention.pdf : cartes d'attention par echelle (ckpt fold 1, local)
  - fig_tsne.pdf      : t-SNE des embeddings z (extras_main.npz)
  - fig_beta.pdf      : distribution des portes d'echelle beta par classe
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PREP = os.path.join(RES, "prep_log")
FIG = os.path.join(HERE, "..", "paper", "figures")
sys.path.insert(0, HERE)
from derm_complexity import MERIT  # memes classes que kernel_body

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                     "figure.dpi": 200})
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

ex = np.load(os.path.join(RES, "main", "extras_main.npz"))
z, beta, yt, p, u = (ex["z"], ex["beta"], ex["y"], ex["p"], ex["u"])
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL = 4

# ---------------- t-SNE ----------------
from sklearn.manifold import TSNE
emb = TSNE(2, random_state=42, perplexity=30,
           init="pca").fit_transform(z.astype(np.float64))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
cmap = plt.get_cmap("tab10")
for k, cls in enumerate(CLASSES):
    m = yt == k
    axes[0].scatter(emb[m, 0], emb[m, 1], s=3, color=cmap(k), label=cls,
                    alpha=0.7, linewidths=0)
axes[0].legend(frameon=False, fontsize=6, markerscale=2, ncol=2)
axes[0].set_title("t-SNE of test embeddings (class)", fontsize=8)
sc = axes[1].scatter(emb[:, 0], emb[:, 1], s=3, c=u, cmap="viridis",
                     alpha=0.8, linewidths=0)
plt.colorbar(sc, ax=axes[1], fraction=0.045, label=r"uncertainty $u$")
axes[1].set_title("t-SNE coloured by evidential uncertainty", fontsize=8)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_tsne.pdf")); plt.close(fig)
print("tsne OK")

# ---------------- beta par classe ----------------
fig, ax = plt.subplots(figsize=(3.5, 2.3))
x = np.arange(len(CLASSES))
bot = np.zeros(len(CLASSES))
cols = ["#4878a8", "#55a868", "#c44e52"]
for s in range(beta.shape[1]):
    vals = np.array([beta[yt == k, s].mean() for k in range(7)])
    ax.bar(x, vals, bottom=bot, color=cols[s],
           label=f"scale {s + 1} (stride {8 * 2 ** s})")
    bot += vals
ax.set_xticks(x, CLASSES)
ax.set_ylabel(r"mean scale gate $\beta_s$")
ax.legend(frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_beta.pdf")); plt.close(fig)
print("beta OK | beta mean:", beta.mean(0))

# ---------------- cartes d'attention (ckpt fold 1) ----------------
import json
lab = pd.read_csv(os.path.join(PREP, "ham_labels.csv"))
X = np.load(os.path.join(PREP, "ham_256.npy"), mmap_mode="r")
y_all = lab["y"].values
# indices de test = ceux sauvegardes PAR LE KERNEL (sklearn Kaggle)
te_i = np.array(json.load(open(os.path.join(
    RES, "main", "derm_main.json")))["te_idx"])
assert len(te_i) == len(yt) and (y_all[te_i] == yt).all()

model = MERIT()
sd = torch.load(os.path.join(RES, "main", "ckpt_main_f1.pt"),
                map_location="cpu")
model.load_state_dict(sd)
model = model.to(DEV).eval()

atts = {}
def hook(s):
    def fn(mod, inp, out):
        atts[s] = out[1].detach().float().cpu().numpy()
    return fn
for s, pool in enumerate(model.pools):
    pool.w.register_forward_hook(lambda m, i, o: None)
hs = [model.pools[s].register_forward_hook(
    (lambda s0: lambda m, i, o:
     atts.__setitem__(s0, o[1].detach().float().cpu().numpy()))(s))
      for s in range(3)]

# choix de cas reels : melanome correct sur, melanome correct incertain,
# erreur tres incertaine, nv correct tres sur
pred = p.argmax(1)
cand = dict(
    mel_conf=np.where((yt == MEL) & (pred == MEL))[0],
    mel_unc=np.where((yt == MEL) & (pred == MEL))[0],
    err_unc=np.where(pred != yt)[0],
    nv_conf=np.where((yt == 5) & (pred == 5))[0])
pick = dict(
    mel_conf=cand["mel_conf"][np.argmin(u[cand["mel_conf"]])],
    mel_unc=cand["mel_unc"][np.argmax(u[cand["mel_unc"]])],
    err_unc=cand["err_unc"][np.argmax(u[cand["err_unc"]])],
    nv_conf=cand["nv_conf"][np.argmin(u[cand["nv_conf"]])])
TITLES = dict(mel_conf="melanoma, correct, low $u$",
              mel_unc="melanoma, correct, high $u$",
              err_unc="misclassified, highest $u$",
              nv_conf="nevus, correct, low $u$")

fig, axes = plt.subplots(4, 4, figsize=(7.0, 7.2))
with torch.no_grad():
    for r, (key, ti) in enumerate(pick.items()):
        gi = te_i[ti]
        im = np.ascontiguousarray(X[gi])
        xx = torch.from_numpy(((im.astype(np.float32) / 255 - MEAN)
                               / STD).transpose(2, 0, 1)[None]).to(DEV)
        atts.clear()
        model(xx)
        axes[r, 0].imshow(im)
        axes[r, 0].set_title(
            f"{TITLES[key]}\ntrue={CLASSES[yt[ti]]} "
            f"pred={CLASSES[pred[ti]]} u={u[ti]:.3f}", fontsize=6.5)
        for s in range(3):
            g = int(np.sqrt(atts[s].shape[1]))
            amap = atts[s][0].reshape(g, g)
            amap = F.interpolate(torch.tensor(amap)[None, None],
                                 size=(256, 256), mode="bilinear",
                                 align_corners=False)[0, 0].numpy()
            axes[r, s + 1].imshow(im)
            axes[r, s + 1].imshow(amap, cmap="inferno", alpha=0.45)
            if r == 0:
                axes[r, s + 1].set_title(
                    f"attention, stride {8 * 2 ** s}", fontsize=7)
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIG, "fig_attention.pdf"), bbox_inches="tight")
plt.close(fig)
for h in hs:
    h.remove()
print("attention OK | picks:", {k: int(v) for k, v in pick.items()})
