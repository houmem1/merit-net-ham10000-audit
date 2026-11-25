# ===================== MERIT-Net — Kaggle kernel body =====================
# Prepended by push_run.py with:  CFG = {...}
# Trains MERIT-Net (or a baseline/ablation variant) on HAM10000 under a frozen
# lesion-grouped test split + lesion-grouped K-fold CV, evaluates PH2 zero-shot,
# and writes results JSON (+ optional extras npz / checkpoints) to
# /kaggle/working. Every reported number comes from this run.
import json
import math
import os
import random
import subprocess
import sys
import time

t00 = time.time()
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"],
                   check=True)
    import timm

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

INPUT_ROOT = os.environ.get("DERM_INPUT", "/kaggle/input")
OUT_ROOT = os.environ.get("DERM_OUT", "/kaggle/working")
FAKE = os.environ.get("DERM_FAKE", "") == "1"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = ("cuda" if torch.cuda.is_available()
          and torch.cuda.device_count() > 0 else "cpu")
AMP = device == "cuda"
print("device:", device,
      torch.cuda.get_device_name(0) if AMP else "", flush=True)

C = dict(size=256, folds_run=5, epochs=10, bs=32, lr=2e-4, wd=1e-2,
         arch="merit", backbone="convnext_tiny", scales=(1, 2, 3),
         pool="att_gated", scale_fuse="gate", head="edl", loss="cbedl",
         beta_eff=0.999, mixup=0.0,
         lam_max=1.0, t_anneal=10, focal_gamma=2.0, ls=0.1, dim=256,
         ema=0.999, tta=True, pretrained=True, save_extras=False,
         save_ckpt=False, max_epsamples=0, tag="run")
C.setdefault("task", "multi")      # "multi" (7 classes) | "screen" (binaire)
C.update(CFG)  # noqa: F821  (injected by push_run.py)
# revision: optional seed override (default 42 = campaign behaviour) and an
# explicit flag for the image-level *control* split, which deliberately
# violates lesion disjointness (leakage-inflation experiment only).
C.setdefault("seed", 42)
if int(C["seed"]) != SEED:
    SEED = int(C["seed"])
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
LEAK_CONTROL = bool(C.get("image_level_control", False))
C["scales"] = tuple(C["scales"])
# scale indices are defined on ConvNeXt's 4-level pyramid (strides
# 4/8/16/32); EfficientNet exposes 5 levels (2/4/8/16/32), so remap to
# keep the same strides 8/16/32 (resp. 32 for single-scale).
if C["arch"] == "merit" and C["backbone"].startswith("tf_efficientnet"):
    C["scales"] = {(1, 2, 3): (2, 3, 4), (3,): (4,)}.get(
        C["scales"], C["scales"])
print("config:", C, flush=True)
K = 7
MEL, NV = 4, 5           # indices in sorted classes (checked after load)

# ----------------------------- data -----------------------------
def find_cache():
    hits = {}
    for base, dirs, files in os.walk(INPUT_ROOT):
        for f in files:
            if f.startswith(("ham_", "ph2_")):
                hits[f] = os.path.join(base, f)
    return hits

if FAKE:                                   # local dry-run: synthetic data
    n = 400
    X_all = np.random.randint(0, 255, (n, C["size"], C["size"], 3), np.uint8)
    y_all = np.random.randint(0, K, n)
    les = np.array(["L%d" % (i // 2) for i in range(n)])
    Xp2 = np.random.randint(0, 255, (40, C["size"], C["size"], 3), np.uint8)
    yp2 = np.random.randint(0, 2, 40)
    CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
else:
    cache = find_cache()
    print("cache files:", sorted(cache), flush=True)
    lab = pd.read_csv(cache["ham_labels.csv"])
    CLASSES = sorted(lab["dx"].unique())
    assert CLASSES[MEL] == "mel" and CLASSES[NV] == "nv", CLASSES
    assert (lab["y"].values == lab["dx"].map(
        {c: i for i, c in enumerate(CLASSES)}).values).all()
    y_all = lab["y"].values.astype(np.int64)
    les = lab["lesion_id"].values
    hk = (f"ham_{C['size']}.npy" if f"ham_{C['size']}.npy" in cache
          else "ham_256.npy")
    pk = (f"ph2_{C['size']}.npy" if f"ph2_{C['size']}.npy" in cache
          else "ph2_256.npy")
    X_all = np.load(cache[hk], mmap_mode="r")
    ph2 = pd.read_csv(cache["ph2_labels.csv"])
    Xp2 = np.load(cache[pk])
    print("cache utilise:", hk, pk, flush=True)
    yp2 = (ph2["clin"].values == 2).astype(np.int64)   # melanoma vs nevus
    assert len(X_all) == len(y_all)

if C["task"] == "screen":
    MALSET = (0, 1, 4)              # akiec, bcc, mel = malin/premalin
    y_all = np.isin(y_all, MALSET).astype(np.int64)
    if not FAKE:
        yp2 = yp2                   # deja binaire (mel vs nevus)
    K = 2
    MEL = 1                         # classe positive = maligne
print("images:", len(y_all), "| class counts:",
      np.bincount(y_all, minlength=K).tolist(), flush=True)

# frozen lesion-grouped protocol: SGKF(5, seed 42) -> first split = test
# (20%); SGKF(5, seed 43) on the remainder = CV folds. The image order is
# frozen (labels csv sorted by image_id), so split membership is identical
# across every kernel of the campaign.
sp = None
if C.get("split_inline"):
    C["split_file"] = "inline"
    sp = C.pop("split_inline")      # pas dans le JSON de sortie
elif C.get("split_file"):
    # splits IMPOSES (ceux de la campagne Kaggle) pour l'appariement
    sp = json.load(open(C["split_file"]))
if sp is not None:
    te_i = np.array(sorted(sp["te_idx"]))
    trv_i = np.array(sorted(set(range(len(y_all))) - set(sp["te_idx"])
                            - set(sp.get("drop_idx", []))))
else:
    sg1 = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    trv_i, te_i = next(iter(sg1.split(np.zeros(len(y_all)), y_all, les)))
    trv_i, te_i = np.sort(trv_i), np.sort(te_i)
assert LEAK_CONTROL or set(les[trv_i]) & set(les[te_i]) == set()
Xtr = np.ascontiguousarray(X_all[trv_i]); ytr = y_all[trv_i]
Xte = np.ascontiguousarray(X_all[te_i]); yte = y_all[te_i]
gtr = les[trv_i]
del X_all
if C.get("split_file"):
    CV = []
    for va in sp["folds_va"]:
        va = np.array(sorted(va))
        tr = np.array(sorted(set(range(len(ytr))) - set(va)))
        CV.append((tr, va))
else:
    sg2 = StratifiedGroupKFold(5, shuffle=True, random_state=43)
    CV = [(np.sort(a), np.sort(b)) for a, b in
          sg2.split(np.zeros(len(ytr)), ytr, gtr)]
for a, b in CV:
    assert LEAK_CONTROL or set(gtr[a]) & set(gtr[b]) == set()
print(f"trainval {len(ytr)} / test {len(yte)} | "
      f"test counts {np.bincount(yte, minlength=K).tolist()}", flush=True)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

class DS(Dataset):
    def __init__(self, X, y, idx, train):
        self.X, self.y, self.idx, self.train = X, y, list(idx), train
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        im = self.X[j]
        if C["size"] != im.shape[0]:         # cache is 256px
            im = cv2.resize(np.ascontiguousarray(im),
                            (C["size"], C["size"]),
                            interpolation=cv2.INTER_AREA)
        if self.train:
            if random.random() < 0.5:
                im = im[:, ::-1]
            if random.random() < 0.5:
                im = im[::-1]
            k = random.randint(0, 3)
            if k:
                im = np.rot90(im, k)
            im = im.astype(np.float32)
            im = im * random.uniform(0.9, 1.1) + random.uniform(-10, 10)
            im = np.clip(im, 0, 255)
        x = (im.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(
            x.transpose(2, 0, 1))), int(self.y[j])

NW = int(os.environ.get("DERM_WORKERS", "2"))

def dl(ds, bs, shuffle):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=NW,
                      pin_memory=True, drop_last=False)

# ----------------------------- model -----------------------------
class AttnPool(nn.Module):
    """(Gated) attention pooling over tokens (Ilse et al., 2018)."""
    def __init__(self, c, datt=128, gated=True):
        super().__init__()
        self.V = nn.Linear(c, datt)
        self.U = nn.Linear(c, datt) if gated else None
        self.w = nn.Linear(datt, 1)
    def forward(self, f):                      # B,N,C
        h = torch.tanh(self.V(f))
        if self.U is not None:
            h = h * torch.sigmoid(self.U(f))
        a = torch.softmax(self.w(h), dim=1)    # B,N,1
        return (a * f).sum(1), a.squeeze(-1)

class MeanPool(nn.Module):
    def forward(self, f):
        return f.mean(1), None

class MERIT(nn.Module):
    """Multi-scale (gated-)attention pooling + learned scale gate + head."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg["dim"]
        self.bb = timm.create_model(cfg["backbone"],
                                    pretrained=cfg["pretrained"],
                                    features_only=True,
                                    out_indices=cfg["scales"])
        chs = self.bb.feature_info.channels()
        ns = len(chs)
        if cfg["pool"] == "mean":
            self.pools = nn.ModuleList([MeanPool() for _ in chs])
        else:
            gated = cfg["pool"] == "att_gated"
            self.pools = nn.ModuleList(
                [AttnPool(c, gated=gated) for c in chs])
        self.projs = nn.ModuleList([nn.Linear(c, d) for c in chs])
        self.gate = (nn.Sequential(nn.Linear(ns * d, d), nn.GELU(),
                                   nn.Linear(d, ns))
                     if ns > 1 and cfg["scale_fuse"] == "gate" else None)
        self.ns = ns
        self.neck = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3))
        self.head = nn.Linear(d, K)
    def forward(self, x):
        feats = self.bb(x)
        vs, atts = [], []
        for f, pool, proj, c in zip(feats, self.pools, self.projs,
                                    self.bb.feature_info.channels()):
            assert f.shape[1] == c, (f.shape, c)        # NCHW expected
            t = f.flatten(2).transpose(1, 2)            # B,N,C
            v, a = pool(t)
            vs.append(F.gelu(proj(v)))
            atts.append(a)
        V = torch.stack(vs, 1)                           # B,ns,d
        if self.gate is not None:
            beta = torch.softmax(self.gate(V.flatten(1)), -1)   # B,ns
            v = (beta.unsqueeze(-1) * V).sum(1)
        else:                                 # mean fuse / single scale
            beta = torch.full((V.shape[0], self.ns), 1.0 / self.ns,
                              device=V.device)
            v = V.mean(1)
        z = self.neck(v)
        return self.head(z), z, beta, atts

class Plain(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["dim"]
        kw = {}
        if cfg["backbone"].startswith(("deit3", "vit_", "swinv2")):
            kw["img_size"] = cfg["size"]
        self.net = timm.create_model(cfg["backbone"],
                                     pretrained=cfg["pretrained"],
                                     num_classes=0, **kw)
        self.proj = nn.Linear(self.net.num_features, d)
        self.neck = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3))
        self.head = nn.Linear(d, K)
    def forward(self, x):
        z = self.neck(F.gelu(self.proj(self.net(x))))
        return self.head(z), z, None, None

def build():
    return (MERIT(C) if C["arch"] == "merit" else Plain(C)).to(device)

# ----------------------------- losses -----------------------------
def cb_weights(counts, beta):
    """Class-balanced weights via effective number of samples (Cui 2019)."""
    counts = np.maximum(counts.astype(np.float64), 1)
    w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    w = w * K / w.sum()
    return torch.tensor(w, dtype=torch.float32, device=device)

def inv_weights(counts):
    c = np.maximum(counts.astype(np.float64), 1)
    return torch.tensor(c.sum() / (K * c), dtype=torch.float32,
                        device=device)

def kl_dirichlet_unit(alpha):
    """KL( Dir(alpha) || Dir(1) ) per sample."""
    S = alpha.sum(1)
    kl = (torch.lgamma(S) - torch.lgamma(alpha).sum(1)
          - math.lgamma(K)
          + ((alpha - 1.0) * (torch.digamma(alpha)
                              - torch.digamma(S)[:, None])).sum(1))
    return kl

def edl_loss(logits, y, w, lam):
    """Type-II MLE Bayes-risk (digamma form) + annealed KL regulariser."""
    evid = F.softplus(logits.float())
    alpha = evid + 1.0
    S = alpha.sum(1, keepdim=True)
    nll = torch.digamma(S)[:, 0] - torch.digamma(
        alpha.gather(1, y[:, None]))[:, 0]
    if w is not None:
        nll = nll * w[y]
    loss = nll.mean()
    if lam > 0:
        yoh = F.one_hot(y, K).float()
        a_t = yoh + (1.0 - yoh) * alpha
        loss = loss + lam * kl_dirichlet_unit(a_t).mean()
    return loss

def focal_loss(logits, y, gamma):
    logp = F.log_softmax(logits.float(), 1)
    logpy = logp.gather(1, y[:, None])[:, 0]
    return (-(1.0 - logpy.exp()) ** gamma * logpy).mean()

def criterion(logits, y, W, lam):
    lo = C["loss"]
    if lo == "ce":
        return F.cross_entropy(logits, y, label_smoothing=C["ls"])
    if lo == "wce":
        return F.cross_entropy(logits, y, weight=W["inv"],
                               label_smoothing=C["ls"])
    if lo == "cbce":
        return F.cross_entropy(logits, y, weight=W["cb"],
                               label_smoothing=C["ls"])
    if lo == "focal":
        return focal_loss(logits, y, C["focal_gamma"])
    if lo == "edl":
        return edl_loss(logits, y, None, lam)
    if lo == "cbedl":
        return edl_loss(logits, y, W["cb"], lam)
    raise ValueError(lo)

IS_EDL = C["loss"] in ("edl", "cbedl")
assert (C["head"] == "edl") == IS_EDL

class EMA:
    """EMA with decay warmup min(decay,(1+t)/(10+t)) — avoids shadow lag on
    short schedules."""
    def __init__(self, model, decay):
        self.decay = decay
        self.t = 0
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}
    def update(self, model):
        self.t += 1
        d = min(self.decay, (1 + self.t) / (10 + self.t))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k] = v.detach().clone()
    def copy_to(self, model):
        model.load_state_dict(self.shadow)

# ----------------------------- predict / metrics -----------------------------
@torch.no_grad()
def predict(model, X, y, idx, extras=False):
    model.eval()
    P, U, Y, Z, B = [], [], [], [], []
    for x, yy in dl(DS(X, y, idx, False), C["bs"], False):
        x = x.to(device, non_blocking=True)
        views = [x]
        if C["tta"] == "tta4":
            views += [torch.flip(x, dims=[3]), torch.flip(x, dims=[2]),
                      torch.flip(x, dims=[2, 3])]
        elif C["tta"]:
            views += [torch.flip(x, dims=[3])]
        with autocast("cuda", enabled=AMP):
            outs = [model(v) for v in views]
        l1, z, beta = outs[0][0], outs[0][1], outs[0][2]
        if IS_EDL:
            e = torch.stack([F.softplus(o[0].float())
                             for o in outs]).mean(0)
            alpha = e + 1.0
            S = alpha.sum(1, keepdim=True)
            p = (alpha / S).cpu().numpy()
            u = (K / S[:, 0]).cpu().numpy()
        else:
            p = torch.stack([F.softmax(o[0].float(), 1)
                             for o in outs]).mean(0)
            p = p.cpu().numpy()
            u = 1.0 - p.max(1)
        P.append(p); U.append(u); Y.append(np.asarray(yy))
        if extras:
            Z.append(z.float().cpu().numpy())
            if beta is not None:
                B.append(beta.float().cpu().numpy())
    out = [np.concatenate(Y), np.concatenate(P), np.concatenate(U)]
    if extras:
        out += [np.concatenate(Z), np.concatenate(B) if B else None]
    return out

def ece15(y, p):
    conf = p.max(1); pred = p.argmax(1); correct = (pred == y)
    bins = np.linspace(0, 1, 16)
    e = 0.0
    for a, b in zip(bins[:-1], bins[1:]):
        m = (conf > a) & (conf <= b)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)

def metrics(y, p):
    p = p / p.sum(1, keepdims=True)      # re-normalise (rounded ensembles)
    g = p.argmax(1)
    rec = confusion_matrix(y, g, labels=list(range(K)),
                           normalize="true").diagonal()
    return dict(
        bal_acc=float(balanced_accuracy_score(y, g)),
        acc=float(accuracy_score(y, g)),
        f1_macro=float(f1_score(y, g, average="macro")),
        auc_macro=float(roc_auc_score(y, p[:, 1]) if K == 2 else
                        roc_auc_score(y, p, multi_class="ovr",
                                      average="macro")),
        auc_mel=float(roc_auc_score((y == MEL).astype(int), p[:, MEL])),
        mel_recall=float(rec[MEL]),
        nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        ece15=ece15(y, p),
        cm=confusion_matrix(y, g, labels=list(range(K))).tolist(),
        per_class_recall=[float(r) for r in rec])

# ----------------------------- training -----------------------------
def train_fold(tr_idx, va_idx, fold):
    model = build()
    cnt = np.bincount(ytr[tr_idx], minlength=K)
    W = dict(inv=inv_weights(cnt), cb=cb_weights(cnt, C["beta_eff"]))
    tdl = dl(DS(Xtr, ytr, tr_idx, True), C["bs"], True)
    opt = torch.optim.AdamW(model.parameters(), lr=C["lr"],
                            weight_decay=C["wd"])
    spe = (len(tdl) if not C["max_epsamples"]
           else max(1, math.ceil(min(C["max_epsamples"],
                                     len(tr_idx)) / C["bs"])))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=C["lr"], epochs=C["epochs"], steps_per_epoch=spe,
        pct_start=0.2)
    scaler = GradScaler("cuda", enabled=AMP)
    ema = EMA(model, C["ema"])
    hist = []
    for ep in range(C["epochs"]):
        lam = C["lam_max"] * min(1.0, (ep + 1) / max(1, C["t_anneal"]))
        model.train()
        tot, seen, corr = 0.0, 0, 0
        nb = 0
        for bi, (x, y) in enumerate(tdl):
            if C["max_epsamples"] and bi * C["bs"] >= C["max_epsamples"]:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            mix = (C["mixup"] > 0 and not IS_EDL
                   and random.random() < 0.5)
            if mix:
                lmb = np.random.beta(C["mixup"], C["mixup"])
                perm = torch.randperm(x.size(0), device=x.device)
                x = lmb * x + (1 - lmb) * x[perm]
            with autocast("cuda", enabled=AMP):
                logits, _, _, _ = model(x)
                if mix:
                    loss = (lmb * criterion(logits, y, W, lam)
                            + (1 - lmb) * criterion(logits, y[perm],
                                                    W, lam))
                else:
                    loss = criterion(logits, y, W, lam)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            tot += loss.item(); nb += 1
            corr += int((logits.argmax(1) == y).sum()); seen += len(y)
        ema.copy_to(model)
        yv, pv, uv = predict(model, Xtr, ytr, va_idx)[:3]
        m = metrics(yv, pv)
        hist.append(dict(ep=ep + 1, loss=tot / max(1, nb),
                         train_acc=corr / max(1, seen),
                         val_bal_acc=m["bal_acc"], val_acc=m["acc"],
                         val_mel_recall=m["mel_recall"], lam=lam))
        print(f"[f{fold} e{ep + 1}/{C['epochs']}] loss={hist[-1]['loss']:.3f}"
              f" tracc={hist[-1]['train_acc']:.3f}"
              f" val_bal={m['bal_acc']:.4f} val_acc={m['acc']:.4f}",
              flush=True)
    yv, pv, uv = predict(model, Xtr, ytr, va_idx)[:3]
    vm = metrics(yv, pv)
    ex = C["save_extras"] and fold == 1
    pr = predict(model, Xte, yte, np.arange(len(yte)), extras=ex)
    yt, pt, ut = pr[:3]
    tm = metrics(yt, pt)
    if ex:
        np.savez_compressed(
            OUT_ROOT + f"/extras_{C['tag']}.npz", z=pr[3],
            beta=(pr[4] if pr[4] is not None else np.zeros(1)),
            y=yt, p=pt, u=ut)
    # PH2 zero-shot
    yp, pp, up = predict(model, Xp2, yp2, np.arange(len(yp2)))[:3]
    ph2m = dict(auc_mel=float(roc_auc_score(yp2, pp[:, MEL])),
                mean_u=float(up.mean()))
    print(f"[fold {fold}] TEST bal={tm['bal_acc']:.4f} acc={tm['acc']:.4f} "
          f"f1={tm['f1_macro']:.4f} aucM={tm['auc_macro']:.4f} "
          f"melR={tm['mel_recall']:.3f} | PH2 auc={ph2m['auc_mel']:.4f}",
          flush=True)
    if C["save_ckpt"]:
        torch.save(model.state_dict(), OUT_ROOT + f"/ckpt_{C['tag']}_f{fold}.pt")
    out = dict(val=vm, test=tm, ph2=ph2m)
    arrays = dict(va_idx=[int(i) for i in va_idx],
                  val_p=np.round(pv, 5).tolist(),
                  val_u=np.round(uv, 5).tolist(),
                  test_p=np.round(pt, 5).tolist(),
                  test_u=np.round(ut, 5).tolist(),
                  ph2_p=np.round(pp, 5).tolist(),
                  ph2_u=np.round(up, 5).tolist())
    del model
    torch.cuda.empty_cache()
    return out, hist, arrays

folds_out, hists, arr_folds = [], [], []
for fi, (tr_idx, va_idx) in enumerate(CV[:C["folds_run"]], 1):
    o, h, a = train_fold(tr_idx, va_idx, fi)
    folds_out.append(o); hists.append(h); arr_folds.append(a)

def agg():
    o = {}
    for mk in ("bal_acc", "acc", "f1_macro", "auc_macro", "auc_mel",
               "mel_recall", "nll", "ece15"):
        v = np.array([f["test"][mk] for f in folds_out])
        o[mk] = dict(mean=float(v.mean()),
                     sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    o["ph2_auc_mel"] = dict(
        mean=float(np.mean([f["ph2"]["auc_mel"] for f in folds_out])),
        sd=float(np.std([f["ph2"]["auc_mel"] for f in folds_out], ddof=1))
        if len(folds_out) > 1 else 0.0)
    return o

p_ens = np.mean([np.array(a["test_p"]) for a in arr_folds], 0)
ens = metrics(yte, p_ens)
pp_ens = np.mean([np.array(a["ph2_p"]) for a in arr_folds], 0)
ens["ph2_auc_mel"] = float(roc_auc_score(yp2, pp_ens[:, MEL]))

out = dict(config={k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in C.items()},
           classes=CLASSES,
           n_trainval=int(len(ytr)), n_test=int(len(yte)),
           class_counts_all=np.bincount(
               np.concatenate([ytr, yte]), minlength=K).tolist(),
           class_counts_test=np.bincount(yte, minlength=K).tolist(),
           trv_idx=[int(i) for i in trv_i], te_idx=[int(i) for i in te_i],
           folds=folds_out, aggregate=agg(), ensemble=ens,
           hist=hists, test_y=yte.tolist(), ph2_y=yp2.tolist(),
           arrays=arr_folds,
           minutes=round((time.time() - t00) / 60, 1))
with open(OUT_ROOT + f"/derm_{C['tag']}.json", "w") as f:
    json.dump(out, f)
A = out["aggregate"]
print(f"[DONE {C['tag']}] bal {A['bal_acc']['mean']:.4f}±"
      f"{A['bal_acc']['sd']:.4f} acc {A['acc']['mean']:.4f} "
      f"f1 {A['f1_macro']['mean']:.4f} aucM {A['auc_macro']['mean']:.4f} "
      f"melR {A['mel_recall']['mean']:.3f} | ens bal {ens['bal_acc']:.4f} "
      f"| PH2 {A['ph2_auc_mel']['mean']:.4f} | {out['minutes']} min",
      flush=True)
