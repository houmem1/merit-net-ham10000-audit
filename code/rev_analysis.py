# -*- coding: utf-8 -*-
"""Revision analysis (BMC Medical Imaging, round 1). Everything is recomputed
from the stored per-fold predictions (results/<tag>/derm_<tag>.json):

  * fold-matched ensembles (3 vs 3, 5 vs 5) for every comparison;
  * lesion-CLUSTER bootstrap CIs (resampling lesions, not images);
  * paired differences of balanced accuracy / accuracy with cluster CIs,
    exact McNemar with explicit direction, Holm adjustment;
  * post-hoc recalibration controls for the evidential outputs;
  * referral counts, validation-locked referral threshold;
  * melanoma-vs-nevus operating point audit (members, weights, thresholds,
    confusion counts, cluster CIs, PH2 at the locked threshold);
  * PH2 AUC CIs; duplicate-pair sensitivity analysis; image-level control.

-> results/revision/rev_analysis.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
REV = os.path.join(RES, "revision")
K, MEL, NV = 7, 4, 5
NBOOT = 2000
RNG = np.random.RandomState(20261125)

def load(t):
    p = os.path.join(RES, t, f"derm_{t}.json")
    return json.load(open(p)) if os.path.exists(p) else None

CAMPAIGN = ["main", "merit-ce", "cnx-cbedl", "cnx-ce", "swinv2t-ce",
            "deit3s-ce", "effb3-ce", "dense121-ce", "r50-ce", "cnx-wce",
            "cnx-focal", "cnx-cbce", "cnx-edl", "m-gap", "m-attplain",
            "m-single", "m-fusemean", "m-lam0", "m-lam01", "m-beta9999",
            "m-edl", "m-effb3", "m-224"]
BOOST = ["boost", "boost2", "boost3", "boost4", "boost5", "screen1",
         "screen2"]
NEW = ["m-edl-lam01", "cnx-cbedl-lam01", "m-lam01-s43", "main-s43",
       "merit-ce-s43"]
D = {t: load(t) for t in CAMPAIGN + BOOST + NEW}
D = {t: d for t, d in D.items() if d is not None}
print("loaded:", len(D), "| new:", [t for t in NEW if t in D])

lab = pd.read_csv(os.path.join(REV, "split_assignments.csv"))
ref = D["main"]
te_idx = np.array(ref["te_idx"])
trv_idx = np.array(ref["trv_idx"])
y = np.array(ref["test_y"])
yp2 = np.array(ref["ph2_y"])
N = len(y)
for t, d in D.items():
    if d["config"].get("task", "multi") == "multi":
        assert d["te_idx"] == ref["te_idx"] and d["test_y"] == ref["test_y"], t
g_te = lab.lesion_id.values[te_idx]
ytr_full = lab.y.values[trv_idx]
g_tr = lab.lesion_id.values[trv_idx]
assert (lab.y.values[te_idx] == y).all()

# ------------------------------------------------------------ helpers
def nfolds(t):
    return len(D[t]["arrays"])

def ens_p(t, nf=None, key="test_p"):
    arr = D[t]["arrays"][:nf] if nf else D[t]["arrays"]
    P = np.mean([np.array(a[key]) for a in arr], 0)
    return P / P.sum(1, keepdims=True)

def ens_u(t, nf=None, key="test_u"):
    arr = D[t]["arrays"][:nf] if nf else D[t]["arrays"]
    return np.mean([np.array(a[key]) for a in arr], 0)

def cluster_boot(groups, n=NBOOT, rng=RNG):
    """List of image-index arrays; each replicate resamples LESIONS."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    out = []
    for _ in range(n):
        pick = rng.randint(0, len(uniq), len(uniq))
        out.append(np.concatenate([members[i] for i in pick]))
    return out

BOOT = cluster_boot(g_te)

def bal_acc(yy, pred, k=K):
    cnt = np.bincount(yy, minlength=k).astype(float)
    hit = np.bincount(yy, weights=(pred == yy).astype(float), minlength=k)
    m = cnt > 0                      # classes absent from yy are skipped
    return float((hit[m] / cnt[m]).mean())

def ci(vals):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def boot_stat(fn, boot=None):
    return ci([fn(idx) for idx in (BOOT if boot is None else boot)])

def ece15(yy, p, M=15):
    conf = p.max(1); pred = p.argmax(1); corr = (pred == yy)
    bins = np.linspace(0, 1, M + 1)
    e = 0.0
    for a, b in zip(bins[:-1], bins[1:]):
        m = (conf > a) & (conf <= b)
        if m.sum():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)

def nll(yy, p):
    return float(-np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, 1)).mean())

def aurc(yy, pred, u):
    order = np.argsort(u, kind="stable")
    err = (pred[order] != yy[order]).astype(float)
    cum = np.cumsum(err) / np.arange(1, len(err) + 1)
    cov = np.arange(1, len(err) + 1) / len(err)
    return float(np.trapezoid(cum, cov))

def mcnemar(ca, cb):
    """ca, cb = boolean correctness of A and B. Returns counts with an
    explicit direction and the exact two-sided p."""
    a_only = int((ca & ~cb).sum()); b_only = int((~ca & cb).sum())
    n = a_only + b_only
    p = binomtest(min(a_only, b_only), n, 0.5).pvalue if n else 1.0
    return dict(a_only=a_only, b_only=b_only, p=float(min(1.0, p)))

def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()

def summary(t, nf=None, mask=None):
    """Ensemble metrics (+ cluster CIs) for tag t over its first nf folds."""
    p = ens_p(t, nf); pred = p.argmax(1)
    m = np.ones(N, bool) if mask is None else mask
    yy, pp, pr = y[m], p[m], pred[m]
    rec = [(float(((pr == k) & (yy == k)).sum() / max(1, (yy == k).sum())))
           for k in range(K)]
    folds = D[t]["folds"][:nf] if nf else D[t]["folds"]
    fb = np.array([f["test"]["bal_acc"] for f in folds])
    out = dict(
        nf=len(folds),
        fold_bal_mean=float(fb.mean()),
        fold_bal_sd=float(fb.std(ddof=1)) if len(fb) > 1 else 0.0,
        bal_acc=bal_acc(yy, pr), acc=float((pr == yy).mean()),
        f1_macro=float(f1_score(yy, pr, average="macro")),
        auc_macro=float(roc_auc_score(yy, pp, multi_class="ovr",
                                      average="macro")),
        auc_mel=float(roc_auc_score((yy == MEL).astype(int), pp[:, MEL])),
        mel_recall=rec[MEL], per_class_recall=rec,
        nll=nll(yy, pp), ece15=ece15(yy, pp))
    mm = np.isin(yy, (MEL, NV))
    out["auc_melnv"] = float(roc_auc_score(
        (yy[mm] == MEL).astype(int),
        pp[mm, MEL] / (pp[mm, MEL] + pp[mm, NV] + 1e-12)))
    if mask is None:
        out["bal_acc_ci"] = boot_stat(lambda i: bal_acc(y[i], pred[i]))
        out["acc_ci"] = boot_stat(lambda i: float((pred[i] == y[i]).mean()))
        out["mel_recall_ci"] = boot_stat(
            lambda i: float((pred[i][y[i] == MEL] == MEL).mean()))
        P2 = ens_p(t, nf, "ph2_p")
        s2 = P2[:, MEL]
        out["ph2_auc"] = float(roc_auc_score(yp2, s2))
        rr = np.random.RandomState(7)
        v = []
        for _ in range(NBOOT):
            i = rr.randint(0, len(yp2), len(yp2))
            if 0 < yp2[i].sum() < len(i):
                v.append(roc_auc_score(yp2[i], s2[i]))
        out["ph2_auc_ci"] = ci(v)
    return out

def contrast(a, b, nf, label=None):
    """Paired contrast A - B on fold-matched ensembles."""
    pa, pb = ens_p(a, nf).argmax(1), ens_p(b, nf).argmax(1)
    ca, cb = pa == y, pb == y
    dbal = bal_acc(y, pa) - bal_acc(y, pb)
    dacc = float(ca.mean() - cb.mean())
    dbal_b = [bal_acc(y[i], pa[i]) - bal_acc(y[i], pb[i]) for i in BOOT]
    dacc_b = [float(ca[i].mean() - cb[i].mean()) for i in BOOT]
    mc = mcnemar(ca, cb)
    return dict(a=a, b=b, nf=nf, label=label or f"{a} - {b}",
                d_bal=float(dbal), d_bal_ci=ci(dbal_b),
                p_boot_bal=float(2 * min(np.mean(np.array(dbal_b) <= 0),
                                         np.mean(np.array(dbal_b) >= 0))),
                d_acc=dacc, d_acc_ci=ci(dacc_b), **mc)

OUT = {"_n_test": int(N), "_n_test_lesions": int(len(set(g_te))),
       "_nboot": NBOOT}

# ============================================== 1. matched summaries
S3, S5 = {}, {}
for t in CAMPAIGN + [n for n in NEW if n in D]:
    S3[t] = summary(t, 3)
    if nfolds(t) == 5:
        S5[t] = summary(t, 5)
OUT["summary3"], OUT["summary5"] = S3, S5
for t in CAMPAIGN:
    s = S3[t]
    print(f"3f {t:<12} bal {100*s['bal_acc']:.1f} "
          f"[{100*s['bal_acc_ci'][0]:.1f},{100*s['bal_acc_ci'][1]:.1f}] "
          f"acc {100*s['acc']:.1f} melR {100*s['mel_recall']:.1f} "
          f"PH2 {s['ph2_auc']:.3f}")
for t in S5:
    s = S5[t]
    print(f"5f {t:<12} bal {100*s['bal_acc']:.1f} "
          f"[{100*s['bal_acc_ci'][0]:.1f},{100*s['bal_acc_ci'][1]:.1f}] "
          f"acc {100*s['acc']:.1f} [{100*s['acc_ci'][0]:.1f},"
          f"{100*s['acc_ci'][1]:.1f}]")

# ============================================== 2. paired contrasts
PRIMARY = [
    ("merit-ce", "cnx-ce", 5, "Attention effect under CE: MSAP+CE - ConvNeXt-T+CE"),
    ("main", "cnx-cbedl", 5, "Attention effect under CB-EDL: MERIT-Net - ConvNeXt-T+CB-EDL"),
    ("main", "merit-ce", 5, "Objective effect, MSAP encoder: CB-EDL - CE"),
    ("cnx-cbedl", "cnx-ce", 5, "Objective effect, plain encoder: CB-EDL - CE"),
    ("m-lam01", "main", 3, "KL ceiling: lambda=0.1 - lambda=1 (MERIT)"),
    ("m-lam0", "main", 3, "KL ceiling: lambda=0 - lambda=1 (MERIT)"),
    ("m-lam01", "merit-ce", 3, "Remaining gap: MERIT lambda=0.1 - MSAP+CE"),
]
prim = [contrast(a, b, nf, lab_) for a, b, nf, lab_ in PRIMARY]
for c, h in zip(prim, holm([c["p"] for c in prim])):
    c["p_holm"] = h
OUT["contrasts_primary"] = prim

SECOND = []
for t in ("merit-ce", "cnx-ce", "swinv2t-ce", "deit3s-ce", "effb3-ce",
          "dense121-ce"):
    SECOND.append((t, "r50-ce", 3, f"{t} - ResNet-50"))
for t in ("swinv2t-ce", "deit3s-ce", "effb3-ce", "dense121-ce", "r50-ce",
          "merit-ce"):
    SECOND.append((t, "cnx-ce", 3, f"{t} - ConvNeXt-T+CE"))
for t in ("cnx-wce", "cnx-focal", "cnx-cbce", "cnx-edl", "cnx-cbedl"):
    SECOND.append((t, "cnx-ce", 3, f"{t} - ConvNeXt-T+CE (objective)"))
for t in ("m-gap", "m-attplain", "m-single", "m-fusemean", "m-beta9999",
          "m-edl", "m-effb3", "m-224"):
    SECOND.append((t, "main", 3, f"{t} - MERIT-Net (ablation)"))
SECOND += [("main", "cnx-cbce", 3, "MERIT-Net - ConvNeXt-T+CB-CE"),
           ("main", "cnx-wce", 3, "MERIT-Net - ConvNeXt-T+weighted CE"),
           ("main", "cnx-ce", 3, "MERIT-Net - ConvNeXt-T+CE"),
           ("main", "r50-ce", 3, "MERIT-Net - ResNet-50"),
           ("main", "merit-ce", 3, "MERIT-Net - MSAP+CE (3 folds)"),
           ("merit-ce", "cnx-ce", 3, "MSAP+CE - ConvNeXt-T+CE (3 folds)"),
           ("main", "cnx-cbedl", 3, "MERIT-Net - ConvNeXt-T+CB-EDL (3 folds)"),
           ("m-lam0", "merit-ce", 3, "MERIT lambda=0 - MSAP+CE"),
           ("m-lam01", "cnx-ce", 3, "MERIT lambda=0.1 - ConvNeXt-T+CE")]
if "cnx-cbedl-lam01" in D:
    SECOND += [("cnx-cbedl-lam01", "cnx-cbedl", 3,
                "ConvNeXt-T CB-EDL: lambda=0.1 - lambda=1"),
               ("cnx-cbedl-lam01", "cnx-ce", 3,
                "ConvNeXt-T CB-EDL lambda=0.1 - ConvNeXt-T+CE")]
if "m-edl-lam01" in D:
    SECOND += [("m-edl-lam01", "m-edl", 3,
                "Unweighted EDL: lambda=0.1 - lambda=1 (MERIT)"),
               ("m-lam01", "m-edl-lam01", 3,
                "Class weights at lambda=0.1: CB-EDL - EDL (MERIT)")]
if "main-s43" in D and "m-lam01-s43" in D:
    SECOND += [("m-lam01-s43", "main-s43", 3,
                "Seed 43: lambda=0.1 - lambda=1 (MERIT)")]
if "merit-ce-s43" in D and "m-lam01-s43" in D:
    SECOND += [("m-lam01-s43", "merit-ce-s43", 3,
                "Seed 43: MERIT lambda=0.1 - MSAP+CE"),
               ("main-s43", "merit-ce-s43", 3,
                "Seed 43: MERIT-Net - MSAP+CE")]
sec = [contrast(a, b, nf, lab_) for a, b, nf, lab_ in SECOND]
for c, h in zip(sec, holm([c["p"] for c in sec])):
    c["p_holm"] = h
OUT["contrasts_secondary"] = sec
for c in prim + sec:
    print(f"{c['label']:<58} nf{c['nf']} dBal {100*c['d_bal']:+.1f} "
          f"[{100*c['d_bal_ci'][0]:+.1f},{100*c['d_bal_ci'][1]:+.1f}] "
          f"dAcc {100*c['d_acc']:+.1f} A-only {c['a_only']} "
          f"B-only {c['b_only']} p {c['p']:.2g} holm {c['p_holm']:.2g}")

# interaction (class weights x KL) on the MERIT encoder, 3 folds
if "m-edl-lam01" in D:
    P = {t: ens_p(t, 3).argmax(1) for t in ("main", "m-lam01", "m-edl",
                                            "m-edl-lam01")}
    def inter(i):
        b = {t: bal_acc(y[i], P[t][i]) for t in P}
        return (b["m-lam01"] - b["main"]) - (b["m-edl-lam01"] - b["m-edl"])
    OUT["interaction_weights_kl"] = dict(
        est=float(inter(np.arange(N))), ci=boot_stat(inter))
    print("interaction (CB gain of lam0.1) - (UW gain):",
          OUT["interaction_weights_kl"])

# seed replication (descriptive, outside the Holm families): same
# configuration at seed 43 minus seed 42, and two-seed pooled contrasts
if all(t in D for t in ("main-s43", "m-lam01-s43", "merit-ce-s43")):
    PS = {t: ens_p(t, 3).argmax(1) for t in (
        "main", "m-lam01", "merit-ce", "main-s43", "m-lam01-s43",
        "merit-ce-s43")}
    def _d(fn):
        return dict(est=float(fn(np.arange(N))), ci=boot_stat(fn))
    def _bal(t):
        return lambda i: bal_acc(y[i], PS[t][i])
    def _acc(t):
        return lambda i: float((PS[t][i] == y[i]).mean())
    SR = {"seed_diff": {}, "pooled": {}}
    for t in ("main", "m-lam01", "merit-ce"):
        SR["seed_diff"][t] = dict(
            bal=_d(lambda i, t=t: _bal(t + "-s43")(i) - _bal(t)(i)),
            acc=_d(lambda i, t=t: _acc(t + "-s43")(i) - _acc(t)(i)))
    for cid, a, b in (("kl01", "m-lam01", "main"),
                      ("obj", "main", "merit-ce"),
                      ("gap01", "m-lam01", "merit-ce")):
        def pb(i, a=a, b=b, f=_bal):
            return 0.5 * ((f(a)(i) - f(b)(i))
                          + (f(a + "-s43")(i) - f(b + "-s43")(i)))
        def pa(i, a=a, b=b):
            return pb(i, a, b, _acc)
        SR["pooled"][cid] = dict(a=a, b=b, bal=_d(pb), acc=_d(pa))
    OUT["seed_replication"] = SR
    print("seed replication:", json.dumps(SR))

# margins over ResNet-50 (3-fold ensembles and fold means)
OUT["margins_vs_r50"] = {
    t: dict(ens=S3[t]["bal_acc"] - S3["r50-ce"]["bal_acc"],
            folds=S3[t]["fold_bal_mean"] - S3["r50-ce"]["fold_bal_mean"])
    for t in ("merit-ce", "cnx-ce", "swinv2t-ce", "deit3s-ce", "effb3-ce",
              "dense121-ce")}

# ============================================== 3. calibration controls
def alpha_of(a, key_p, key_u):
    p = np.array(a[key_p]); p = p / p.sum(1, keepdims=True)
    u = np.array(a[key_u])
    S = K / np.clip(u, 1e-6, None)
    return p * S[:, None]

def fit_T(logp, yy):
    def f(lt):
        z = logp / np.exp(lt)
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(yy)), yy].mean()
    r = minimize_scalar(f, bounds=(-4, 4), method="bounded")
    return float(np.exp(r.x))

def apply_T(logp, T):
    z = logp / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def fit_prior(alpha, yy):
    """p_k = (e_k + c) / (sum e + K c); c=1 is the training prior."""
    e = np.clip(alpha - 1.0, 0, None)
    def f(lc):
        c = np.exp(lc)
        p = (e + c) / (e.sum(1, keepdims=True) + K * c)
        return -np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, 1)).mean()
    r = minimize_scalar(f, bounds=(-12, 3), method="bounded")
    return float(np.exp(r.x))

def calib_block(t, nf, edl):
    """Per-fold post-hoc recalibration fitted on that fold's VALIDATION
    predictions, applied to its test predictions, then fold-averaged."""
    arrs = D[t]["arrays"][:nf]
    raw = ens_p(t, nf); pred = raw.argmax(1)
    res = dict(nf=nf)
    def block(P, name, extra=None):
        pr = P.argmax(1)
        res[name] = dict(
            acc=float((pr == y).mean()), bal_acc=bal_acc(y, pr),
            ece15=ece15(y, P), nll=nll(y, P),
            ece15_ci=boot_stat(lambda i: ece15(y[i], P[i])),
            aurc_conf=aurc(y, pr, 1 - P.max(1)),
            aurc_conf_ci=boot_stat(
                lambda i: aurc(y[i], pr[i], 1 - P[i].max(1))))
        if extra:
            res[name].update(extra)
    block(raw, "raw")
    # temperature scaling on log-probabilities
    Ts, Pt = [], []
    for a in arrs:
        pv = np.array(a["val_p"]); pv = pv / pv.sum(1, keepdims=True)
        yv = ytr_full[np.array(a["va_idx"])]
        T = fit_T(np.log(np.clip(pv, 1e-12, 1)), yv); Ts.append(T)
        pt = np.array(a["test_p"]); pt = pt / pt.sum(1, keepdims=True)
        Pt.append(apply_T(np.log(np.clip(pt, 1e-12, 1)), T))
    block(np.mean(Pt, 0), "temp", dict(T=[float(x) for x in Ts]))
    if edl:
        cs, Pc = [], []
        for a in arrs:
            av = alpha_of(a, "val_p", "val_u")
            yv = ytr_full[np.array(a["va_idx"])]
            c = fit_prior(av, yv); cs.append(c)
            at = alpha_of(a, "test_p", "test_u")
            e = np.clip(at - 1.0, 0, None)
            Pc.append((e + c) / (e.sum(1, keepdims=True) + K * c))
        block(np.mean(Pc, 0), "prior", dict(c=[float(x) for x in cs]))
        u = ens_u(t, nf)
        res["aurc_u"] = aurc(y, pred, u)
        res["aurc_u_ci"] = boot_stat(lambda i: aurc(y[i], pred[i], u[i]))
        res["mean_u_correct"] = float(u[pred == y].mean())
        res["mean_u_error"] = float(u[pred != y].mean())
        res["auroc_u_error"] = float(roc_auc_score(pred != y, u))
    res["auroc_conf_error"] = float(
        roc_auc_score(pred != y, 1 - raw.max(1)))
    return res

CAL = {}
for t, nf, edl in (("main", 5, True), ("cnx-cbedl", 5, True),
                   ("merit-ce", 5, False), ("cnx-ce", 5, False),
                   ("cnx-edl", 3, True), ("m-lam01", 3, True),
                   ("m-lam0", 3, True), ("m-edl", 3, True),
                   ("main", 3, True), ("merit-ce", 3, False)):
    CAL[f"{t}@{nf}"] = calib_block(t, nf, edl)
    r = CAL[f"{t}@{nf}"]
    line = (f"cal {t}@{nf}: raw ECE {r['raw']['ece15']:.3f} NLL "
            f"{r['raw']['nll']:.3f} AURCc {r['raw']['aurc_conf']:.4f} | "
            f"T ECE {r['temp']['ece15']:.3f} NLL {r['temp']['nll']:.3f} "
            f"AURCc {r['temp']['aurc_conf']:.4f} T={np.mean(r['temp']['T']):.2f}")
    if edl:
        line += (f" | prior ECE {r['prior']['ece15']:.3f} NLL "
                 f"{r['prior']['nll']:.3f} AURCc {r['prior']['aurc_conf']:.4f}"
                 f" acc {100*r['prior']['acc']:.1f} c={np.mean(r['prior']['c']):.3g}"
                 f" | AURC_u {r['aurc_u']:.4f}")
    print(line)
OUT["calibration"] = CAL

def rel_points(P, M=15):
    conf = P.max(1); corr = (P.argmax(1) == y)
    qs = np.linspace(0, 1, M + 1)
    xs, ys = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 5:
            xs.append(float(conf[m].mean())); ys.append(float(corr[m].mean()))
    return dict(conf=xs, acc=ys)

def recal_prior_probs(t, nf):
    Pc = []
    for a in D[t]["arrays"][:nf]:
        av = alpha_of(a, "val_p", "val_u")
        yv = ytr_full[np.array(a["va_idx"])]
        c = fit_prior(av, yv)
        at = alpha_of(a, "test_p", "test_u")
        e = np.clip(at - 1.0, 0, None)
        Pc.append((e + c) / (e.sum(1, keepdims=True) + K * c))
    return np.mean(Pc, 0)

# paired AURC differences after recalibration (EDL best vs CE), 5 folds
def recal_T_probs(t, nf):
    Pt = []
    for a in D[t]["arrays"][:nf]:
        pv = np.array(a["val_p"]); pv = pv / pv.sum(1, keepdims=True)
        yv = ytr_full[np.array(a["va_idx"])]
        T = fit_T(np.log(np.clip(pv, 1e-12, 1)), yv)
        pt = np.array(a["test_p"]); pt = pt / pt.sum(1, keepdims=True)
        Pt.append(apply_T(np.log(np.clip(pt, 1e-12, 1)), T))
    return np.mean(Pt, 0)
Pm, Pc_ = recal_T_probs("main", 5), ens_p("merit-ce", 5)
prm, prc = Pm.argmax(1), Pc_.argmax(1)
um = ens_u("main", 5)
OUT["aurc_paired"] = dict(
    main_Tconf_minus_meritce_conf=dict(
        est=aurc(y, prm, 1 - Pm.max(1)) - aurc(y, prc, 1 - Pc_.max(1)),
        ci=boot_stat(lambda i: aurc(y[i], prm[i], 1 - Pm[i].max(1))
                     - aurc(y[i], prc[i], 1 - Pc_[i].max(1)))),
    main_u_minus_meritce_conf=dict(
        est=aurc(y, prm, um) - aurc(y, prc, 1 - Pc_.max(1)),
        ci=boot_stat(lambda i: aurc(y[i], prm[i], um[i])
                     - aurc(y[i], prc[i], 1 - Pc_[i].max(1)))),
    main_u_minus_main_conf=dict(
        est=aurc(y, prm, um) - aurc(y, prm, 1 - ens_p("main", 5).max(1)),
        ci=boot_stat(lambda i: aurc(y[i], prm[i], um[i])
                     - aurc(y[i], prm[i], 1 - ens_p("main", 5)[i].max(1)))))
print("AURC paired:", json.dumps(OUT["aurc_paired"]))
OUT["reliability"] = dict(
    main_raw=rel_points(ens_p("main", 5)), main_temp=rel_points(Pm),
    main_prior=rel_points(recal_prior_probs("main", 5)),
    meritce_raw=rel_points(Pc_))

# ============================================== 4. referral audit
def val_u_pool(t, key="val_u"):
    v = np.full(len(ytr_full), np.nan)
    for a in D[t]["arrays"]:
        v[np.array(a["va_idx"])] = np.array(a[key])
    return v

def referral(t, score, tau):
    p = ens_p(t, 5); pred = p.argmax(1)
    keep = score <= tau
    mel = y == MEL
    r = dict(tau=float(tau), n_kept=int(keep.sum()),
             coverage=float(keep.mean()),
             acc_kept=float((pred[keep] == y[keep]).mean()),
             bal_kept=bal_acc(y[keep], pred[keep]),
             errors_kept=int((pred[keep] != y[keep]).sum()),
             mel_total=int(mel.sum()),
             mel_referred=int((mel & ~keep).sum()),
             mel_kept_tp=int((mel & keep & (pred == MEL)).sum()),
             mel_kept_fn=int((mel & keep & (pred != MEL)).sum()),
             classes_present_kept=int(len(set(y[keep]))))
    return r, keep, pred

REFR = {"levels": {}}
u_main = ens_u("main", 5)
p_ce = ens_p("merit-ce", 5); c_ce = 1 - p_ce.max(1)
p_cx = ens_p("cnx-ce", 5); c_cx = 1 - p_cx.max(1)
p_mn = ens_p("main", 5); c_mn = 1 - p_mn.max(1)
for cov in (0.95, 0.90, 0.80, 0.70, 0.50):
    row = {}
    for name, t, sc in (("merit_u", "main", u_main),
                        ("merit_conf", "main", c_mn),
                        ("msapce_conf", "merit-ce", c_ce),
                        ("cnxce_conf", "cnx-ce", c_cx)):
        row[name] = referral(t, sc, np.quantile(sc, cov))[0]
    REFR["levels"][str(int(cov * 100))] = row
# paired differences at 80 % coverage (test-indexed cut-offs, as in Table)
def fn_kept(i, t, sc, cov):
    p = ens_p(t, 5)[i]; pred = p.argmax(1); s = sc[i]; yy = y[i]
    keep = s <= np.quantile(s, cov)
    return (int(((yy == MEL) & keep & (pred != MEL)).sum()),
            float((pred[keep] == yy[keep]).mean()))
d_fn, d_acc = [], []
for i in BOOT:
    a = fn_kept(i, "main", u_main, 0.8); b = fn_kept(i, "merit-ce", c_ce, 0.8)
    d_fn.append(a[0] - b[0]); d_acc.append(a[1] - b[1])
REFR["diff80_meritU_minus_msapceConf"] = dict(
    mel_fn_kept=dict(
        est=REFR["levels"]["80"]["merit_u"]["mel_kept_fn"]
        - REFR["levels"]["80"]["msapce_conf"]["mel_kept_fn"], ci=ci(d_fn)),
    acc_kept=dict(
        est=REFR["levels"]["80"]["merit_u"]["acc_kept"]
        - REFR["levels"]["80"]["msapce_conf"]["acc_kept"], ci=ci(d_acc)))
# validation-locked thresholds (80 % and 90 % coverage on pooled OOF val)
vu = val_u_pool("main"); vu = vu[~np.isnan(vu)]
REFR["val_locked"] = {}
for cov in (0.9, 0.8):
    tau = float(np.quantile(vu, cov))
    r, keep, pred = referral("main", u_main, tau)
    r["val_target_coverage"] = cov
    REFR["val_locked"][f"merit_u_{int(cov*100)}"] = r
    vc = []
    for a in D["merit-ce"]["arrays"]:
        pv = np.array(a["val_p"]); pv = pv / pv.sum(1, keepdims=True)
        vc.append(1 - pv.max(1))
    tau_c = float(np.quantile(np.concatenate(vc), cov))
    r2 = referral("merit-ce", c_ce, tau_c)[0]
    r2["val_target_coverage"] = cov
    REFR["val_locked"][f"msapce_conf_{int(cov*100)}"] = r2
OUT["referral"] = REFR
for lv, row in REFR["levels"].items():
    for nm, r in row.items():
        print(f"ref {lv}% {nm:<12} kept {r['n_kept']} acc {100*r['acc_kept']:.1f} "
              f"mel: referred {r['mel_referred']} TP {r['mel_kept_tp']} "
              f"FN {r['mel_kept_fn']} / {r['mel_total']}")
print("ref diff80:", json.dumps(REFR["diff80_meritU_minus_msapceConf"]))
print("val-locked:", json.dumps(REFR["val_locked"]))

# ============================================== 5. melanoma vs nevus
POOL = ["boost", "boost2", "boost3", "boost4", "boost5", "merit-ce",
        "cnx-ce"]
mask = np.isin(y, (MEL, NV))
ym = (y[mask] == MEL).astype(int)
gm = g_te[mask]
BOOTM = cluster_boot(gm, rng=np.random.RandomState(11))
Vm = np.isin(ytr_full, (MEL, NV))

def tscore(t, nf=None):
    P = ens_p(t, nf)
    return (P[:, MEL] / (P[:, MEL] + P[:, NV] + 1e-12))[mask]

def p2score(t, nf=None):
    P = ens_p(t, nf, "ph2_p")
    return P[:, MEL] / (P[:, MEL] + P[:, NV] + 1e-12)

def vvec(t):
    v = np.full(len(ytr_full), np.nan)
    for a in D[t]["arrays"]:
        P = np.array(a["val_p"]); P = P / P.sum(1, keepdims=True)
        v[np.array(a["va_idx"])] = P[:, MEL] / (P[:, MEL] + P[:, NV] + 1e-12)
    return v

def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan")] * 2
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]

def melnv_system(name, tags, w=None, nf=None):
    w = np.ones(len(tags)) / len(tags) if w is None else np.asarray(w)
    st = sum(wi * tscore(t, nf) for wi, t in zip(w, tags))
    V = np.stack([vvec(t) for t in tags])
    cm = Vm & ~np.isnan(V).any(0)
    sv = (w[:, None] * V[:, cm]).sum(0)
    yv = (ytr_full[cm] == MEL).astype(int)
    thrs = np.sort(np.unique(sv))
    pts = []
    for tgt in (0.90, 0.905, 0.91, 0.915, 0.92, 0.93):
        cand = [t for t in thrs if (sv[yv == 0] < t).mean() >= tgt]
        if cand:
            pts.append((f"valSpec{tgt*100:.1f}", float(cand[0])))
    bals = [((sv[yv == 1] >= t).mean() + (sv[yv == 0] < t).mean()) / 2
            for t in thrs]
    pts.append(("valYouden", float(thrs[int(np.argmax(bals))])))
    out = dict(name=name, tags=tags, weights=[float(x) for x in w],
               n_val=int(cm.sum()), val_auc=float(roc_auc_score(yv, sv)),
               auc=float(roc_auc_score(ym, st)),
               auc_ci=ci([roc_auc_score(ym[i], st[i]) for i in BOOTM]),
               points=[])
    s2 = sum(wi * p2score(t, nf) for wi, t in zip(w, tags))
    out["ph2_auc"] = float(roc_auc_score(yp2, s2))
    for pn, thr in pts:
        pred = (st >= thr).astype(int)
        tp = int(((pred == 1) & (ym == 1)).sum()); fn = int(((pred == 0) & (ym == 1)).sum())
        tn = int(((pred == 0) & (ym == 0)).sum()); fp = int(((pred == 1) & (ym == 0)).sum())
        def at(i, f):
            pr = pred[i]; yy = ym[i]
            se = pr[yy == 1].mean(); sp = 1 - pr[yy == 0].mean()
            return dict(sens=se, spec=sp, acc=(pr == yy).mean(),
                        bal=(se + sp) / 2)[f]
        row = dict(point=pn, thr=thr, tp=tp, fn=fn, tn=tn, fp=fp,
                   sens=tp / (tp + fn), spec=tn / (tn + fp),
                   acc=(tp + tn) / len(ym),
                   bal=(tp / (tp + fn) + tn / (tn + fp)) / 2,
                   val_sens=float((sv[yv == 1] >= thr).mean()),
                   val_spec=float((sv[yv == 0] < thr).mean()))
        for f in ("sens", "spec", "acc", "bal"):
            row[f + "_ci"] = ci([at(i, f) for i in BOOTM])
        p2 = (s2 >= thr).astype(int)
        k1 = int(p2[yp2 == 1].sum()); n1 = int((yp2 == 1).sum())
        k0 = int((p2[yp2 == 0] == 0).sum()); n0 = int((yp2 == 0).sum())
        row["ph2"] = dict(tp=k1, n_mel=n1, tn=k0, n_nv=n0,
                          sens=k1 / n1, spec=k0 / n0,
                          sens_ci=wilson(k1, n1), spec_ci=wilson(k0, n0))
        out["points"].append(row)
    return out

rng0 = np.random.RandomState(0)          # same search as melnv_points.py
V = np.stack([vvec(t) for t in POOL])
cmv = Vm & ~np.isnan(V).any(0)
yv_ = (ytr_full[cmv] == MEL).astype(int)
best = None
for _ in range(3000):
    w = rng0.dirichlet(np.ones(len(POOL)))
    a = roc_auc_score(yv_, w @ V[:, cmv])
    if best is None or a > best[0]:
        best = (a, w)
MN = dict(n=int(mask.sum()), n_mel=int(ym.sum()), n_nv=int((ym == 0).sum()),
          n_lesions=int(len(set(gm))),
          n_mel_lesions=int(len(set(gm[ym == 1]))),
          systems=[
              melnv_system("EfficientNetV2-S 384 (single family)", ["boost4"]),
              melnv_system("EffV2-S 320 + 384 (equal weights)", ["boost2", "boost4"]),
              melnv_system("EffV2-S 384 + 320 + MSAP-CE (equal weights)", ["boost4", "boost2", "merit-ce"]),
              melnv_system("Weighted fusion (7 systems)", POOL, best[1]),
              melnv_system("MERIT-Net (CB-EDL, matched budget)", ["main"]),
              melnv_system("MSAP encoder + CE (matched budget)", ["merit-ce"]),
              melnv_system("ConvNeXt-T + CE (matched budget)", ["cnx-ce"]),
              melnv_system("MERIT ConvNeXt-S 384 (boost5)", ["boost5"]),
          ])
OUT["melnv"] = MN
for s in MN["systems"]:
    yp = [p for p in s["points"] if p["point"] == "valYouden"][0]
    print(f"melnv {s['name']:<44} AUC {s['auc']:.4f} "
          f"[{s['auc_ci'][0]:.3f},{s['auc_ci'][1]:.3f}] Youden thr {yp['thr']:.4f} "
          f"TP {yp['tp']} FN {yp['fn']} TN {yp['tn']} FP {yp['fp']} "
          f"sens {100*yp['sens']:.1f} [{100*yp['sens_ci'][0]:.1f},{100*yp['sens_ci'][1]:.1f}] "
          f"spec {100*yp['spec']:.1f} [{100*yp['spec_ci'][0]:.1f},{100*yp['spec_ci'][1]:.1f}] "
          f"acc {100*yp['acc']:.1f} [{100*yp['acc_ci'][0]:.1f},{100*yp['acc_ci'][1]:.1f}] "
          f"| PH2 AUC {s['ph2_auc']:.3f} sens {yp['ph2']['tp']}/{yp['ph2']['n_mel']} "
          f"spec {yp['ph2']['tn']}/{yp['ph2']['n_nv']}")
print("fusion weights:", dict(zip(POOL, np.round(best[1], 4))))

# malignant-vs-benign derived score (akiec + bcc + mel), for the record
MAL = [0, 1, 4]
ymal = np.isin(y, MAL).astype(int)
OUT["malignant"] = {}
for t in ("boost4", "screen1", "screen2", "boost2", "merit-ce", "main"):
    P = ens_p(t) if D[t]["config"].get("task") != "screen" else None
    if P is None:
        Pb = np.mean([np.array(a["test_p"]) for a in D[t]["arrays"]], 0)
        s = Pb[:, 1] / Pb.sum(1)
    else:
        s = P[:, MAL].sum(1)
    OUT["malignant"][t] = dict(
        auc=float(roc_auc_score(ymal, s)),
        auc_ci=ci([roc_auc_score(ymal[i], s[i]) for i in BOOT]))
print("malignant AUC:", json.dumps(OUT["malignant"]))

# ============================================== 6. boost rows + CIs
BST = {}
for t in BOOST:
    d = D[t]
    binary = d["config"].get("task") == "screen"
    Pb = np.mean([np.array(a["test_p"]) for a in d["arrays"]], 0)
    Pb = Pb / Pb.sum(1, keepdims=True)
    pr = Pb.argmax(1)
    yy = ymal if binary else y
    kk = 2 if binary else K
    P2 = np.mean([np.array(a["ph2_p"]) for a in d["arrays"]], 0)
    s2 = P2[:, 1] if binary else P2[:, MEL]
    rr = np.random.RandomState(7); v = []
    for _ in range(NBOOT):
        i = rr.randint(0, len(yp2), len(yp2))
        if 0 < yp2[i].sum() < len(i):
            v.append(roc_auc_score(yp2[i], s2[i]))
    fb = np.array([f["test"]["bal_acc"] for f in d["folds"]])
    BST[t] = dict(
        binary=binary, nf=len(d["folds"]),
        fold_bal_mean=float(fb.mean()), fold_bal_sd=float(fb.std(ddof=1)),
        bal_acc=bal_acc(yy, pr, kk),
        bal_acc_ci=boot_stat(lambda i: bal_acc(yy[i], pr[i], kk)),
        acc=float((pr == yy).mean()),
        acc_ci=boot_stat(lambda i: float((pr[i] == yy[i]).mean())),
        auc=float(roc_auc_score(yy, Pb[:, 1]) if binary else
                  roc_auc_score(yy, Pb, multi_class="ovr", average="macro")),
        mel_recall=(None if binary else
                    float((pr[y == MEL] == MEL).mean())),
        ph2_auc=float(roc_auc_score(yp2, s2)), ph2_auc_ci=ci(v),
        config={k: d["config"].get(k) for k in
                ("arch", "backbone", "size", "epochs", "bs", "mixup", "tta",
                 "loss", "task")})
    b = BST[t]
    print(f"boost {t:<8} bal {100*b['bal_acc']:.1f} "
          f"[{100*b['bal_acc_ci'][0]:.1f},{100*b['bal_acc_ci'][1]:.1f}] "
          f"acc {100*b['acc']:.1f} [{100*b['acc_ci'][0]:.1f},"
          f"{100*b['acc_ci'][1]:.1f}] AUC {b['auc']:.3f} PH2 {b['ph2_auc']:.3f}"
          f" [{b['ph2_auc_ci'][0]:.3f},{b['ph2_auc_ci'][1]:.3f}]")
OUT["boost"] = BST

# PH2 paired AUC differences among matched-budget models
def ph2_diff(a, b, nf=3):
    sa = ens_p(a, nf, "ph2_p")[:, MEL]; sb = ens_p(b, nf, "ph2_p")[:, MEL]
    rr = np.random.RandomState(7); v = []
    for _ in range(NBOOT):
        i = rr.randint(0, len(yp2), len(yp2))
        if 0 < yp2[i].sum() < len(i):
            v.append(roc_auc_score(yp2[i], sa[i]) - roc_auc_score(yp2[i], sb[i]))
    return dict(est=float(roc_auc_score(yp2, sa) - roc_auc_score(yp2, sb)),
                ci=ci(v))
OUT["ph2_diffs"] = {
    "m-edl - main": ph2_diff("m-edl", "main"),
    "dense121-ce - main": ph2_diff("dense121-ce", "main"),
    "merit-ce - main": ph2_diff("merit-ce", "main"),
    "m-edl - merit-ce": ph2_diff("m-edl", "merit-ce"),
    "m-beta9999 - main": ph2_diff("m-beta9999", "main")}
print("PH2 diffs:", json.dumps(OUT["ph2_diffs"]))

# ============================================== 7. duplicate-pair sensitivity
pairs = pd.read_csv(os.path.join(REV, "abhishek2025_pairs_partition.csv"))
bad_imgs = []
for _, r in pairs[pairs.span == "TEST-TRAIN"].iterrows():
    bad_imgs.append(r.img_a if r.part_a == "test" else r.img_b)
te_img = lab.image_id.values[te_idx]
excl = ~np.isin(te_img, bad_imgs)
assert (~excl).sum() == len(bad_imgs)
SENS = {"n_excluded": int((~excl).sum()), "excluded": bad_imgs,
        "dx_excluded": [str(lab.dx.values[te_idx][i])
                        for i in np.where(~excl)[0]], "rows": {}}
for t in CAMPAIGN:
    for nf in ((3, 5) if nfolds(t) == 5 else (3,)):
        a = (S5 if nf == 5 else S3)[t]
        b = summary(t, nf, mask=excl)
        SENS["rows"][f"{t}@{nf}"] = dict(
            bal_full=a["bal_acc"], bal_excl=b["bal_acc"],
            acc_full=a["acc"], acc_excl=b["acc"])
SENS["max_abs_d_bal"] = float(max(abs(r["bal_full"] - r["bal_excl"])
                                  for r in SENS["rows"].values()))
SENS["max_abs_d_acc"] = float(max(abs(r["acc_full"] - r["acc_excl"])
                                  for r in SENS["rows"].values()))
# primary contrasts recomputed without the 5 images
SENS["contrasts"] = []
for a, b, nf, lab_ in PRIMARY:
    pa, pb = ens_p(a, nf).argmax(1)[excl], ens_p(b, nf).argmax(1)[excl]
    ye = y[excl]
    mc = mcnemar(pa == ye, pb == ye)
    SENS["contrasts"].append(dict(label=lab_, d_bal=bal_acc(ye, pa)
                                  - bal_acc(ye, pb), **mc))
OUT["dup_sensitivity"] = SENS
print("dup sensitivity: excluded", SENS["n_excluded"], SENS["dx_excluded"],
      "max |dBal| %.3f pt, max |dAcc| %.3f pt" %
      (100 * SENS["max_abs_d_bal"], 100 * SENS["max_abs_d_acc"]))

# ============================================== 8. image-level control
ctl = load("ctl-imgsplit")
cln = load("ctl-imgclean")
if ctl is not None:
    te_c = np.array(ctl["te_idx"]); yc = np.array(ctl["test_y"])
    trv_c = np.array(ctl["trv_idx"])
    les_all = lab.lesion_id.values
    exposed = np.isin(les_all[te_c], les_all[trv_c])
    gc = les_all[te_c]
    Bc = cluster_boot(gc, rng=np.random.RandomState(5))
    def arm(d):
        P = np.mean([np.array(a["test_p"]) for a in d["arrays"]], 0)
        P = P / P.sum(1, keepdims=True); pr = P.argmax(1)
        fb = np.array([f["test"]["bal_acc"] for f in d["folds"]])
        def sub(m):
            return dict(
                n=int(m.sum()), bal_acc=bal_acc(yc[m], pr[m]),
                acc=float((pr[m] == yc[m]).mean()),
                bal_acc_ci=ci([bal_acc(yc[i][m[i]], pr[i][m[i]])
                               for i in Bc if m[i].sum()]),
                acc_ci=ci([float((pr[i][m[i]] == yc[i][m[i]]).mean())
                           for i in Bc if m[i].sum()]),
                mel_recall=float((pr[m][yc[m] == MEL] == MEL).mean()),
                per_class_n=np.bincount(yc[m], minlength=K).tolist())
        return dict(nf=len(d["folds"]), n_trainval=int(d["n_trainval"]),
                    fold_bal_mean=float(fb.mean()),
                    fold_bal_sd=float(fb.std(ddof=1)),
                    all=sub(np.ones(len(yc), bool)), exposed=sub(exposed),
                    unexposed=sub(~exposed)), pr
    CT = dict(n_test=int(len(yc)), n_exposed=int(exposed.sum()),
              frac_exposed=float(exposed.mean()),
              n_test_lesions=int(len(set(gc))),
              grouped_ref=dict(bal_acc=S3["cnx-ce"]["bal_acc"],
                               bal_acc_ci=S3["cnx-ce"]["bal_acc_ci"],
                               acc=S3["cnx-ce"]["acc"],
                               acc_ci=S3["cnx-ce"]["acc_ci"],
                               fold_bal_mean=S3["cnx-ce"]["fold_bal_mean"],
                               fold_bal_sd=S3["cnx-ce"]["fold_bal_sd"],
                               n_trainval=int(D["cnx-ce"]["n_trainval"])))
    CT["leaky"], pr_l = arm(ctl)
    if cln is not None:
        assert cln["te_idx"] == ctl["te_idx"]
        assert not set(les_all[np.array(cln["trv_idx"])]) & set(gc)
        CT["clean"], pr_c = arm(cln)
        CT["n_dropped"] = int(len(trv_c) - cln["n_trainval"])
        def dd(i, f):
            e = exposed[i]; yy = yc[i]; a_, b_ = pr_l[i], pr_c[i]
            if f == "bal":
                g = lambda m, p: bal_acc(yy[m], p[m])
            else:
                g = lambda m, p: float((p[m] == yy[m]).mean())
            de = g(e, a_) - g(e, b_); du = g(~e, a_) - g(~e, b_)
            da = g(np.ones(len(yy), bool), a_) - g(np.ones(len(yy), bool), b_)
            return de, du, de - du, da
        full = np.arange(len(yc))
        for f in ("bal", "acc"):
            est = dd(full, f)
            bs = np.array([dd(i, f) for i in Bc])
            CT[f"leaky_minus_clean_{f}"] = dict(
                exposed=est[0], exposed_ci=ci(bs[:, 0]),
                unexposed=est[1], unexposed_ci=ci(bs[:, 1]),
                did=est[2], did_ci=ci(bs[:, 2]),
                all=est[3], all_ci=ci(bs[:, 3]))
        ce, cc = pr_l == yc, pr_c == yc
        CT["mcnemar_exposed"] = mcnemar(ce[exposed], cc[exposed])
        CT["mcnemar_unexposed"] = mcnemar(ce[~exposed], cc[~exposed])
    OUT["image_level_control"] = CT
    print("image-level control:", json.dumps(CT)[:2500])
# expected exposure under random image-level 80/20 splits
rr = np.random.RandomState(3)
les_all = lab.lesion_id.values
fr = []
for _ in range(200):
    perm = rr.permutation(len(les_all))
    te_r, tr_r = perm[:2003], perm[2003:]
    fr.append(np.isin(les_all[te_r], les_all[tr_r]).mean())
OUT["random_split_exposure"] = dict(mean=float(np.mean(fr)),
                                    sd=float(np.std(fr)), n=len(fr))
OUT["_n_secondary"] = len(sec)
print("random image split: test images with a same-lesion training image:",
      OUT["random_split_exposure"])

json.dump(OUT, open(os.path.join(REV, "rev_analysis.json"), "w"), indent=1)
print("rev_analysis.json written")
