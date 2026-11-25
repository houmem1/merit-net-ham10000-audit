# -*- coding: utf-8 -*-
"""Revision assets: numbers_rev.tex, tab_*.tex, new/updated figures and the
supplementary tables, ALL generated from results/revision/rev_analysis.json
(+ split_assignments.csv, dupscreen.json). No number is typed by hand."""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
REV = os.path.join(RES, "revision")
PAPER = os.path.join(HERE, "..", "paper_rev")
FIG = os.path.join(PAPER, "figures")
R = json.load(open(os.path.join(REV, "rev_analysis.json")))
lab = pd.read_csv(os.path.join(REV, "split_assignments.csv"))
dupscreen = json.load(open(os.path.join(REV, "dupscreen.json")))
pairs = pd.read_csv(os.path.join(REV, "abhishek2025_pairs_partition.csv"))
S3, S5 = R["summary3"], R["summary5"]

NUM = {}
def put(k, v):
    NUM[k] = str(v)

def pc(x, d=1):
    return f"{100 * x:.{d}f}"
def f3(x):
    return f"{x:.3f}"
def cir(c, d=1):                       # percentage CI
    return f"{100 * c[0]:.{d}f}--{100 * c[1]:.{d}f}"
def ci3(c):
    return f"{c[0]:.3f}--{c[1]:.3f}"
def sg(x, d=1):                        # signed percentage points
    v = 100 * x
    if abs(v) < 0.05:
        v = 0.0
    return f"${'+' if v >= 0 else '-'}{abs(v):.{d}f}$"
def sgci(c, d=1):
    return f"{sg(c[0], d)} to {sg(c[1], d)}"
def sg3(x):
    return f"${'+' if x >= 0 else '-'}{abs(x):.3f}$"
def pv(p):
    if p >= 0.9995:
        return "1.00"
    if p >= 0.01:
        return f"{p:.2f}" if p >= 0.095 else f"{p:.3f}"
    if p >= 0.001:
        return f"{p:.4f}"
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    if round(m, 1) >= 10:
        m, e = 1.0, e + 1
    return f"${m:.1f}\\times10^{{{e}}}$"

NAMES = {
    "main": "MERIT-Net (MSAP + CB-EDL)", "merit-ce": "MSAP encoder + CE",
    "cnx-ce": "ConvNeXt-T + CE", "cnx-cbedl": "ConvNeXt-T + CB-EDL",
    "r50-ce": "ResNet-50 + CE", "effb3-ce": "EfficientNet-B3 + CE",
    "dense121-ce": "DenseNet-121 + CE", "deit3s-ce": "DeiT3-S + CE",
    "swinv2t-ce": "SwinV2-T + CE", "cnx-wce": "ConvNeXt-T + weighted CE",
    "cnx-focal": "ConvNeXt-T + focal", "cnx-cbce": "ConvNeXt-T + CB-CE",
    "cnx-edl": "ConvNeXt-T + EDL (unweighted)",
    "m-gap": "mean pooling", "m-attplain": "ungated attention",
    "m-single": "single scale (stride 32)",
    "m-fusemean": "mean scale fusion",
    "m-lam0": r"$\lambda_{\max}{=}0$", "m-lam01": r"$\lambda_{\max}{=}0.1$",
    "m-beta9999": r"$\beta_{\mathrm{eff}}{=}0.9999$",
    "m-edl": "unweighted EDL", "m-effb3": "EfficientNet-B3 backbone",
    "m-224": r"$224^2$ input",
    "cnx-cbedl-lam01": r"ConvNeXt-T + CB-EDL, $\lambda_{\max}{=}0.1$",
    "m-edl-lam01": r"MERIT, unweighted EDL, $\lambda_{\max}{=}0.1$",
}
CPX = json.load(open(os.path.join(RES, "complexity.json")))
PAR = {"main": "merit_cnx", "merit-ce": "merit_cnx", "cnx-ce": "plain_cnx",
       "cnx-cbedl": "plain_cnx", "r50-ce": "r50", "effb3-ce": "effb3",
       "dense121-ce": "dense121", "deit3s-ce": "deit3s",
       "swinv2t-ce": "swinv2t"}
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DF = 3

# ------------------------------------------------------------ numbers
for pre, S in (("s3", S3), ("s5", S5)):
    for t, s in S.items():
        k = f"{pre}.{t}."
        put(k + "bal", pc(s["bal_acc"])); put(k + "acc", pc(s["acc"]))
        put(k + "melrec", pc(s["mel_recall"]))
        put(k + "dfrec", pc(s["per_class_recall"][DF]))
        put(k + "ece", f3(s["ece15"])); put(k + "nll", f3(s["nll"]))
        put(k + "aucmelnv", f3(s["auc_melnv"]))
        put(k + "folds", f"{pc(s['fold_bal_mean'])}$\\pm${pc(s['fold_bal_sd'])}")
        if "bal_acc_ci" in s:
            put(k + "balci", cir(s["bal_acc_ci"]))
            put(k + "accci", cir(s["acc_ci"]))
            put(k + "ph2", f3(s["ph2_auc"]))
            put(k + "ph2ci", ci3(s["ph2_auc_ci"]))

PRIM_ID = ["att_ce", "att_edl", "obj_msap", "obj_plain", "kl01", "kl0",
           "gap01"]
def put_contrast(cid, c):
    k = f"c.{cid}."
    put(k + "dbal", sg(c["d_bal"])); put(k + "dbalci", sgci(c["d_bal_ci"]))
    put(k + "dbalabs", f"{abs(100 * c['d_bal']):.1f}")
    put(k + "dacc", sg(c["d_acc"])); put(k + "daccci", sgci(c["d_acc_ci"]))
    put(k + "daccabs", f"{abs(100 * c['d_acc']):.1f}")
    put(k + "aonly", c["a_only"]); put(k + "bonly", c["b_only"])
    put(k + "p", pv(c["p"])); put(k + "pholm", pv(c["p_holm"]))
for cid, c in zip(PRIM_ID, R["contrasts_primary"]):
    put_contrast(cid, c)
SEC = {}
for c in R["contrasts_secondary"]:
    cid = f"{c['a']}__{c['b']}"
    SEC[cid] = c
    put_contrast(cid, c)
put("misc.n_secondary", len(SEC))
put("misc.nboot", f"{R['_nboot']:,}".replace(",", "{,}"))

for key, r in R["calibration"].items():
    k = f"cal.{key}."
    for st in ("raw", "temp", "prior"):
        if st in r:
            put(k + st + ".ece", f3(r[st]["ece15"]))
            put(k + st + ".ececi", ci3(r[st]["ece15_ci"]))
            put(k + st + ".nll", f3(r[st]["nll"]))
            put(k + st + ".aurc", f3(r[st]["aurc_conf"]))
            put(k + st + ".acc", pc(r[st]["acc"]))
    put(k + "T", f"{np.mean(r['temp']['T']):.2f}")
    put(k + "auroc_conf", f3(r["auroc_conf_error"]))
    if "aurc_u" in r:
        put(k + "aurcu", f3(r["aurc_u"]))
        put(k + "aurcuci", ci3(r["aurc_u_ci"]))
        put(k + "ucorr", f3(r["mean_u_correct"]))
        put(k + "uerr", f3(r["mean_u_error"]))
        put(k + "auroc_u", f3(r["auroc_u_error"]))
ap = R["aurc_paired"]
put("aurc.u_vs_conf", sg3(ap["main_u_minus_main_conf"]["est"]))
put("aurc.u_vs_confci", f"{sg3(ap['main_u_minus_main_conf']['ci'][0])} to "
    f"{sg3(ap['main_u_minus_main_conf']['ci'][1])}")
put("aurc.T_vs_ce", sg3(ap["main_Tconf_minus_meritce_conf"]["est"]))
put("aurc.T_vs_ceci",
    f"{sg3(ap['main_Tconf_minus_meritce_conf']['ci'][0])} to "
    f"{sg3(ap['main_Tconf_minus_meritce_conf']['ci'][1])}")

RF = R["referral"]
for lv, row in RF["levels"].items():
    for nm, r in row.items():
        k = f"ref.{lv}.{nm}."
        put(k + "kept", r["n_kept"]); put(k + "acc", pc(r["acc_kept"]))
        put(k + "melref", r["mel_referred"]); put(k + "tp", r["mel_kept_tp"])
        put(k + "fn", r["mel_kept_fn"]); put(k + "meltot", r["mel_total"])
d80 = RF["diff80_meritU_minus_msapceConf"]
put("ref.d80.fn", int(d80["mel_fn_kept"]["est"]))
put("ref.d80.fnci", f"{int(d80['mel_fn_kept']['ci'][0])} to "
    f"{int(d80['mel_fn_kept']['ci'][1])}")
put("ref.d80.acc", sg(d80["acc_kept"]["est"]))
put("ref.d80.accci", sgci(d80["acc_kept"]["ci"]))
for nm, r in RF["val_locked"].items():
    k = f"vl.{nm}."
    put(k + "cov", pc(r["coverage"])); put(k + "kept", r["n_kept"])
    put(k + "acc", pc(r["acc_kept"])); put(k + "melref", r["mel_referred"])
    put(k + "tp", r["mel_kept_tp"]); put(k + "fn", r["mel_kept_fn"])
    put(k + "tau", f3(r["tau"]))

for t, b in R["boost"].items():
    k = f"b.{t}."
    put(k + "bal", pc(b["bal_acc"])); put(k + "balci", cir(b["bal_acc_ci"]))
    put(k + "acc", pc(b["acc"])); put(k + "accci", cir(b["acc_ci"]))
    put(k + "auc", f3(b["auc"])); put(k + "ph2", f3(b["ph2_auc"]))
    put(k + "ph2ci", ci3(b["ph2_auc_ci"]))
for t, m in R["malignant"].items():
    put(f"mal.{t}.auc", f3(m["auc"])); put(f"mal.{t}.aucci", ci3(m["auc_ci"]))

MN = R["melnv"]
put("mn.n", f"{MN['n']:,}".replace(",", "{,}")); put("mn.n_mel", MN["n_mel"])
put("mn.n_nv", f"{MN['n_nv']:,}".replace(",", "{,}"))
put("mn.n_mel_les", MN["n_mel_lesions"])
put("mn.prev", pc(MN["n_mel"] / MN["n"]))
SID = ["p4", "p24", "p42m", "fus", "merit", "msapce", "cnxce", "p5"]
MSYS = dict(zip(SID, MN["systems"]))
def youden(s):
    return [p for p in s["points"] if p["point"] == "valYouden"][0]
for sid, s in MSYS.items():
    yp = youden(s); k = f"mn.{sid}."
    put(k + "auc", f3(s["auc"])); put(k + "aucci", ci3(s["auc_ci"]))
    put(k + "thr", f3(yp["thr"])); put(k + "nval", f"{s['n_val']:,}".replace(",", "{,}"))
    put(k + "valauc", f3(s["val_auc"]))
    for f in ("tp", "fn", "tn", "fp"):
        put(k + f, f"{yp[f]:,}".replace(",", "{,}"))
    for f in ("sens", "spec", "acc", "bal"):
        put(k + f, pc(yp[f])); put(k + f + "ci", cir(yp[f + "_ci"]))
    p2 = yp["ph2"]
    put(k + "ph2tp", p2["tp"]); put(k + "ph2nmel", p2["n_mel"])
    put(k + "ph2tn", p2["tn"]); put(k + "ph2nnv", p2["n_nv"])
    put(k + "ph2sens", pc(p2["sens"])); put(k + "ph2sensci", cir(p2["sens_ci"]))
    put(k + "ph2spec", pc(p2["spec"])); put(k + "ph2specci", cir(p2["spec_ci"]))
    put(k + "ph2auc", f3(s["ph2_auc"]))
    put(k + "ph2specdrop", f"{100 * (yp['spec'] - p2['spec']):.0f}")
fus = MSYS["fus"]
for t, w in zip(fus["tags"], fus["weights"]):
    put(f"mn.w.{t}", f"{w:.3f}")

pd_ = R["ph2_diffs"]
put("ph2.m-edl_main", sg3(pd_["m-edl - main"]["est"]))
put("ph2.m-edl_mainci", f"{sg3(pd_['m-edl - main']['ci'][0])} to "
    f"{sg3(pd_['m-edl - main']['ci'][1])}")
put("ph2.m-edl_meritce", sg3(pd_["m-edl - merit-ce"]["est"]))
put("ph2.m-edl_meritceci", f"{sg3(pd_['m-edl - merit-ce']['ci'][0])} to "
    f"{sg3(pd_['m-edl - merit-ce']['ci'][1])}")

# ---- data description
cnt = lab.groupby("lesion_id").size()
te = lab[lab.partition == "test"]
put("data.n_les", f"{lab.lesion_id.nunique():,}".replace(",", "{,}"))
put("data.multi_les", f"{100 * (cnt > 1).mean():.1f}")
put("data.multi_img", f"{100 * cnt[cnt > 1].sum() / len(lab):.1f}")
put("data.n_test", f"{len(te):,}".replace(",", "{,}"))
put("data.n_test_les", f"{te.lesion_id.nunique():,}".replace(",", "{,}"))
put("data.n_trv", f"{len(lab) - len(te):,}".replace(",", "{,}"))
put("data.img_per_les", f"{len(te) / te.lesion_id.nunique():.2f}")
dt = te.dx_type.value_counts(normalize=True)
put("data.te_histo", f"{100 * dt['histo']:.1f}")
put("data.te_follow", f"{100 * dt['follow_up']:.1f}")
put("data.te_cons", f"{100 * dt['consensus']:.1f}")
put("data.te_conf", f"{100 * dt['confocal']:.1f}")
nvt = te[te.dx == "nv"].dx_type.value_counts()
put("data.te_nv_histo", int(nvt["histo"]))
put("data.te_nv_follow", int(nvt["follow_up"]))
put("data.te_nv_cons", int(nvt["consensus"]))
for c in ("mel", "bcc", "akiec"):
    assert (te[te.dx == c].dx_type == "histo").all(), c
put("data.mal_prev", f"{100 * te.dx.isin(['mel', 'bcc', 'akiec']).mean():.1f}")

sp = pairs.span.value_counts()
put("dup.n_tt", int(sp.get("TEST-TRAIN", 0)))
put("dup.n_cv", int(sp.get("cv-cv", 0)))
put("dup.n_same", int(sp.get("same cv", 0)))
dxs = R["dup_sensitivity"]["dx_excluded"]
words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
full = {"nv": ("nevus", "nevi"), "bkl": ("benign keratosis", "benign keratoses")}
parts = []
for d in sorted(set(dxs), key=lambda x: -dxs.count(x)):
    n = dxs.count(d)
    parts.append(f"{words[n]} {full[d][0] if n == 1 else full[d][1]}")
put("dup.dx_tt", " and ".join(parts))
put("dup.hash_min", int(dupscreen["min_dist_quantiles"]["0"]))
assert dupscreen["n_exact"] == 0
assert not any(max(c["phash"], c["dhash"]) <= 3 for c in dupscreen["candidates"])
put("dup.max_dbal", f"{100 * R['dup_sensitivity']['max_abs_d_bal']:.2f}")
put("dup.max_dacc", f"{100 * R['dup_sensitivity']['max_abs_d_acc']:.2f}")
rs = R["random_split_exposure"]
put("rnd.n", rs["n"]); put("rnd.mean", f"{100 * rs['mean']:.1f}")
put("rnd.sd", f"{100 * rs['sd']:.1f}")

CT = R.get("image_level_control")
if CT:
    g = CT["grouped_ref"]
    put("ctl.grp.bal", pc(g["bal_acc"])); put("ctl.grp.balci", cir(g["bal_acc_ci"]))
    put("ctl.grp.acc", pc(g["acc"])); put("ctl.grp.accci", cir(g["acc_ci"]))
    put("ctl.n_test", f"{CT['n_test']:,}".replace(",", "{,}"))
    put("ctl.n_exposed", CT["n_exposed"])
    put("ctl.frac_exposed", f"{100 * CT['frac_exposed']:.1f}")
    for arm in ("leaky", "clean"):
        if arm not in CT:
            continue
        for sub in ("all", "exposed", "unexposed"):
            r = CT[arm][sub]
            k = f"ctl.{'' if arm == 'leaky' else 'clean.'}{sub}."
            put(k + "bal", pc(r["bal_acc"])); put(k + "balci", cir(r["bal_acc_ci"]))
            put(k + "acc", pc(r["acc"])); put(k + "accci", cir(r["acc_ci"]))
            put(k + "n", f"{r['n']:,}".replace(",", "{,}"))
    if "clean" in CT:
        put("clean.n_drop", CT["n_dropped"])
        put("clean.n_pool", f"{CT['clean']['n_trainval']:,}".replace(",", "{,}"))
        for f in ("bal", "acc"):
            d = CT[f"leaky_minus_clean_{f}"]
            for s_ in ("exposed", "unexposed", "did", "all"):
                put(f"ctl.d.{f}.{s_}", sg(d[s_]))
                put(f"ctl.d.{f}.{s_}abs", f"{abs(100 * d[s_]):.1f}")
                put(f"ctl.d.{f}.{s_}ciabs", cir(d[s_ + "_ci"]))
                put(f"ctl.d.{f}.{s_}ci", sgci(d[s_ + "_ci"]))
        for s_ in ("exposed", "unexposed"):
            m = CT[f"mcnemar_{s_}"]
            put(f"ctl.mc.{s_}.leaky", m["a_only"])
            put(f"ctl.mc.{s_}.clean", m["b_only"])
            put(f"ctl.mc.{s_}.p", pv(m["p"]))
if "interaction_weights_kl" in R:
    iw = R["interaction_weights_kl"]
    put("klw.inter", sg(iw["est"])); put("klw.interci", sgci(iw["ci"]))
if "seed_replication" in R:
    sr = R["seed_replication"]
    for t, d in sr["seed_diff"].items():
        for f in ("bal", "acc"):
            put(f"seed.{t}.d{f}", sg(d[f]["est"]))
            put(f"seed.{t}.d{f}ci", sgci(d[f]["ci"]))
    put("seed.maxabs", f"{max(abs(100 * d['bal']['est']) for d in sr['seed_diff'].values()):.1f}")
    for cid, d in sr["pooled"].items():
        for f in ("bal", "acc"):
            put(f"pool.{cid}.d{f}", sg(d[f]["est"]))
            put(f"pool.{cid}.d{f}ci", sgci(d[f]["ci"]))
            put(f"pool.{cid}.d{f}abs", f"{abs(100 * d[f]['est']):.1f}")

REPO = os.path.join(REV, "repo.json")
if os.path.exists(REPO):
    rp = json.load(open(REPO))
    put("repo.url", r"\url{%s}" % rp["url"])
    put("repo.doi", r"\url{%s}" % rp["doi"] if rp.get("doi") else "pending")

# ------------------------------------------------------------ tables
def W(name, txt):
    with open(os.path.join(PAPER, name), "w", encoding="utf-8") as f:
        f.write("% GENERATED by rev_make_assets.py - do not edit\n" + txt)

def bci(s, k="bal_acc"):
    return f"{pc(s[k])} [{cir(s[k + '_ci'])}]"

# ---- tab_split
img = pd.crosstab(lab.dx, lab.partition)
les = pd.crosstab(lab.dx, lab.partition, values=lab.lesion_id, aggfunc="nunique").fillna(0).astype(int)
trv = lab[lab.partition != "test"]
rows = []
for c in ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]:
    rows.append(f"{c} & {img.loc[c, 'test']} & {les.loc[c, 'test']} & "
                f"{(trv.dx == c).sum()} & {trv[trv.dx == c].lesion_id.nunique()} & "
                + " & ".join(f"{img.loc[c, f'cv{k}']} / {les.loc[c, f'cv{k}']}" for k in range(1, 6)) + r" \\")
rows.append(r"\midrule" + "\n" + f"Total & {len(te)} & {te.lesion_id.nunique()} & {len(trv)} & "
            f"{trv.lesion_id.nunique()} & " + " & ".join(
                f"{(lab.partition == f'cv{k}').sum()} / {lab[lab.partition == f'cv{k}'].lesion_id.nunique()}"
                for k in range(1, 6)) + r" \\")
W("tab_split.tex", r"""\begin{table}[!t]
\caption{\rev{Images and lesions by diagnosis and partition under the frozen
lesion-grouped protocol (lesion membership from the official
\texttt{lesion\_id}). No lesion appears in more than one partition. CV
columns give images / lesions of each validation fold.}}
\label{tab:split}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
 & \multicolumn{2}{c}{Test} & \multicolumn{2}{c}{Train/validation pool} & \multicolumn{5}{c}{Validation folds (images / lesions)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-10}
Diagnosis & images & lesions & images & lesions & CV1 & CV2 & CV3 & CV4 & CV5 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---- tab_ctl
def th(n):
    return f"{n:,}".replace(",", "{,}")
if CT:
    g = CT["grouped_ref"]
    rows = [f"Lesion-grouped protocol (reference) & {th(g['n_trainval'])} & all test images & "
            f"{th(R['_n_test'])} & {pc(g['bal_acc'])} [{cir(g['bal_acc_ci'])}] & {pc(g['acc'])} [{cir(g['acc_ci'])}] \\\\"]
    for arm, lbl in (("leaky", "Image-level split, pool as drawn (leaky arm)"),
                     ("clean", "Image-level split, lesion-sharing images removed (clean arm)")):
        if arm not in CT:
            continue
        rows.append(r"\midrule")
        for sub, sl in (("all", "all test images"), ("exposed", "exposed test images"),
                        ("unexposed", "unexposed test images")):
            r = CT[arm][sub]
            first = f"{lbl} & {th(CT[arm]['n_trainval'])}" if sub == "all" else " & "
            rows.append(first + f" & {sl} & {th(r['n'])} & {pc(r['bal_acc'])} [{cir(r['bal_acc_ci'])}] & "
                        f"{pc(r['acc'])} [{cir(r['acc_ci'])}] \\\\")
    if "clean" in CT:
        db, da = CT["leaky_minus_clean_bal"], CT["leaky_minus_clean_acc"]
        rows.append(r"\midrule")
        for s_, sl in (("all", "all test images"), ("exposed", "exposed test images"),
                       ("unexposed", "unexposed test images"), ("did", "exposed $-$ unexposed")):
            rows.append(("Leaky $-$ clean (paired difference)" if s_ == "all" else "") + f" & & {sl} & & "
                        f"{sg(db[s_])} [{sgci(db[s_ + '_ci'])}] & {sg(da[s_])} [{sgci(da[s_ + '_ci'])}] \\\\")
    W("tab_ctl.tex", r"""\begin{table}[!t]
\caption{\rev{Controlled measurement of split-induced inflation (plain
ConvNeXt-T\,+\,CE, identical recipe, three-fold ensembles). ``Exposed''
test images are those whose lesion also has an image in the leaky
training pool. In the clean arm the lesion-sharing pool images are
removed, so the same test images are evaluated without within-lesion
leakage. Brackets: 95\% lesion-cluster bootstrap CI.}}
\label{tab:ctl}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrlrcc}
\toprule
Training condition & Pool (images) & Test subset & $n$ & BalAcc (\%) & Acc (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---- tab_enc
def enc_row(t, S, name=None):
    s = S[t]
    par = CPX[PAR[t]]["params_M"]
    return (f"{name or NAMES[t]} & {par} & {pc(s['fold_bal_mean'])}$\\pm${pc(s['fold_bal_sd'])} & "
            f"{bci(s)} & {bci(s, 'acc')} & {pc(s['f1_macro'])} & {f3(s['auc_macro'])} & "
            f"{pc(s['mel_recall'])} & {f3(s['auc_mel'])} & {f3(s['ph2_auc'])} [{ci3(s['ph2_auc_ci'])}] \\\\")
ORDER = ["main", "merit-ce", "cnx-cbedl", "cnx-ce", "swinv2t-ce", "deit3s-ce", "effb3-ce", "dense121-ce", "r50-ce"]
W("tab_enc.tex", r"""\begin{sidewaystable}[!p]
\caption{\rev{Encoders under the matched recipe on the frozen lesion-grouped
test set (2{,}014 images, 1{,}495 lesions), fold-matched. Upper block:
three-fold ensembles (folds 1--3) of all nine configurations; lower block:
five-fold ensembles of the $2{\times}2$ factorial. ``Folds'' = mean $\pm$ SD
of single-fold models; ensemble = probability averaging; brackets = 95\%
lesion-cluster bootstrap CI (\PHtwo: image bootstrap). BalAcc = balanced
accuracy; MelRec = melanoma recall; AUC$_{\mathrm{mel}}$ = melanoma
one-vs-rest AUC.}}
\label{tab:enc}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
Model & Params (M) & BalAcc folds (\%) & BalAcc ens. (\%) & Acc ens. (\%) &
F1$_{\mathrm{macro}}$ (\%) & AUC$_{\mathrm{macro}}$ & MelRec (\%) &
AUC$_{\mathrm{mel}}$ & \PHtwo{} AUC \\
\midrule
\multicolumn{10}{l}{\emph{Three-fold ensembles (all configurations)}} \\
""" + "\n".join(enc_row(t, S3) for t in ORDER) + r"""
\midrule
\multicolumn{10}{l}{\emph{Five-fold ensembles (factorial only)}} \\
""" + "\n".join(enc_row(t, S5) for t in ORDER[:4]) + r"""
\bottomrule
\end{tabular}}
\end{sidewaystable}
""")

# ---- tab_prim
PL = [
    (r"MSAP\,+\,CE $-$ ConvNeXt-T\,+\,CE", "MSAP encoder, under CE"),
    (r"MERIT-Net $-$ ConvNeXt-T\,+\,CB-EDL", "MSAP encoder, under CB-EDL"),
    (r"MERIT-Net $-$ MSAP\,+\,CE", "CB-EDL objective, MSAP encoder"),
    (r"ConvNeXt-T\,+\,CB-EDL $-$ ConvNeXt-T\,+\,CE", "CB-EDL objective, plain encoder"),
    (r"MERIT $\lambda_{\max}{=}0.1$ $-$ MERIT-Net ($\lambda_{\max}{=}1$)", "KL ceiling 0.1 vs 1"),
    (r"MERIT $\lambda_{\max}{=}0$ $-$ MERIT-Net ($\lambda_{\max}{=}1$)", "KL ceiling 0 vs 1"),
    (r"MERIT $\lambda_{\max}{=}0.1$ $-$ MSAP\,+\,CE", "Remaining gap to CE"),
]
rows = []
for (ab, q), c in zip(PL, R["contrasts_primary"]):
    rows.append(f"{q} & {ab} & {c['nf']} & {sg(c['d_bal'])} [{sgci(c['d_bal_ci'])}] & {sg(c['d_acc'])} [{sgci(c['d_acc_ci'])}] & "
                f"{c['a_only']} & {c['b_only']} & {pv(c['p'])} & {pv(c['p_holm'])} \\\\")
W("tab_prim.tex", r"""\begin{sidewaystable}[!p]
\caption{\rev{Primary paired contrasts on fold-matched ensembles (same test
images). $\Delta$ = A $-$ B in percentage points with 95\% lesion-cluster
bootstrap CI. ``Only A'' / ``only B'' = number of test images classified
correctly by exactly one of the two models; $p$ = exact two-sided McNemar
test (a test of overall accuracy); Holm adjustment within the seven primary
contrasts.}}
\label{tab:prim}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccc}
\toprule
Question & Contrast (A $-$ B) & Folds & $\Delta$BalAcc [95\% CI] & $\Delta$Acc [95\% CI] & Only A & Only B & $p$ & $p_{\mathrm{Holm}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{sidewaystable}
""")

# ---- tab_obj
OBJ = ["cnx-ce", "cnx-wce", "cnx-focal", "cnx-cbce", "cnx-edl", "cnx-cbedl"]
if "cnx-cbedl-lam01" in S3:
    OBJ.append("cnx-cbedl-lam01")
rows = []
for t in OBJ:
    s = S3[t]
    if t == "cnx-ce":
        d = "reference"
    else:
        c = SEC[f"{t}__cnx-ce"]
        d = f"{sg(c['d_bal'])} [{sgci(c['d_bal_ci'])}]"
    rows.append(f"{NAMES[t]} & {pc(s['fold_bal_mean'])}$\\pm${pc(s['fold_bal_sd'])} & {bci(s)} & {pc(s['acc'])} & "
                f"{pc(s['mel_recall'])} & {pc(s['per_class_recall'][DF])} & {f3(s['ece15'])} & {f3(s['nll'])} & {d} \\\\")
W("tab_obj.tex", r"""\begin{table}[!t]
\caption{\rev{Imbalance objectives on the plain ConvNeXt-T encoder
(three-fold ensembles, shared recipe). DfRec = dermatofibroma (rarest
class, 22 test images) recall. Last column: paired difference in balanced
accuracy to cross-entropy with 95\% lesion-cluster CI.}}
\label{tab:obj}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
Objective & BalAcc folds (\%) & BalAcc ens. (\%) & Acc (\%) & MelRec (\%) & DfRec (\%) & ECE & NLL & $\Delta$BalAcc vs CE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---- tab_abl
CALK = R["calibration"]
ABL = [("main", "full model (three-fold ensemble)", "1", "0.999"),
       ("m-gap", None, "1", "0.999"), ("m-attplain", None, "1", "0.999"),
       ("m-single", None, "1", "0.999"), ("m-fusemean", None, "1", "0.999"),
       ("m-lam0", "KL term removed", "0", "0.999"),
       ("m-lam01", "reduced KL ceiling", "0.1", "0.999"),
       ("m-beta9999", "stronger class weights", "1", "0.9999"),
       ("m-edl", "unweighted EDL", "1", "-- (none)"),
       ("m-effb3", None, "1", "0.999"), ("m-224", None, "1", "0.999")]
A_old = json.load(open(os.path.join(RES, "analysis.json")))
rows = []
for t, nm, lam, be in ABL:
    s = S3[t]
    nm = nm or NAMES[t]
    if t == "main":
        d, p = "reference", "--"
        au = CALK["main@3"]["aurc_u"]
    else:
        c = SEC[f"{t}__main"] if f"{t}__main" in SEC else None
        if c is None:
            c = [x for x, i in zip(R["contrasts_primary"], PRIM_ID) if i in ("kl01", "kl0") and x["a"] == t][0]
        d = f"{sg(c['d_bal'])} [{sgci(c['d_bal_ci'])}]"
        p = f"{sg(c['d_acc'])}; {pv(c['p_holm'])}"
        au = CALK[f"{t}@3"]["aurc_u"] if f"{t}@3" in CALK else A_old[t]["aurc_u"]
    rows.append(f"{nm} & {lam} & {be} & {pc(s['fold_bal_mean'])}$\\pm${pc(s['fold_bal_sd'])} & {pc(s['bal_acc'])} & "
                f"{pc(s['acc'])} & {pc(s['mel_recall'])} & {d} & {p} & {f3(s['ece15'])} & {f3(au)} & {f3(s['ph2_auc'])} \\\\")
W("tab_abl.tex", r"""\begin{sidewaystable}[!p]
\caption{\rev{MERIT-Net ablations, all on the same three folds (three-fold
ensembles); the first row is the full model on those folds.
$\Delta$BalAcc = variant $-$ full model with 95\% lesion-cluster CI;
$\Delta$Acc; $p_{\mathrm{Holm}}$ = Holm-adjusted exact McNemar test. ECE of
the raw Dirichlet mean; AURC$_u$ uses the vacuity $u$.}}
\label{tab:abl}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccccc}
\toprule
Variant & $\lambda_{\max}$ & $\beta_{\mathrm{eff}}$ & BalAcc folds (\%) & BalAcc ens. (\%) & Acc (\%) & MelRec (\%) &
$\Delta$BalAcc vs full & $\Delta$Acc; $p_{\mathrm{Holm}}$ & ECE & AURC$_u$ & \PHtwo{} AUC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{sidewaystable}
""")

# ---- tab_klw
def klw_row(lbl, t):
    if t not in S3:
        return None
    s = S3[t]
    return (f"{lbl} & {pc(s['fold_bal_mean'])}$\\pm${pc(s['fold_bal_sd'])} & {bci(s)} & {bci(s, 'acc')} & "
            f"{pc(s['mel_recall'])} & {pc(s['per_class_recall'][DF])} & {f3(s['ece15'])} \\\\")
blocks = [
    (r"\emph{MERIT encoder, seed 42: class weights $\times$ KL ceiling}",
     [(r"unweighted EDL, $\lambda_{\max}{=}1$", "m-edl"),
      (r"unweighted EDL, $\lambda_{\max}{=}0.1$", "m-edl-lam01"),
      (r"CB-EDL, $\lambda_{\max}{=}1$ (MERIT-Net)", "main"),
      (r"CB-EDL, $\lambda_{\max}{=}0.1$", "m-lam01"),
      (r"cross-entropy (MSAP\,+\,CE)", "merit-ce")]),
    (r"\emph{Plain ConvNeXt-T encoder, seed 42}",
     [(r"CB-EDL, $\lambda_{\max}{=}1$", "cnx-cbedl"),
      (r"CB-EDL, $\lambda_{\max}{=}0.1$", "cnx-cbedl-lam01"),
      (r"cross-entropy", "cnx-ce")]),
    (r"\emph{MERIT encoder, seed 43 (replication)}",
     [(r"CB-EDL, $\lambda_{\max}{=}1$", "main-s43"),
      (r"CB-EDL, $\lambda_{\max}{=}0.1$", "m-lam01-s43"),
      (r"cross-entropy (MSAP\,+\,CE)", "merit-ce-s43")])]
rows = []
for hdr, items in blocks:
    rr = [klw_row(l, t) for l, t in items]
    rr = [x for x in rr if x]
    if len(rr) == len(items):
        rows += ([r"\midrule"] if rows else []) + [f"\\multicolumn{{7}}{{l}}{{{hdr}}} \\\\"] + rr
W("tab_klw.tex", r"""\begin{table}[!t]
\caption{\rev{Class weights, KL ceiling and reproducibility (three-fold
ensembles on the shared folds, matched recipe; brackets = 95\%
lesion-cluster CI). The unweighted $\lambda_{\max}{=}0.1$ cell, the plain
ConvNeXt-T $\lambda_{\max}{=}0.1$ run and the seed-43 runs are revision
experiments.}}
\label{tab:klw}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
Configuration & BalAcc folds (\%) & BalAcc ens. (\%) & Acc ens. (\%) & MelRec (\%) & DfRec (\%) & ECE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---- tab_cal
rows = []
for key, nm in (("main@5", "MERIT-Net (5 folds)"), ("cnx-cbedl@5", "ConvNeXt-T + CB-EDL (5)"),
                ("cnx-edl@3", "ConvNeXt-T + EDL (3)"), ("m-lam01@3", r"MERIT $\lambda_{\max}{=}0.1$ (3)"),
                ("merit-ce@5", "MSAP encoder + CE (5)"), ("cnx-ce@5", "ConvNeXt-T + CE (5)")):
    r = CALK[key]
    pr = (f"{f3(r['prior']['ece15'])} / {f3(r['prior']['nll'])}" if "prior" in r else "--")
    au = (f"{f3(r['aurc_u'])} [{ci3(r['aurc_u_ci'])}]" if "aurc_u" in r else "--")
    rows.append(f"{nm} & {pc(r['raw']['acc'])} & {f3(r['raw']['ece15'])} [{ci3(r['raw']['ece15_ci'])}] / {f3(r['raw']['nll'])} & "
                f"{f3(r['temp']['ece15'])} / {f3(r['temp']['nll'])} & {pr} & {au} & "
                f"{f3(r['raw']['aurc_conf'])} [{ci3(r['raw']['aurc_conf_ci'])}] & {f3(r['temp']['aurc_conf'])} \\\\")
W("tab_cal.tex", r"""\begin{sidewaystable}[!p]
\caption{\rev{Calibration and selective prediction before and after post-hoc
recalibration (test ensembles; number of folds in parentheses).
ECE / NLL of the raw probabilities, after temperature scaling and after
the prior-weight correction (evidential models only), each fitted on
validation predictions. AURC$_u$ ranks by the vacuity $u$;
AURC$_{\mathrm{conf}}$ by $1-\max_k \hat p_k$ (raw, and after temperature
scaling). Lower is better; brackets = 95\% lesion-cluster CI.}}
\label{tab:cal}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
Model & Acc (\%) & Raw ECE [CI] / NLL & Temp.\ ECE / NLL & Prior-corr.\ ECE / NLL & AURC$_u$ [CI] & AURC$_{\mathrm{conf}}$ raw [CI] & AURC$_{\mathrm{conf}}$ temp. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{sidewaystable}
""")

# ---- tab_ref
rows = []
for lv in ("95", "90", "80", "70", "50"):
    for nm, lbl in (("merit_u", "MERIT-Net, $u$"), ("msapce_conf", r"MSAP\,+\,CE, confidence")):
        r = RF["levels"][lv][nm]
        rows.append(f"{lv}\\% (test-indexed) & {lbl} & {r['n_kept']} & {pc(r['acc_kept'])} & "
                    f"{r['mel_referred']} & {r['mel_kept_tp']} & {r['mel_kept_fn']} \\\\")
rows.append(r"\midrule")
for tgt in ("90", "80"):
    for nm, lbl in ((f"merit_u_{tgt}", "MERIT-Net, $u$"), (f"msapce_conf_{tgt}", r"MSAP\,+\,CE, confidence")):
        r = RF["val_locked"][nm]
        rows.append(f"val.-locked, target {tgt}\\% (realised {pc(r['coverage'])}\\%) & {lbl} & {r['n_kept']} & "
                    f"{pc(r['acc_kept'])} & {r['mel_referred']} & {r['mel_kept_tp']} & {r['mel_kept_fn']} \\\\")
W("tab_ref.tex", r"""\begin{table}[!t]
\caption{\rev{Referral in counts (five-fold ensembles; """ + str(RF["levels"]["80"]["merit_u"]["mel_total"]) + r""" melanoma images
among the 2{,}014 test images). Upper block: cut-offs indexed on the test
scores (fixed test coverage). Lower block: threshold fixed on pooled
out-of-fold validation scores and applied unchanged to the test ensemble.
``Kept'' = automatically decided images; melanomas are either referred,
kept and correctly called melanoma, or kept and missed.}}
\label{tab:ref}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccc}
\toprule
Coverage & Model, score & Kept ($n$) & Acc.\ on kept (\%) & Melanomas referred & Melanomas kept, correct & Melanomas kept, missed \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---- tab_boost
BR = [("boost", "P1", r"MSAP encoder (ConvNeXt-S) + CE, $320^2$", "7-class"),
      ("boost2", "P2", r"EfficientNetV2-S + CE, $320^2$", "7-class"),
      ("boost3", "P3", r"DeiT3-S + CE, mixup, $320^2$", "7-class"),
      ("boost4", "P4", r"EfficientNetV2-S + CE, mixup, $384^2$", "7-class"),
      ("boost5", "P5", r"MSAP encoder (ConvNeXt-S) + CE, mixup, $384^2$", "7-class"),
      ("screen1", "B1", r"EfficientNetV2-S, binary target, mixup, $384^2$", "malignant vs benign"),
      ("screen2", "B2", r"MSAP encoder (ConvNeXt-S), binary target, mixup, $384^2$", "malignant vs benign")]
rows = []
for t, pid, desc, task in BR:
    b = R["boost"][t]
    rows.append(f"{pid} & {desc} & {task} & {pc(b['fold_bal_mean'])}$\\pm${pc(b['fold_bal_sd'])} & "
                f"{pc(b['bal_acc'])} [{cir(b['bal_acc_ci'])}] & {pc(b['acc'])} [{cir(b['acc_ci'])}] & "
                f"{f3(b['auc'])} & {f3(b['ph2_auc'])} [{ci3(b['ph2_auc_ci'])}] \\\\")
W("tab_boost.tex", r"""\begin{sidewaystable}[!p]
\caption{\rev{Performance-oriented configurations (exploratory stage;
five-fold ensembles on the same frozen lesion-grouped folds, 16 epochs,
four-view TTA). None uses the evidential head; P1 and P5 use the MSAP
encoder. For the binary rows, balanced accuracy and accuracy refer to the
malignant-versus-benign task and AUC is the malignant AUC; for 7-class
rows AUC is macro one-vs-rest. Brackets = 95\% CI (lesion-cluster
bootstrap; \PHtwo: image bootstrap of the melanoma-vs-nevus AUC).}}
\label{tab:boost}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}
\toprule
ID & Configuration & Task & BalAcc folds (\%) & BalAcc ens. (\%) & Acc ens. (\%) & AUC & \PHtwo{} AUC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{sidewaystable}
""")

# ---- tab_melnv
ML = [("fus", "Weighted fusion of 7 systems"), ("p4", "P4 (EfficientNetV2-S, $384^2$)"),
      ("p5", "P5 (MSAP encoder, ConvNeXt-S, $384^2$)"),
      ("msapce", "MSAP encoder + CE (matched budget)"),
      ("cnxce", "ConvNeXt-T + CE (matched budget)"),
      ("merit", "MERIT-Net (matched budget)")]
rows = []
for sid, nm in ML:
    s = MSYS[sid]; yp = youden(s)
    rows.append(f"{nm} & {yp['thr']:.3f} & {yp['tp']} / {yp['fn']} & {yp['tn']} / {yp['fp']} & "
                f"{pc(yp['sens'])} [{cir(yp['sens_ci'])}] & {pc(yp['spec'])} [{cir(yp['spec_ci'])}] & "
                f"{pc(yp['acc'])} [{cir(yp['acc_ci'])}] & {f3(s['auc'])} [{ci3(s['auc_ci'])}] \\\\")
W("tab_melnv.tex", r"""\begin{sidewaystable}[!p]
\caption{\rev{Melanoma-versus-nevus task on the restricted test subset
(""" + f"{MN['n']:,}".replace(",", "{,}") + r""" images: """ + str(MN["n_mel"]) + r""" melanomas, """ + f"{MN['n_nv']:,}".replace(",", "{,}") + r""" nevi; exploratory).
For each system the threshold is the Youden point of its pooled
out-of-fold validation score, applied unchanged to the test score
(five-fold ensemble). TP / FN = melanomas detected / missed; TN / FP = nevi
correctly / wrongly classified. Brackets = 95\% lesion-cluster bootstrap
CI. All systems and operating points examined are listed in Additional
file~1 (Table~S7).}}
\label{tab:melnv}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
System & Threshold & TP / FN & TN / FP & Sensitivity (\%) & Specificity (\%) & Accuracy (\%) & AUC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{sidewaystable}
""")

# ------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})
C1, C2, C3, C4 = "#4878a8", "#c44e52", "#55a868", "#8172b2"

# ---- fig_forest
items = [(q, c) for (ab, q), c in zip(PL, R["contrasts_primary"])]
extra = [("main__cnx-cbce", "MERIT-Net vs ConvNeXt-T + CB-CE"),
         ("cnx-cbce__cnx-ce", "CB-CE vs CE (ConvNeXt-T)"),
         ("m-gap__main", "mean pooling vs gated attention (MERIT)"),
         ("m-single__main", "single scale vs three scales (MERIT)"),
         ("cnx-ce__r50-ce", "ConvNeXt-T vs ResNet-50 (CE)")]
for cid, lbl in (("cnx-cbedl-lam01__cnx-cbedl", "KL ceiling 0.1 vs 1 (ConvNeXt-T CB-EDL)"),
                 ("m-lam01-s43__main-s43", "KL ceiling 0.1 vs 1 (MERIT, seed 43)")):
    if cid in SEC:
        extra.append((cid, lbl))
items += [(lbl, SEC[cid]) for cid, lbl in extra]
fig, ax = plt.subplots(figsize=(6.6, 0.33 * len(items) + 0.9))
for i, (lbl, c) in enumerate(items):
    yv = len(items) - 1 - i
    col = C1 if i < 7 else "#666666"
    lo, hi = 100 * c["d_bal_ci"][0], 100 * c["d_bal_ci"][1]
    ax.plot([lo, hi], [yv, yv], color=col, lw=1.6)
    ax.plot(100 * c["d_bal"], yv, "o", color=col, ms=4)
    ax.text(1.02, yv, f"{c['nf']} folds", transform=ax.get_yaxis_transform(), va="center", fontsize=6.5)
ax.axvline(0, color="k", lw=0.7, ls=":")
ax.axhline(len(items) - 7.5, color="gray", lw=0.5)
ax.set_yticks(range(len(items))); ax.set_yticklabels([l for l, _ in items][::-1], fontsize=7)
ax.set_xlabel("difference in balanced accuracy, A − B (percentage points)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_forest.pdf")); plt.close(fig)

# ---- fig_ctl
if CT:
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    groups = [("exposed", "exposed test images"), ("unexposed", "unexposed test images")]
    arms = [("leaky", "leaky arm", C2)] + ([("clean", "clean arm", C1)] if "clean" in CT else [])
    wd = 0.36
    for ai, (arm, al, col) in enumerate(arms):
        xs = np.arange(len(groups)) + (ai - (len(arms) - 1) / 2) * wd
        vals = [100 * CT[arm][g_]["bal_acc"] for g_, _ in groups]
        lo = [100 * (CT[arm][g_]["bal_acc"] - CT[arm][g_]["bal_acc_ci"][0]) for g_, _ in groups]
        hi = [100 * (CT[arm][g_]["bal_acc_ci"][1] - CT[arm][g_]["bal_acc"]) for g_, _ in groups]
        ax.bar(xs, vals, width=wd, color=col, label=al, yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.8))
        for x_, v in zip(xs, vals):
            ax.text(x_, 41, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color="white")
    gref = CT["grouped_ref"]
    ax.axhline(100 * gref["bal_acc"], color="k", ls="--", lw=0.9,
               label=f"lesion-grouped protocol ({100 * gref['bal_acc']:.1f})")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{l}\n(n = {CT['leaky'][g_]['n']})" for g_, l in groups])
    ax.set_ylabel("balanced accuracy (%)"); ax.set_ylim(40, 100)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=1)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ctl.pdf")); plt.close(fig)

# ---- fig_reliability
RL = R["reliability"]
fig, ax = plt.subplots(figsize=(3.4, 3.0))
for key, lbl, col, mk in (("main_raw", f"MERIT-Net raw (ECE {CALK['main@5']['raw']['ece15']:.3f})", C1, "o"),
                          ("main_temp", f"MERIT-Net temp.-scaled ({CALK['main@5']['temp']['ece15']:.3f})", C4, "^"),
                          ("main_prior", f"MERIT-Net prior-corrected ({CALK['main@5']['prior']['ece15']:.3f})", C3, "D"),
                          ("meritce_raw", f"MSAP + CE raw ({CALK['merit-ce@5']['raw']['ece15']:.3f})", C2, "s")):
    ax.plot(RL[key]["conf"], RL[key]["acc"], marker=mk, ms=3, lw=1.0, color=col, label=lbl)
ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.8)
ax.set_xlabel("confidence"); ax.set_ylabel("observed accuracy")
ax.legend(frameon=False, fontsize=5.8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_reliability.pdf")); plt.close(fig)

# ---- fig_melnv (same data as before, descriptive labels)
MEL, NV = 4, 5
def load(t):
    return json.load(open(os.path.join(RES, t, f"derm_{t}.json")))
Dm = {t: load(t) for t in set(fus["tags"] + ["boost4", "main"])}
y7 = np.array(Dm["boost4"]["test_y"]); mask = np.isin(y7, (MEL, NV))
ym = (y7[mask] == MEL).astype(int)
def tscore(t):
    P = np.mean([np.array(a["test_p"]) for a in Dm[t]["arrays"]], 0)
    P = P / P.sum(1, keepdims=True)
    return (P[:, MEL] / (P[:, MEL] + P[:, NV] + 1e-12))[mask]
s_f = sum(w * tscore(t) for w, t in zip(fus["weights"], fus["tags"]))
assert abs(roc_auc_score(ym, s_f) - fus["auc"]) < 1e-9
fig, ax = plt.subplots(figsize=(3.5, 3.1))
for lbl, s_, col in ((f"weighted fusion, 7 systems (AUC {roc_auc_score(ym, s_f):.3f})", s_f, C1),
                     (f"P4 EfficientNetV2-S 384² (AUC {roc_auc_score(ym, tscore('boost4')):.3f})", tscore("boost4"), C3),
                     (f"MSAP + CE, matched (AUC {roc_auc_score(ym, tscore('merit-ce')):.3f})", tscore("merit-ce"), C2),
                     (f"MERIT-Net, matched (AUC {roc_auc_score(ym, tscore('main')):.3f})", tscore("main"), C4)):
    fpr, tpr, _ = roc_curve(ym, s_)
    ax.plot(fpr, tpr, lw=1.1, color=col, label=lbl)
yp = youden(fus)
ax.plot(1 - yp["spec"], yp["sens"], "o", ms=5, color=C1, mec="black", mew=0.5, zorder=5,
        label=f"locked threshold (sens {100 * yp['sens']:.1f}, spec {100 * yp['spec']:.1f})")
ax.axhline(0.9, color="gray", ls=":", lw=0.6); ax.axvline(0.1, color="gray", ls=":", lw=0.6)
ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.6)
ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity")
ax.legend(frameon=False, fontsize=5.6, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_melnv.pdf")); plt.close(fig)

# ------------------------------------------------------------ supplement
def tabular(cols, head, rows, cap, lab_, size=r"\scriptsize"):
    return (r"\begin{table}[!ht]" + "\n" + r"\caption{" + cap + "}\n\\label{" + lab_ + "}\n\\centering\n" + size + "\n"
            r"\resizebox{\textwidth}{!}{%" + "\n\\begin{tabular}{" + cols + "}\n\\toprule\n" + head + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
SUP = []
# S1 reference standard by dx in test and train/val
dxo = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
rows = []
for part, sub in (("Test", te), ("Train/validation", trv)):
    ct = pd.crosstab(sub.dx, sub.dx_type).reindex(dxo).fillna(0).astype(int)
    for c in dxo:
        rows.append(f"{part} & {c} & {len(sub[sub.dx == c])} & {sub[sub.dx == c].lesion_id.nunique()} & "
                    + " & ".join(str(ct.loc[c, k]) if k in ct.columns else "0" for k in ("histo", "follow_up", "consensus", "confocal")) + r" \\")
    rows.append(r"\midrule")
SUP.append(tabular("llcccccc", "Partition & Diagnosis & Images & Lesions & Histopathology & Follow-up & Consensus & Confocal",
                   rows[:-1], "Images, lesions and reference standard (number of images per ground-truth type, from the HAM10000 metadata) by diagnosis and partition.", "tab:s1"))
# S2 duplicates
rows = [f"{r.img_a} & {r.les_a} & {r.dx_a} & {r.part_a} & {r.img_b} & {r.les_b} & {r.dx_b} & {r.part_b} & "
        f"{ {'TEST-TRAIN': 'test / train-val', 'cv-cv': 'two CV folds', 'same cv': 'same fold'}[r.span] }".replace("_", r"\_") + r" \\"
        for r in pairs.itertuples()]
SUP.append(tabular("lllllllll", "Image A & Lesion A & Dx & Partition & Image B & Lesion B & Dx & Partition & Span", rows,
                   "The 18 same-lesion image pairs with different lesion identifiers published by Abhishek et al. (2025), located in the partitions of this study.", "tab:s2"))
DS = R["dup_sensitivity"]
rows = []
for key in [f"{t}@3" for t in ORDER] + [f"{t}@5" for t in ORDER[:4]] + ["cnx-cbce@3", "cnx-wce@3", "m-lam01@3", "m-lam0@3"]:
    r = DS["rows"][key]; t, nf = key.split("@")
    rows.append(f"{NAMES.get(t, t) if t not in ('m-lam01', 'm-lam0') else 'MERIT ' + NAMES[t]} & {nf} & {pc(r['bal_full'], 2)} & {pc(r['bal_excl'], 2)} & {pc(r['acc_full'], 2)} & {pc(r['acc_excl'], 2)}" + r" \\")
SUP.append(tabular("lccccc", r"Model & Folds & BalAcc, all 2{,}014 (\%) & BalAcc, 2{,}009 (\%) & Acc, all (\%) & Acc, 2{,}009 (\%)", rows,
                   f"Sensitivity analysis: results with and without the {DS['n_excluded']} test images that form a published same-lesion pair with a training/validation image.", "tab:s2b"))
rows = [f"{PL[i][1]} & {sg(c['d_bal'])} & {c['a_only']} & {c['b_only']} & {pv(c['p'])}" + r" \\" for i, c in enumerate(DS["contrasts"])]
SUP.append(tabular("lcccc", r"Primary contrast & $\Delta$BalAcc & Only A & Only B & McNemar $p$", rows,
                   f"Primary contrasts recomputed after excluding the {DS['n_excluded']} affected test images (compare with Table 4 of the article).", "tab:s2c"))
# S3 secondary contrasts
def nm(t):
    base = t.replace("-s43", "")
    n_ = NAMES.get(base, base)
    if base.startswith("m-") and base != "m-edl-lam01":
        n_ = "MERIT, " + n_
    return n_ + (" (seed 43)" if t.endswith("-s43") else "")
seen, rows = set(), []
for c in R["contrasts_secondary"]:
    k = (c["a"], c["b"])
    if k in seen:
        continue
    seen.add(k)
    rows.append(f"{nm(c['a'])} & {nm(c['b'])} & {c['nf']} & {sg(c['d_bal'])} [{sgci(c['d_bal_ci'])}] & {sg(c['d_acc'])} [{sgci(c['d_acc_ci'])}] & "
                f"{c['a_only']} & {c['b_only']} & {pv(c['p'])} & {pv(c['p_holm'])}" + r" \\")
SUP.append(tabular("llccccccc", r"A & B & Folds & $\Delta$BalAcc (A $-$ B) [95\% CI] & $\Delta$Acc [95\% CI] & Only A & Only B & $p$ & $p_{\mathrm{Holm}}$", rows,
                   "Secondary paired contrasts (fold-matched three-fold ensembles; lesion-cluster bootstrap CIs; exact McNemar test; Holm adjustment within this family). Descriptive.", "tab:s3", r"\tiny"))
# S4 per-class recall
rows = []
for t in ORDER + ["cnx-wce", "cnx-focal", "cnx-cbce", "cnx-edl", "m-gap", "m-attplain", "m-single", "m-fusemean", "m-lam0", "m-lam01", "m-beta9999", "m-edl", "m-effb3", "m-224"] + [x for x in ("m-edl-lam01", "cnx-cbedl-lam01", "main-s43", "m-lam01-s43", "merit-ce-s43") if x in S3]:
    rows.append(f"{nm(t)} & " + " & ".join(pc(x) for x in S3[t]["per_class_recall"]) + f" & {pc(S3[t]['bal_acc'])}" + r" \\")
SUP.append(tabular("lcccccccc", "Configuration & " + " & ".join(CLASSES) + " & BalAcc", rows,
                   r"Per-class recall (\%) of the three-fold ensembles on the frozen test set (test images per class: akiec 68, bcc 100, bkl 205, df 22, mel 222, nv 1{,}363, vasc 34).", "tab:s4"))
# S5 all melnv systems x points
rows = []
SN = {"p4": "P4", "p24": "P2 + P4 (equal weights)", "p42m": "P4 + P2 + MSAP+CE (equal weights)", "fus": "Weighted fusion (7 systems)"}
for sid in ("p4", "p24", "p42m", "fus"):
    s = MSYS[sid]
    for p in s["points"]:
        pn = p["point"].replace("valSpec", "val. specificity $\\ge$ ").replace("valYouden", "val. Youden")
        rows.append(f"{SN[sid]} & {pn} & {p['thr']:.4f} & {p['tp']} / {p['fn']} & {p['tn']} / {p['fp']} & {pc(p['sens'])} & {pc(p['spec'])} & {pc(p['acc'])} & {f3(s['auc'])}" + r" \\")
    rows.append(r"\midrule")
SUP.append(tabular("llccccccc", r"System & Validation-defined operating point & Threshold & TP / FN & TN / FP & Sens (\%) & Spec (\%) & Acc (\%) & AUC", rows[:-1],
                   "All melanoma-versus-nevus systems and operating points that were evaluated on the test subset during the exploratory stage (4 systems $\\times$ 7 validation-defined thresholds). The article reports the validation-Youden point.", "tab:s5"))
# S6 PH2
rows = []
for t in ORDER + ["cnx-wce", "cnx-focal", "cnx-cbce", "cnx-edl", "m-gap", "m-attplain", "m-single", "m-fusemean", "m-lam0", "m-lam01", "m-beta9999", "m-edl", "m-effb3", "m-224"]:
    rows.append(f"{nm(t)} & 3 & {f3(S3[t]['ph2_auc'])} & {ci3(S3[t]['ph2_auc_ci'])}" + r" \\")
rows.append(r"\midrule")
for t in ORDER[:4]:
    rows.append(f"{nm(t)} & 5 & {f3(S5[t]['ph2_auc'])} & {ci3(S5[t]['ph2_auc_ci'])}" + r" \\")
rows.append(r"\midrule")
for t, pid, desc, task in BR:
    b = R["boost"][t]
    rows.append(f"{pid}: {desc} & 5 & {f3(b['ph2_auc'])} & {ci3(b['ph2_auc_ci'])}" + r" \\")
rows.append(r"\midrule")
for sid, n_ in (("fus", "Weighted fusion of 7 systems (mel-vs-nv score)"),):
    rows.append(f"{n_} & 5 & {f3(MSYS[sid]['ph2_auc'])} & --" + r" \\")
SUP.append(tabular("lccc", r"Model & Folds & \PHtwo{} AUC & 95\% CI", rows,
                   r"Zero-shot \PHtwo{} melanoma-versus-nevus AUC of the fold ensembles (200 images, 40 melanomas; bootstrap over images).", "tab:s6"))
with open(os.path.join(PAPER, "supp_tables.tex"), "w", encoding="utf-8") as f:
    f.write("% GENERATED by rev_make_assets.py\n" + "\n\\clearpage\n".join(SUP))

# ------------------------------------------------------------ write numbers
with open(os.path.join(PAPER, "numbers_rev.tex"), "w", encoding="utf-8") as f:
    f.write("% GENERATED by rev_make_assets.py\n")
    for k, v in NUM.items():
        f.write(f"\\defnum{{{k}}}{{{v}}}\n")
used = set()
for fn in ("main.tex", "results.tex"):
    used |= set(re.findall(r"\\N\{([^}]+)\}", open(os.path.join(PAPER, fn), encoding="utf-8").read()))
mt = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
ab = mt[mt.index(r"\section*{Abstract}"):mt.index(r"\paragraph{Keywords:}")]
ab = re.sub(r"\\N\{([^}]+)\}", lambda m: "X", ab)
ab = re.sub(r"\\(begin|end)\{revised\}|\\section\*\{Abstract\}", " ", ab)
ab = re.sub(r"\\[a-zA-Z]+\*?|[{}$\\]", " ", ab).replace("---", " ")
nw = len([w for w in ab.split() if re.search(r"[A-Za-z0-9]", w)])
NUM["resp.abs_words"] = str(nw)
with open(os.path.join(PAPER, "numbers_rev.tex"), "a", encoding="utf-8") as f:
    f.write(f"\\defnum{{resp.abs_words}}{{{nw}}}\n")
print("abstract words:", nw)
for fn in ("response.tex", "narrative_rev.tex"):
    if os.path.exists(os.path.join(PAPER, fn)):
        used |= set(re.findall(r"\\N\{([^}]+)\}", open(os.path.join(PAPER, fn), encoding="utf-8").read()))
nar = os.path.join(PAPER, "narrative_rev.tex")
defined = set(NUM)
if os.path.exists(nar):
    defined |= set(re.findall(r"\\defnum\{([^}]+)\}", open(nar, encoding="utf-8").read()))
missing = sorted(used - defined)
print("numbers:", len(NUM), "| used in text:", len(used), "| MISSING:", missing)
