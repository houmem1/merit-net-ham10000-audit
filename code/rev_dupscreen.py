# -*- coding: utf-8 -*-
"""Revision: cross-partition near-duplicate screen that does NOT rely on
lesion_id. pHash (32x32 DCT, 64 bit) AND dHash (9x8, 64 bit) on the 256px
cache, test images vs train/val images, minimum over the 8 dihedral
transforms of the test image. Candidates = both Hamming distances <= THR.
-> results/revision/dupscreen.json (+ montage of candidates)"""
import json, os, sys
import cv2, numpy as np, pandas as pd
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
X = np.load(os.path.join(RES, "local_input", "ham_256.npy"), mmap_mode="r")
lab = pd.read_csv(os.path.join(RES, "revision", "split_assignments.csv"))
te = np.where(lab.partition.values == "test")[0]
tr = np.where(lab.partition.values != "test")[0]

def phash(g):
    d = cv2.dct(cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8]
    return (d > np.median(d)).ravel()
def dhash(g):
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (s[:, 1:] > s[:, :-1]).ravel()
def pack(b):
    return np.packbits(b).view(np.uint64)[0]
def dihedral(g):
    out = []
    for k in range(4):
        r = np.rot90(g, k)
        out += [np.ascontiguousarray(r), np.ascontiguousarray(r[:, ::-1])]
    return out
gray = lambda i: cv2.cvtColor(np.ascontiguousarray(X[i]), cv2.COLOR_RGB2GRAY)
Ptr = np.array([pack(phash(gray(i))) for i in tr]); Dtr = np.array([pack(dhash(gray(i))) for i in tr])
POP = np.array([bin(i).count("1") for i in range(65536)], np.uint8)
def ham(a, B):
    x = np.bitwise_xor(a, B)
    return (POP[x & 0xFFFF] + POP[(x >> 16) & 0xFFFF] + POP[(x >> 32) & 0xFFFF] + POP[(x >> 48) & 0xFFFF])
THR = int(sys.argv[1]) if len(sys.argv) > 1 else 10
cands = []
best = []
for n, i in enumerate(te):
    g = gray(i)
    bp = None
    for t, gg in enumerate(dihedral(g)):
        hp = ham(pack(phash(gg)), Ptr).astype(int); hd = ham(pack(dhash(gg)), Dtr).astype(int)
        m = np.maximum(hp, hd)
        j = int(m.argmin())
        if bp is None or m[j] < bp[0]:
            bp = (int(m[j]), j, t, int(hp[j]), int(hd[j]))
        for j2 in np.where((hp <= THR) & (hd <= THR))[0]:
            cands.append((int(i), int(tr[j2]), t, int(hp[j2]), int(hd[j2])))
    best.append(bp[0])
seen = {}
for c in cands:
    k = (c[0], c[1])
    if k not in seen or max(c[3], c[4]) < max(seen[k][3], seen[k][4]):
        seen[k] = c
rows = []
for (i, j), c in sorted(seen.items(), key=lambda kv: max(kv[1][3], kv[1][4])):
    rows.append(dict(test_img=lab.image_id[i], train_img=lab.image_id[j], test_lesion=lab.lesion_id[i], train_lesion=lab.lesion_id[j],
                     test_dx=lab.dx[i], train_dx=lab.dx[j], transform=c[2], phash=c[3], dhash=c[4], ti=i, tj=j))
best = np.array(best)
out = dict(thr=THR, n_test=len(te), n_train=len(tr), n_candidates=len(rows), candidates=rows,
           min_dist_quantiles={str(q): float(np.quantile(best, q)) for q in (0, 0.001, 0.01, 0.05, 0.5)},
           n_exact=int(sum(1 for r in rows if r["phash"] == 0 and r["dhash"] == 0)))
json.dump(out, open(os.path.join(RES, "revision", "dupscreen.json"), "w"), indent=1)
print("candidates <=%d:" % THR, len(rows), "| min-dist quantiles", out["min_dist_quantiles"])
for r in rows[:40]:
    print(r["test_img"], r["train_img"], r["test_lesion"], r["train_lesion"], r["test_dx"], r["train_dx"], "t%d p%d d%d" % (r["transform"], r["phash"], r["dhash"]))
if rows:
    k = min(len(rows), 12)
    canvas = np.concatenate([np.concatenate([np.ascontiguousarray(X[r["ti"]]), np.ascontiguousarray(X[r["tj"]])], 1) for r in rows[:k]], 0)
    cv2.imwrite(os.path.join(RES, "revision", "dupscreen_candidates.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
