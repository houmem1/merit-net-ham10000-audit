# -*- coding: utf-8 -*-
"""Complexite mesuree des variantes MERIT (params, mem pic, latence).
Les classes modele sont identiques a kernel_body.py (copie verifiee par
l'audit). Mesures reelles GPU local -> results/complexity.json"""
import json
import os
import sys
import time

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
K = 7
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))

class AttnPool(nn.Module):
    def __init__(self, c, datt=128, gated=True):
        super().__init__()
        self.V = nn.Linear(c, datt)
        self.U = nn.Linear(c, datt) if gated else None
        self.w = nn.Linear(datt, 1)
    def forward(self, f):
        h = torch.tanh(self.V(f))
        if self.U is not None:
            h = h * torch.sigmoid(self.U(f))
        a = torch.softmax(self.w(h), dim=1)
        return (a * f).sum(1), a.squeeze(-1)

class MERIT(nn.Module):
    def __init__(self, backbone="convnext_tiny", scales=(1, 2, 3), d=256):
        super().__init__()
        self.bb = timm.create_model(backbone, pretrained=False,
                                    features_only=True, out_indices=scales)
        chs = self.bb.feature_info.channels()
        ns = len(chs)
        self.pools = nn.ModuleList([AttnPool(c) for c in chs])
        self.projs = nn.ModuleList([nn.Linear(c, d) for c in chs])
        self.gate = (nn.Sequential(nn.Linear(ns * d, d), nn.GELU(),
                                   nn.Linear(d, ns)) if ns > 1 else None)
        self.ns = ns
        self.neck = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3))
        self.head = nn.Linear(d, K)
    def forward(self, x):
        feats = self.bb(x)
        vs = []
        for f, pool, proj, c in zip(feats, self.pools, self.projs,
                                    self.bb.feature_info.channels()):
            assert f.shape[1] == c, (f.shape, c)
            t = f.flatten(2).transpose(1, 2)
            v, _ = pool(t)
            vs.append(F.gelu(proj(v)))
        V = torch.stack(vs, 1)
        if self.gate is not None:
            beta = torch.softmax(self.gate(V.flatten(1)), -1)
            v = (beta.unsqueeze(-1) * V).sum(1)
        else:
            v = V.mean(1)
        return self.head(self.neck(v))

class Plain(nn.Module):
    def __init__(self, backbone, d=256, size=256):
        super().__init__()
        kw = {}
        if backbone.startswith(("deit3", "vit_")):
            kw["img_size"] = size
        self.net = timm.create_model(backbone, pretrained=False,
                                     num_classes=0, **kw)
        self.proj = nn.Linear(self.net.num_features, d)
        self.neck = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3))
        self.head = nn.Linear(d, K)
    def forward(self, x):
        return self.head(self.neck(F.gelu(self.proj(self.net(x)))))

def measure(name, model, size=256, bs=16, iters=30):
    model = model.to(DEV).eval()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.randn(bs, 3, size, size, device=DEV)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda"):
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters / bs * 1000.0
    mem = torch.cuda.max_memory_allocated() / 2**20
    print(f"{name}: {n_par:.1f}M params | {dt:.2f} ms/img | {mem:.0f} MiB",
          flush=True)
    del model, x
    torch.cuda.empty_cache()
    return dict(params_M=round(n_par, 2), ms_per_img=round(dt, 2),
                peak_MiB=round(mem, 0))

if __name__ == "__main__":
    out = {}
    out["merit_cnx"] = measure("merit_cnx", MERIT())
    out["merit_single"] = measure("merit_single", MERIT(scales=(3,)))
    out["merit_effb3"] = measure("merit_effb3",
                                 MERIT(backbone="tf_efficientnet_b3",
                                       scales=(2, 3, 4)))
    out["plain_cnx"] = measure("plain_cnx", Plain("convnext_tiny"))
    out["r50"] = measure("r50", Plain("resnet50"))
    out["effb3"] = measure("effb3", Plain("tf_efficientnet_b3"))
    out["dense121"] = measure("dense121", Plain("densenet121"))
    out["deit3s"] = measure("deit3s", Plain("deit3_small_patch16_224"))
    out["swinv2t"] = measure("swinv2t", Plain("swinv2_tiny_window8_256"))
    out["_env"] = dict(gpu=torch.cuda.get_device_name(0), bs=16, size=256,
                       amp=True)
    with open(os.path.join(HERE, "..", "results", "complexity.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved complexity.json")
