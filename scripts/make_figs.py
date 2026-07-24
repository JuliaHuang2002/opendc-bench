#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build MAE-vs-horizon error-bar figures from significance JSONs.

Run from repo root:  python3 scripts/make_figs.py
Reads:   results/significance_ali.json, results/significance_mit.json
Writes:  docs/mae_vs_horizon_ali.png, docs/mae_vs_horizon_mit.png
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "docs")
os.makedirs(OUT, exist_ok=True)

# CJK font: try macOS system fonts first, then common Linux locations
for fp in [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        try:
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        except Exception:
            pass
        break
matplotlib.rcParams["axes.unicode_minus"] = False

ALI = json.load(open(os.path.join(RESULTS, "significance_ali.json")))
MIT = json.load(open(os.path.join(RESULTS, "significance_mit.json")))

HZ = [1, 6, 36, 144]
MODELS = [
    ("naive-last",     "\u6301\u4e45\u6027\u57fa\u7ebf naive-last", "#888888", "o", "--"),
    ("ma-36",          "\u6ed1\u52a8\u5747\u503c ma-36",           "#4c9f70", "s", "--"),
    ("bolt-base",      "Chronos-Bolt \u9884\u8bad\u7ec3",          "#2f6db5", "^", "-"),
    ("bolt-lora",      "Chronos-Bolt LoRA \u5fae\u8c03",           "#c0392b", "D", "-"),
]

def plot(ds, title, fname):
    mae, ci = ds["mae"], ds["ci95"]
    x = np.arange(len(HZ))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    off = np.linspace(-0.12, 0.12, len(MODELS))
    for i, (key, lab, col, mk, ls) in enumerate(MODELS):
        ys = [mae[key][f"h={h}"] for h in HZ]
        lo = [mae[key][f"h={h}"] - ci[key][f"h={h}"][0] for h in HZ]
        hi = [ci[key][f"h={h}"][1] - mae[key][f"h={h}"] for h in HZ]
        ax.errorbar(x + off[i], ys, yerr=[lo, hi], label=lab, color=col,
                    marker=mk, linestyle=ls, capsize=3, markersize=6, linewidth=1.6, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in HZ])
    ax.set_xlabel("\u9884\u6d4b\u6b65\u957f\uff08\u70b9\u6570\uff09")
    ax.set_ylabel("\u5e73\u5747\u7edd\u5bf9\u8bef\u5dee MAE\uff08\u542b 95% \u7f6e\u4fe1\u533a\u95f4\uff09")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p

p1 = plot(ALI, "\u963f\u91cc GPU trace (N=1009)\uff1aMAE \u968f\u9884\u6d4b\u6b65\u957f\u53d8\u5316", "mae_vs_horizon_ali.png")
p2 = plot(MIT, "MIT SuperCloud (N=1297)\uff1aMAE \u968f\u9884\u6d4b\u6b65\u957f\u53d8\u5316", "mae_vs_horizon_mit.png")
print("FIG", p1)
print("FIG", p2)
