#!/usr/bin/env python3
"""OpenDC-Bench 显著性检验：Diebold-Mariano + 移动块 bootstrap 置信区间。

读取 preds/{tag}.npz (preds[N,144], trues[N,144])，逐窗口计算每个 horizon 的
绝对误差损失 loss_i^h = mean_{j<h} |pred_ij - true_ij|，得到长度 N 的配对损失序列。

- MAE 与 95% 置信区间：对每个模型的损失序列做移动块 bootstrap。
- DM 检验：比较两模型损失差 d_i=loss_A-loss_B，用 Newey-West(HAC, lag=h-1) 方差 +
  Harvey-Leybourne-Newbold 小样本校正，双侧 t 分布 p 值。
  窗口高度重叠 -> 必须用 HAC / 块 bootstrap 处理自相关。
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
from scipy import stats

HORIZONS = [1, 6, 36, 144]


def load_loss(preds_dir, tag):
    d = np.load(Path(preds_dir) / f"{tag}.npz")
    preds, trues = d["preds"], d["trues"]
    return preds, trues


def per_window_loss(preds, trues, h):
    # 每窗口在 horizon h 上的平均绝对误差
    return np.abs(preds[:, :h] - trues[:, :h]).mean(axis=1)


def dm_test(loss_a, loss_b, h):
    """Diebold-Mariano，HAC 方差 (lag=h-1) + HLN 小样本校正。返回 (stat, pval, mean_diff)。"""
    d = loss_a - loss_b
    n = len(d)
    dbar = d.mean()
    if np.allclose(d, 0.0):
        return 0.0, 1.0, 0.0
    # Newey-West 长期方差
    gamma0 = np.mean((d - dbar) ** 2)
    lag = max(h - 1, 0)
    s = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        s += 2 * w * cov
    var_dbar = s / n
    if var_dbar <= 0:
        return np.nan, np.nan, dbar
    dm = dbar / np.sqrt(var_dbar)
    # HLN 小样本校正
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * corr
    pval = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(pval), float(dbar)


def block_bootstrap_ci(loss, block=50, n_boot=2000, alpha=0.05, seed=0):
    """移动块 bootstrap，返回损失均值的 (lo, hi) 置信区间。"""
    rng = np.random.default_rng(seed)
    n = len(loss)
    nblocks = int(np.ceil(n / block))
    max_start = n - block
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=nblocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]
        means[b] = loss[idx].mean()
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ali", "mit"], required=True)
    ap.add_argument("--preds-dir", default="preds")
    ap.add_argument("--out", default=None)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    suf = "__" + args.dataset
    # 参与比较的模型（tag 不含后缀部分在此拼）
    models = {
        "naive-last":     f"baseline__naive-last{suf}",
        "ma-36":          f"baseline__ma-36{suf}",
        "seasonal-naive": f"baseline__seasonal-naive{suf}",
        "bolt-base":      f"bolt-base{suf}",
        "bolt-lora":      f"bolt-lora{suf}",
    }
    # 加载 + 校验 trues 一致
    data = {}
    ref_trues = None
    for name, tag in models.items():
        p, t = load_loss(args.preds_dir, tag)
        if ref_trues is None:
            ref_trues = t
        else:
            assert t.shape == ref_trues.shape and np.allclose(t, ref_trues, atol=1e-3), \
                f"{name} trues mismatch! windows not aligned"
        data[name] = p
    N = ref_trues.shape[0]
    print(f"[sig] dataset={args.dataset}  N={N} windows  models={list(models)}", flush=True)

    result = {"dataset": args.dataset, "n_windows": N, "horizons": HORIZONS, "mae": {}, "ci95": {}, "dm": {}}

    # MAE + bootstrap CI
    for name in models:
        result["mae"][name] = {}
        result["ci95"][name] = {}
        for h in HORIZONS:
            loss = per_window_loss(data[name], ref_trues, h)
            result["mae"][name][f"h={h}"] = float(loss.mean())
            lo, hi = block_bootstrap_ci(loss, block=args.block, n_boot=args.n_boot, seed=h)
            result["ci95"][name][f"h={h}"] = [round(lo, 4), round(hi, 4)]

    # DM 关键对比
    comparisons = [
        ("bolt-base", "naive-last"),
        ("bolt-lora", "naive-last"),
        ("bolt-lora", "bolt-base"),
        ("bolt-base", "ma-36"),
    ]
    for a, b in comparisons:
        key = f"{a}_vs_{b}"
        result["dm"][key] = {}
        for h in HORIZONS:
            la = per_window_loss(data[a], ref_trues, h)
            lb = per_window_loss(data[b], ref_trues, h)
            stat, pval, dbar = dm_test(la, lb, h)
            result["dm"][key][f"h={h}"] = {"dm": round(stat, 3), "p": round(pval, 4),
                                            "mean_diff": round(dbar, 4)}

    # 打印
    print("\n" + "=" * 78)
    print(f"  MAE (95% block-bootstrap CI)  —  dataset={args.dataset}, N={N}")
    print("=" * 78)
    print(f"  {'model':<16}" + "".join(f"{'h='+str(h):>15}" for h in HORIZONS))
    for name in models:
        row = f"  {name:<16}"
        for h in HORIZONS:
            m = result["mae"][name][f"h={h}"]
            lo, hi = result["ci95"][name][f"h={h}"]
            row += f"{m:>7.3f}[{lo:.2f},{hi:.2f}]"[:15].rjust(15)
        print(row)

    print("\n" + "=" * 78)
    print("  Diebold-Mariano (HLN 校正, p<0.05 记 *; mean_diff<0 表示前者更优)")
    print("=" * 78)
    for key, hd in result["dm"].items():
        print(f"  {key}")
        for h in HORIZONS:
            e = hd[f"h={h}"]
            sig = " *" if e["p"] < 0.05 else "  "
            print(f"    h={h:<4} DM={e['dm']:>7.3f}  p={e['p']:<7.4f}{sig}  Δloss={e['mean_diff']:+.4f}")
    print("=" * 78, flush=True)

    out = args.out or f"results/significance_{args.dataset}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[sig] saved {out}", flush=True)


if __name__ == "__main__":
    main()
