#!/usr/bin/env python3
"""OpenDC-Bench 朴素基线评测

复用 run_zeroshot.evaluate 的完全相同窗口采样（build_windows: linspace 起点、
lookback、n_windows），保证基线与基础模型逐窗口 1:1 对齐，可用于配对显著性检验。

每个基线用 DUMP_PREDS 机制导出 preds/{tag}.npz (preds[n_windows,144], trues[...])。
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_zeroshot import evaluate, HORIZONS, DEFAULT_LOOKBACK, DEFAULT_N_WINDOWS, load_test_series

MAX_H = max(HORIZONS)


def make_naive():
    # 持久性：重复上下文最后一个值
    def predict(ctx):
        return np.full(MAX_H, ctx[-1], dtype=np.float32)
    return predict


def make_mean():
    # 上下文均值
    def predict(ctx):
        return np.full(MAX_H, float(np.mean(ctx)), dtype=np.float32)
    return predict


def make_ma(k):
    # 最近 k 步滑动平均，向前平推
    def predict(ctx):
        w = ctx[-k:] if len(ctx) >= k else ctx
        return np.full(MAX_H, float(np.mean(w)), dtype=np.float32)
    return predict


def make_seasonal(period):
    # 季节朴素：以最近一个周期的形态平铺到未来 MAX_H 步
    def predict(ctx):
        if len(ctx) < period:
            return np.full(MAX_H, ctx[-1], dtype=np.float32)
        last_period = ctx[-period:]
        reps = int(np.ceil(MAX_H / period))
        tiled = np.tile(last_period, reps)[:MAX_H]
        return tiled.astype(np.float32)
    return predict


BASELINES = {
    "naive-last":       lambda a: make_naive(),
    "context-mean":     lambda a: make_mean(),
    "ma-6":             lambda a: make_ma(6),
    "ma-36":            lambda a: make_ma(36),
    "seasonal-naive":   lambda a: make_seasonal(a.season_period),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/alibaba_10min_test_v2.npy")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--preds-dir", default="preds")
    ap.add_argument("--n-windows", type=int, default=DEFAULT_N_WINDOWS)
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--season-period", type=int, default=144)
    ap.add_argument("--dump-preds", action="store_true")
    ap.add_argument("--only", default=None, help="逗号分隔仅跑部分基线")
    ap.add_argument("--tag-suffix", default="", help="追加到 MODEL_TAG，用于区分数据集避免 npz 覆盖")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    if args.dump_preds:
        os.makedirs(args.preds_dir, exist_ok=True)

    series = load_test_series(args.data)
    names = list(BASELINES.keys())
    if args.only:
        names = [n for n in names if n in set(args.only.split(","))]

    all_res = {}
    for name in names:
        predict_fn = BASELINES[name](args)
        tag = f"baseline__{name}{args.tag_suffix}"
        if args.dump_preds:
            os.environ["DUMP_PREDS"] = "1"
            os.environ["MODEL_TAG"] = tag
        print(f"\n[baseline] === {name} (period={args.season_period}) ===", flush=True)
        t0 = time.time()
        mae, runtime = evaluate(predict_fn, series, args.lookback, args.n_windows)
        result = {
            "model": tag, "device": "cpu",
            "n_windows": args.n_windows, "lookback": args.lookback,
            "mae": mae, "runtime_sec": round(runtime, 2),
        }
        out_json = Path(args.results_dir) / f"{tag}.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        all_res[name] = mae
        print(f"[baseline] {name}: " + "  ".join(f"{k}={v:.3f}" for k, v in mae.items()), flush=True)
        print(f"[baseline] saved {out_json}", flush=True)

    print("\n" + "=" * 64)
    print(f"  Baseline summary ({args.data})")
    print("=" * 64)
    hdr = "  " + f"{'model':<18}" + "".join(f"{h:>10}" for h in [f'h={x}' for x in HORIZONS])
    print(hdr)
    for name, mae in all_res.items():
        row = "  " + f"{name:<18}" + "".join(f"{mae[f'h={h}']:>10.3f}" for h in HORIZONS)
        print(row)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
