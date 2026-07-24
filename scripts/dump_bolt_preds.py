#!/usr/bin/env python3
"""导出 Chronos-Bolt (zero-shot 或 LoRA-ft) 的逐窗口 preds/trues，用于配对显著性检验。

复用 run_zeroshot.evaluate + DUMP_PREDS 机制，窗口采样与基线/其它模型完全一致。
LoRA 复现逻辑与 finetune_bolt_lora.py 保持一致：加载 base -> 挂 adapter -> merge_and_unload。
"""
import os, sys, argparse, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_zeroshot import evaluate, HORIZONS, DEFAULT_LOOKBACK, DEFAULT_N_WINDOWS, load_test_series

BASE_MODEL = "amazon/chronos-bolt-base"
MAX_H = max(HORIZONS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True, help="MODEL_TAG，决定 preds/{tag}.npz 文件名")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 目录；不给则纯 zero-shot base")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n-windows", type=int, default=DEFAULT_N_WINDOWS)
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from chronos import BaseChronosPipeline
    print(f"[model] loading {BASE_MODEL} on {args.device}", flush=True)
    pipe = BaseChronosPipeline.from_pretrained(BASE_MODEL, device_map=args.device, torch_dtype=torch.float32)

    if args.adapter:
        from peft import PeftModel
        print(f"[lora] attaching adapter {args.adapter}", flush=True)
        peft_model = PeftModel.from_pretrained(pipe.model, args.adapter)
        merged = peft_model.merge_and_unload()
        merged.eval()
        pipe.model = merged
        print("[lora] merged.", flush=True)

    def predict(ctx):
        x = torch.tensor(ctx, dtype=torch.float32)
        with torch.no_grad():
            _, mean = pipe.predict_quantiles(inputs=x, prediction_length=MAX_H, quantile_levels=[0.1, 0.5, 0.9])
        return mean[0].cpu().numpy()

    os.environ["DUMP_PREDS"] = "1"
    os.environ["MODEL_TAG"] = args.tag

    series = load_test_series(args.data)
    mae, runtime = evaluate(predict, series, args.lookback, args.n_windows)
    print(f"[done] {args.tag}: " + "  ".join(f"{k}={v:.4f}" for k, v in mae.items()) + f"  runtime={runtime:.1f}s", flush=True)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.results_dir) / f"{args.tag}.json", "w") as f:
        json.dump({"model": args.tag, "adapter": args.adapter, "n_windows": args.n_windows,
                   "lookback": args.lookback, "mae": mae, "runtime_sec": round(runtime, 1)}, f, indent=2)


if __name__ == "__main__":
    main()
