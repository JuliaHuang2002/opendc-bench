#!/usr/bin/env python3
"""OpenDC-Bench 通用 zero-shot 评测脚手架"""
import os, sys, json, time, argparse, logging, warnings
from pathlib import Path
import numpy as np
import torch

warnings.filterwarnings("ignore")
for name in ["transformers", "chronos", "huggingface_hub", "torch"]:
    logging.getLogger(name).setLevel(logging.ERROR)

HORIZONS = [1, 6, 36, 144]
DEFAULT_LOOKBACK = 288
DEFAULT_N_WINDOWS = 1297


def load_test_series(data_path="data/test.npy"):
    arr = np.load(data_path).astype(np.float32)
    if arr.ndim > 1:
        arr = arr.squeeze()
    print(f"[data] {data_path}: shape={arr.shape}, mean={arr.mean():.2f}, std={arr.std():.2f}", flush=True)
    return arr


def build_windows(series, lookback, max_horizon, n_windows):
    last_start = len(series) - lookback - max_horizon
    if n_windows >= last_start + 1:
        return np.arange(0, last_start + 1)
    return np.linspace(0, last_start, n_windows).astype(int)


def evaluate(predict_fn, series, lookback, n_windows):
    max_h = max(HORIZONS)
    starts = build_windows(series, lookback, max_h, n_windows)
    print(f"[eval] {len(starts)} windows, lookback={lookback}, horizons={HORIZONS}", flush=True)
    preds, trues = [], []
    t0 = time.time()
    for i, s in enumerate(starts):
        ctx = series[s:s + lookback]
        tgt = series[s + lookback:s + lookback + max_h]
        pred = np.asarray(predict_fn(ctx)).reshape(-1)[:max_h]
        if len(pred) < max_h:
            pred = np.concatenate([pred, np.full(max_h - len(pred), pred[-1])])
        preds.append(pred)
        trues.append(tgt)
        if (i + 1) % 100 == 0 or i == len(starts) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"[eval] {i+1}/{len(starts)} ({rate:.2f} win/s, ETA {(len(starts)-i-1)/rate:.0f}s)", flush=True)
    preds = np.array(preds); trues = np.array(trues)
    if os.environ.get("DUMP_PREDS"):
        os.makedirs("preds", exist_ok=True)
        model_tag = os.environ.get("MODEL_TAG", "unknown")
        np.savez_compressed(f"preds/{model_tag}.npz", preds=preds, trues=trues)
        print(f"[dump] saved preds/{model_tag}.npz  preds{preds.shape}  trues{trues.shape}", flush=True)
    mae = {f"h={h}": float(np.abs(preds[:, :h] - trues[:, :h]).mean()) for h in HORIZONS}
    return mae, time.time() - t0


# ============ Adapters ============
def make_chronos_t5(model_name, device):
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(model_name, device_map=device, torch_dtype=torch.float32)
    max_h = max(HORIZONS)
    def predict(ctx):
        x = torch.tensor(ctx, dtype=torch.float32)
        out = pipe.predict(inputs=x, prediction_length=max_h, num_samples=20)
        return out[0].median(dim=0).values.cpu().numpy()
    return predict


def make_chronos_bolt(model_name, device):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(model_name, device_map=device, torch_dtype=torch.float32)
    max_h = max(HORIZONS)
    def predict(ctx):
        x = torch.tensor(ctx, dtype=torch.float32)
        quantiles, mean = pipe.predict_quantiles(inputs=x, prediction_length=max_h, quantile_levels=[0.1, 0.5, 0.9])
        return mean[0].cpu().numpy()
    return predict


def make_timesfm(model_name, device):
    import timesfm
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu" if device.startswith("cuda") else "cpu",
            per_core_batch_size=32, horizon_len=max(HORIZONS), context_len=DEFAULT_LOOKBACK,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_name),
    )
    def predict(ctx):
        point_fcst, _ = tfm.forecast(inputs=[ctx.astype(np.float32)], freq=[0])
        return np.asarray(point_fcst[0])
    return predict


def make_moirai(model_name, device):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    max_h = max(HORIZONS)
    module = MoiraiModule.from_pretrained(model_name)
    model = MoiraiForecast(
        module=module, prediction_length=max_h, context_length=DEFAULT_LOOKBACK,
        patch_size=16, num_samples=20, target_dim=1,
        feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    ).to(device)
    model.eval()
    def predict(ctx):
        past = torch.tensor(ctx, dtype=torch.float32, device=device).view(1, -1, 1)
        past_obs = torch.ones_like(past, dtype=torch.bool)
        past_pad = torch.zeros_like(past[..., 0], dtype=torch.bool)
        with torch.no_grad():
            samples = model(past_target=past, past_observed_target=past_obs, past_is_pad=past_pad)
        return samples[0].median(dim=0).values.cpu().numpy()
    return predict


def make_lag_llama(model_name, device):
    # PyTorch 2.8: force weights_only=False globally (Lightning's internal load needs it)
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)
    torch.load = _patched_load

    from huggingface_hub import snapshot_download
    from lag_llama.gluon.estimator import LagLlamaEstimator
    from gluonts.dataset.common import ListDataset
    import pandas as pd

    # Prefer local checkpoint if present, else download
    local_ckpt = "checkpoints/lag-llama.ckpt"
    ckpt_path = local_ckpt if os.path.exists(local_ckpt) else os.path.join(snapshot_download(model_name), "lag-llama.ckpt")

    # Read exact architecture from checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt["hyper_parameters"]
    ea = hp.get("model_kwargs") or hp.get("estimator_args") or hp
    print(f"[lag-llama] loaded arch: input_size={ea.get('input_size')}, "
          f"n_layer={ea.get('n_layer')}, n_head={ea.get('n_head')}, "
          f"n_embd_per_head={ea.get('n_embd_per_head')}, "
          f"time_feat={ea.get('time_feat')}", flush=True)

    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=max(HORIZONS),
        context_length=DEFAULT_LOOKBACK,
        input_size=ea["input_size"],
        n_layer=ea["n_layer"],
        n_embd_per_head=ea["n_embd_per_head"],
        n_head=ea["n_head"],
        scaling=ea.get("scaling", "mean"),
        time_feat=ea.get("time_feat", True),
        rope_scaling=ea.get("rope_scaling", None),
        device=torch.device(device), batch_size=1, num_parallel_samples=20,
    )
    predictor = estimator.create_predictor(estimator.create_transformation(), estimator.create_lightning_module())
    max_h = max(HORIZONS)
    def predict(ctx):
        ds = ListDataset([{"start": pd.Period("2024-01-01", freq="10min"), "target": ctx.astype(np.float32)}], freq="10min")
        return np.asarray(list(predictor.predict(ds))[0].quantile(0.5))[:max_h]
    return predict


ADAPTERS = {
    "amazon/chronos-t5-": make_chronos_t5,
    "amazon/chronos-bolt-": make_chronos_bolt,
    "google/timesfm-": make_timesfm,
    "Salesforce/moirai-": make_moirai,
    "time-series-foundation-models/Lag-Llama": make_lag_llama,
}


def get_adapter(model_name):
    for prefix, factory in ADAPTERS.items():
        if model_name.startswith(prefix) or model_name == prefix:
            return factory
    raise ValueError(f"No adapter for {model_name}. Known: {list(ADAPTERS.keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("--data", default="data/alibaba_10min_test_v2.npy")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-windows", type=int, default=DEFAULT_N_WINDOWS)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    out_json = Path(args.results_dir) / f"{args.model_name.replace('/', '__')}.json"
    print(f"[run] model={args.model_name} device={args.device} n_windows={args.n_windows} lookback={args.lookback}", flush=True)

    series = load_test_series(args.data)
    factory = get_adapter(args.model_name)
    t_load = time.time()
    predict_fn = factory(args.model_name, args.device)
    print(f"[run] model loaded in {time.time()-t_load:.1f}s", flush=True)

    mae, runtime = evaluate(predict_fn, series, args.lookback, args.n_windows)

    result = {
        "model": args.model_name, "device": args.device,
        "n_windows": args.n_windows, "lookback": args.lookback,
        "mae": mae, "runtime_sec": round(runtime, 1),
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 56)
    print(f"  Result: {args.model_name}")
    print("=" * 56)
    print(f"  Horizon     MAE")
    print(f"  ---------   ------")
    for h in HORIZONS:
        print(f"  h={h:<6}    {mae[f'h={h}']:.3f}")
    print(f"\n  runtime: {runtime:.0f}s ({runtime/60:.1f} min)")
    print(f"  saved:   {out_json}")
    print("=" * 56, flush=True)


if __name__ == "__main__":
    main()
