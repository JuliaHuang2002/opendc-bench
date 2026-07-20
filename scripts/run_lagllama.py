"""Lag-Llama zero-shot evaluation on Alibaba GPU trace."""
import sys, torch

# ─── PyTorch 2.8 workaround ───
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

sys.path.insert(0, "scripts")
from run_zeroshot import evaluate, HORIZONS, DEFAULT_LOOKBACK
import numpy as np

CKPT_PATH = "checkpoints/lag-llama.ckpt"
DEVICE = "cuda"


def make_lag_llama(device=DEVICE):
    from lag_llama.gluon.estimator import LagLlamaEstimator

    # ─── Official approach: read ALL arch params from checkpoint ───
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    hp = ckpt["hyper_parameters"]

    # Official checkpoints store params under "estimator_args"
    if "estimator_args" in hp:
        ea = hp["estimator_args"]
    elif "model_kwargs" in hp:
        ea = hp["model_kwargs"]
    else:
        ea = hp

    print(f"[debug] estimator_args keys: {sorted(ea.keys())}")
    print(f"[debug] input_size={ea.get('input_size')}, n_layer={ea.get('n_layer')}, "
          f"n_head={ea.get('n_head')}, n_embd_per_head={ea.get('n_embd_per_head')}, "
          f"time_feat={ea.get('time_feat')}, rope_scaling={ea.get('rope_scaling')}")

    estimator = LagLlamaEstimator(
        prediction_length=max(HORIZONS),
        context_length=DEFAULT_LOOKBACK,
        input_size=ea["input_size"],
        n_layer=ea["n_layer"],
        n_embd_per_head=ea["n_embd_per_head"],
        n_head=ea["n_head"],
        scaling=ea.get("scaling", "mean"),
        time_feat=ea.get("time_feat", True),
        rope_scaling=ea.get("rope_scaling", None),
        num_parallel_samples=20,
        device=torch.device(device),
        batch_size=1,
        ckpt_path=CKPT_PATH,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    return predictor


def predict_fn(history: np.ndarray, horizon: int) -> np.ndarray:
    from gluonts.dataset.common import ListDataset
    import pandas as pd

    ds = ListDataset(
        [{"start": pd.Timestamp("2020-01-01"), "target": history.astype(np.float32)}],
        freq="10min",
    )
    forecasts = list(predictor.predict(ds))
    samples = forecasts[0].samples  # (num_samples, pred_len)
    median = np.median(samples[:, :horizon], axis=0)
    return median


if __name__ == "__main__":
    import pandas as pd
    from run_zeroshot import DEFAULT_LOOKBACK

    # Load test series (same as run_zeroshot.py)
    test_csv = "data/alibaba_gpu_test.csv"
    df = pd.read_csv(test_csv)
    series = df["gpu_utilization"].values

    predictor = make_lag_llama()
    results = evaluate(predict_fn, series, DEFAULT_LOOKBACK, n_windows=10)
    avg = sum(results.values()) / len(results)
    print(f"\nLag-Llama zero-shot  avg MAE = {avg:.3f}")
    for h in HORIZONS:
        print(f"  h={h}: {results[f'h={h}']:.3f}")
