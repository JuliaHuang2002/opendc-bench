"""Evaluate fine-tuned Moirai checkpoints vs zero-shot using same inference path."""
import sys, time, json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_zeroshot import HORIZONS, DEFAULT_LOOKBACK, evaluate

TEST_DATA = Path("data/alibaba_10min_test_v2.npy")
MODEL_NAME = "Salesforce/moirai-1.1-R-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_N_WINDOWS = 1297

def make_moirai_predictor(state_dict_path=None):
    """Build Moirai predict_fn, optionally loading fine-tuned weights."""
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    max_h = max(HORIZONS)
    module = MoiraiModule.from_pretrained(MODEL_NAME)

    if state_dict_path is not None:
        sd = torch.load(state_dict_path, map_location="cpu")
        module.load_state_dict(sd)
        print(f"  loaded weights from {state_dict_path}")

    model = MoiraiForecast(
        module=module, prediction_length=max_h, context_length=DEFAULT_LOOKBACK,
        patch_size=16, num_samples=20, target_dim=1,
        feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    ).to(DEVICE)
    model.eval()

    def predict(ctx):
        past = torch.tensor(ctx, dtype=torch.float32, device=DEVICE).view(1, -1, 1)
        past_obs = torch.ones_like(past, dtype=torch.bool)
        past_pad = torch.zeros_like(past[..., 0], dtype=torch.bool)
        with torch.no_grad():
            samples = model(past_target=past, past_observed_target=past_obs, past_is_pad=past_pad)
        return samples[0].median(dim=0).values.cpu().numpy()

    return predict

def main():
    series = np.load(TEST_DATA).astype(np.float32)
    print("="*60)
    print(f"Moirai-base eval | test len={len(series)} | device={DEVICE}")
    print("="*60)

    configs = [
        ("zero-shot", None, "Moirai-base zero-shot"),
        ("ft-head_only", "checkpoints/moirai_ft_head_only/final.pt", "Moirai-base ft-head_only"),
        ("ft-freeze_ffn", "checkpoints/moirai_ft_freeze_ffn/final.pt", "Moirai-base ft-freeze_ffn"),
        ("ft-full", "checkpoints/moirai_ft_full/final.pt", "Moirai-base ft-full"),
        ("ft-freeze_ffn_v2", "checkpoints/moirai_ft_freeze_ffn_v2/final.pt", "Moirai-base ft-freeze_ffn_v2"),
    ]

    results = []
    for tag, ckpt, display_name in configs:
        ckpt_path = Path(ckpt) if ckpt else None
        if ckpt_path and not ckpt_path.exists():
            print(f"  {tag:20s} | SKIP (not found)")
            continue
        print(f"\n>>> {tag}")
        predict_fn = make_moirai_predictor(ckpt_path)
        mae, runtime = evaluate(predict_fn, series, DEFAULT_LOOKBACK, n_windows=DEFAULT_N_WINDOWS)
        results.append((tag, mae, runtime))

        # Save unified JSON for benchmark_summary
        result = {
            "model": f"Salesforce/moirai-1.1-R-base-{tag}",
            "display_model": display_name,
            "family": "fine-tuned foundation",
            "device": DEVICE,
            "n_windows": DEFAULT_N_WINDOWS,
            "lookback": DEFAULT_LOOKBACK,
            "mae": mae,
            "runtime_sec": round(runtime, 1),
        }
        out_json = Path("/home/hongshao.hzx/notebook/results") / f"moirai_ft_{tag}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved: {out_json}")

    print("\n" + "="*60)
    print(f"{'Model':<20s} | {'h=1':>6s} {'h=6':>6s} {'h=36':>7s} {'h=144':>7s} | {'time':>5s}")
    print("-"*60)
    for tag, mae, rt in results:
        h1, h6, h36, h144 = [mae.get(f'h={h}', float('nan')) for h in HORIZONS]
        print(f"{tag:<20s} | {h1:6.3f} {h6:6.3f} {h36:7.3f} {h144:7.3f} | {rt:5.1f}s")

if __name__ == "__main__":
    main()