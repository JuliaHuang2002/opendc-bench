# OpenDC-Bench

Cross-dataset benchmark for AI data center GPU workload forecasting.

## Datasets
- Alibaba cluster-trace-gpu-v2023 (149 days, 8152 tasks)
- MIT SuperCloud (pending application)

## Day 1 Results (Alibaba, 10-min granularity, 50-day active window)

Train: 40 days (5760 steps), Test: 10 days (1440 steps).

| Horizon | Naive MAE | DLinear MAE |
|--------:|----------:|------------:|
| 10 min  | 0.86      | 1.14        |
| 1 h     | 1.51      | 1.64        |
| 6 h     | 2.90      | 2.91        |
| 24 h    | 4.27      | **3.93**    |

Headroom for deep models exists only at long horizons (h=144). Short-horizon
predictions are dominated by inertia (Naive is hard to beat).

## Scripts
- `scripts/explore_alibaba.py` — event → per-minute timeseries
- `scripts/aggregate_10min.py` — 10-min aggregation + train/test split
- `scripts/baseline_naive.py` / `baseline_horizon.py` — Naive/MA/Seasonal baselines
- `scripts/dlinear.py` — DLinear baseline (first model to beat Naive at h=144)

## Next
- PatchTST / iTransformer baselines
- Time-series foundation models (Chronos, TimesFM, Lag-Llama)
- LLM-based forecasting (LLM-Mixer + Qwen2.5)
