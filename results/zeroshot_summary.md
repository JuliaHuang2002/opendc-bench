# OpenDC-Bench Zero-shot 结果汇总

**数据集**: Alibaba cluster-trace-gpu-v2023 (10-min), test set N=1440
**Lookback**: 288 (48h), **Horizons**: 1 / 6 / 36 / 144 (10min / 1h / 6h / 24h)
**Metric**: MAE, num_samples=20 (median)

| Model | h=1 | h=6 | h=36 | h=144 | Avg | Runtime |
|---|---|---|---|---|---|---|
| amazon/chronos-bolt-small | 0.882 | 1.482 | 2.688 | 3.639 | **2.173** | 35s |
| amazon/chronos-bolt-base | 0.857 | 1.468 | 2.732 | 3.830 | **2.222** | 58s |
| amazon/chronos-bolt-tiny | 0.901 | 1.502 | 2.708 | 3.789 | **2.225** | 28s |
| amazon/chronos-t5-base | 0.856 | 1.483 | 2.795 | 3.863 | **2.249** | 1662s |
| Salesforce/moirai-1.1-R-small | 0.899 | 1.502 | 2.795 | 3.806 | **2.251** | 21s |
| Salesforce/moirai-1.1-R-base | 0.866 | 1.475 | 2.894 | 4.736 | **2.493** | 29s |
| time-series-foundation-models/Lag-Llama | 2.571 | 3.133 | 3.861 | 4.272 | **3.459** | 1019s |