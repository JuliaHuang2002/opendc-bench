import numpy as np
import torch
import time
from chronos import ChronosPipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train = np.load('data/alibaba_10min_train_v2.npy').astype(np.float32)
val   = np.load('data/alibaba_10min_val_v2.npy').astype(np.float32)
test  = np.load('data/alibaba_10min_test_v2.npy').astype(np.float32)
full  = np.concatenate([train, val, test])
test_start = len(train) + len(val)

LOOKBACK, HORIZON = 288, 144

MODEL_NAME = "amazon/chronos-t5-small"
print(f"loading {MODEL_NAME}...")
t0 = time.time()
pipe = ChronosPipeline.from_pretrained(MODEL_NAME, device_map=device, torch_dtype=torch.float32)
print(f"loaded in {time.time()-t0:.1f}s")

import random; random.seed(42)
n_total = len(test) - HORIZON + 1
sample_indices = sorted(random.sample(range(n_total), min(200, n_total)))
print(f"predicting {len(sample_indices)}/{n_total} windows...")

preds, trues = [], []
t0 = time.time()
for k, i in enumerate(sample_indices):
    ctx = torch.tensor(full[test_start + i - LOOKBACK : test_start + i]).float()
    forecast = pipe.predict(ctx, HORIZON, num_samples=20)
    p = forecast[0].median(dim=0).values.cpu().numpy()
    preds.append(p)
    trues.append(full[test_start+i : test_start+i+HORIZON])
    if (k+1) % 50 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (k+1) * (len(sample_indices) - k - 1)
        print(f"  {k+1}/{len(sample_indices)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

preds, trues = np.array(preds), np.array(trues)
print(f"\nChronos-small zero-shot (n={len(preds)} sampled windows)")
print(f"{'Horizon':<8} {'Chronos':>10} {'DLinear+RevIN':>15} {'Naive':>8}")
print("-" * 50)
for h, dl, nv in [(1, 1.01, 0.86), (6, 1.69, 1.51), (36, 2.90, 2.90), (144, 3.93, 4.27)]:
    mae = float(np.abs(preds[:, :h] - trues[:, :h]).mean())
    print(f"h={h:<4}  {mae:>10.3f}  {dl:>15.3f}  {nv:>8.3f}")
