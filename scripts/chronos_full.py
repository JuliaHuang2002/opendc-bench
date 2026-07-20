import numpy as np
import torch, time, sys
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
from chronos import ChronosPipeline

MODEL = sys.argv[1] if len(sys.argv) > 1 else "amazon/chronos-t5-small"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train = np.load('data/alibaba_10min_train_v2.npy').astype(np.float32)
val   = np.load('data/alibaba_10min_val_v2.npy').astype(np.float32)
test  = np.load('data/alibaba_10min_test_v2.npy').astype(np.float32)
full  = np.concatenate([train, val, test])
test_start = len(train) + len(val)

LOOKBACK, HORIZON = 288, 144
print(f"loading {MODEL}...")
t0 = time.time()
pipe = ChronosPipeline.from_pretrained(MODEL, device_map=device, torch_dtype=torch.float32)
print(f"loaded in {time.time()-t0:.1f}s, params: {sum(p.numel() for p in pipe.model.parameters())/1e6:.1f}M")

n_total = len(test) - HORIZON + 1
print(f"full evaluation: {n_total} windows")
preds, trues = [], []
t0 = time.time()
for i in range(n_total):
    ctx = torch.tensor(full[test_start + i - LOOKBACK : test_start + i]).float()
    forecast = pipe.predict(ctx, HORIZON, num_samples=20)
    p = forecast[0].median(dim=0).values.cpu().numpy()
    preds.append(p)
    trues.append(full[test_start+i : test_start+i+HORIZON])
    if (i+1) % 100 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{n_total}  el={el:.0f}s  eta={el/(i+1)*(n_total-i-1):.0f}s")

preds, trues = np.array(preds), np.array(trues)
name = MODEL.split('/')[-1]
np.save(f'results/{name}_preds.npy', preds)
np.save(f'results/{name}_trues.npy', trues)
print(f"\n{name} zero-shot (full n={len(preds)})")
print(f"{'Horizon':<8} {'Chronos':>10} {'DLinear+RevIN':>15} {'Naive':>8}")
for h, dl, nv in [(1,1.01,0.86),(6,1.69,1.51),(36,2.90,2.90),(144,3.93,4.27)]:
    mae = float(np.abs(preds[:, :h] - trues[:, :h]).mean())
    print(f"h={h:<4}  {mae:>10.3f}  {dl:>15.3f}  {nv:>8.3f}")
