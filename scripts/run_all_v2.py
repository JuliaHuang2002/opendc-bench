import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

train = np.load('data/alibaba_10min_train_v2.npy').astype(np.float32)
val   = np.load('data/alibaba_10min_val_v2.npy').astype(np.float32)
test  = np.load('data/alibaba_10min_test_v2.npy').astype(np.float32)
full  = np.concatenate([train, val, test])
test_start = len(train) + len(val)

mu, sd = train.mean(), train.std()
print(f"train mean={mu:.2f} std={sd:.2f}")
train_n = (train - mu) / sd
full_n  = (full  - mu) / sd

LOOKBACK, HORIZON = 288, 144

def make_windows(arr, lb, h):
    X, Y = [], []
    for i in range(len(arr) - lb - h + 1):
        X.append(arr[i:i+lb]); Y.append(arr[i+lb:i+lb+h])
    return np.array(X), np.array(Y)

Xtr, Ytr = make_windows(train_n, LOOKBACK, HORIZON)
print(f"train windows: {Xtr.shape}")

# ============ 1) Naive ============
print("\n=== Naive ===")
def eval_horizon_naive(h):
    n = len(test) - h
    y_true_h = np.array([test[i:i+h] for i in range(n)])
    last_vals = full[test_start-1:test_start-1+n]
    pred = np.repeat(last_vals[:, None], h, axis=1)
    return np.abs(y_true_h - pred).mean()

naive = {h: eval_horizon_naive(h) for h in [1, 6, 36, 144]}
print({k: round(v, 3) for k, v in naive.items()})

# ============ 2) DLinear ============
print("\n=== DLinear ===")
class DLinear(nn.Module):
    def __init__(self, lb=LOOKBACK, h=HORIZON, kernel=25):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel, 1, padding=kernel//2)
        self.lt = nn.Linear(lb, h)
        self.ls = nn.Linear(lb, h)
    def forward(self, x):
        t = self.avg(x.unsqueeze(1)).squeeze(1)[:, :x.size(1)]
        return self.lt(t) + self.ls(x - t)

dl_model = DLinear().to(device)
opt = torch.optim.Adam(dl_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)), 64, shuffle=True)
for epoch in range(30):
    dl_model.train(); losses = []
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(dl_model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    if (epoch+1) % 10 == 0: print(f"  epoch {epoch+1}  loss={np.mean(losses):.4f}")

def rolling_eval(model):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for i in range(len(test) - HORIZON + 1):
            x = full_n[test_start + i - LOOKBACK : test_start + i]
            x = torch.from_numpy(x).float().unsqueeze(0).to(device)
            p = model(x).cpu().numpy()[0]
            preds.append(p * sd + mu)
            trues.append(full[test_start+i : test_start+i+HORIZON])
    return np.array(preds), np.array(trues)

p, t = rolling_eval(dl_model)
dlinear = {h: float(np.abs(p[:, :h] - t[:, :h]).mean()) for h in [1,6,36,144]}
print({k: round(v, 3) for k, v in dlinear.items()})

# ============ 3) PatchTST v2 ============
print("\n=== PatchTSTv2 ===")
class PatchTSTv2(nn.Module):
    def __init__(self, lb=LOOKBACK, h=HORIZON, patch=16, stride=8,
                 d_model=48, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.n_patches = (lb - patch)//stride + 1
        self.proj = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model)*0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model*2,
                                          dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Linear(d_model*self.n_patches, h)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        m = x.mean(1, keepdim=True); s = x.std(1, keepdim=True)+1e-5
        xn = (x - m)/s
        z = self.proj(xn.unfold(1, self.patch, self.stride)) + self.pos
        z = self.encoder(z)
        return self.head(self.drop(z.flatten(1))) * s + m

# 用真正的 val 集做 early stop (不再切训练集)
val_n = (val - mu) / sd
val_full = np.concatenate([train_n[-LOOKBACK:], val_n])
Xva, Yva = [], []
for i in range(len(val_n) - HORIZON + 1):
    Xva.append(val_full[i:i+LOOKBACK])
    Yva.append(val_n[i:i+HORIZON]) if i+HORIZON <= len(val_n) else None
Xva = np.array(Xva[:len(val_n)-HORIZON+1])
Yva = np.array([val_n[i:i+HORIZON] for i in range(len(val_n)-HORIZON+1)])
print(f"  val windows: {Xva.shape}")
xv = torch.from_numpy(Xva.astype(np.float32)).to(device)
yv = torch.from_numpy(Yva.astype(np.float32)).to(device)

pt_model = PatchTSTv2().to(device)
opt = torch.optim.AdamW(pt_model.parameters(), lr=5e-4, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
best_val, best_state = float('inf'), None
for epoch in range(50):
    pt_model.train(); losses = []
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(pt_model(xb), yb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pt_model.parameters(), 1.0)
        opt.step(); losses.append(loss.item())
    sched.step()
    pt_model.eval()
    with torch.no_grad():
        vl = loss_fn(pt_model(xv), yv).item()
    if vl < best_val:
        best_val = vl
        best_state = {k: v.clone() for k, v in pt_model.state_dict().items()}
    if (epoch+1) % 10 == 0: print(f"  epoch {epoch+1}  train={np.mean(losses):.4f}  val={vl:.4f}  best={best_val:.4f}")

pt_model.load_state_dict(best_state)
p, t = rolling_eval(pt_model)
patchtst = {h: float(np.abs(p[:, :h] - t[:, :h]).mean()) for h in [1,6,36,144]}
print({k: round(v, 3) for k, v in patchtst.items()})

# ============ 总表 ============
print(f"\n{'='*60}\n80天版结果 vs 50天版\n{'='*60}")
print(f"{'Horizon':<8} {'Naive':>8} {'DLinear':>10} {'PatchTST':>10}")
print(f"{'(80d)':<8} {'(50d)':>8} {'(50d)':>10} {'(50d)':>10}")
print("-" * 50)
ref = {'naive_50':[0.86,1.51,2.90,4.27], 'dl_50':[1.14,1.64,2.91,3.93], 'pt_50':[1.82,2.24,3.53,5.76]}
for idx, h in enumerate([1,6,36,144]):
    print(f"h={h:<5} {naive[h]:>5.2f} ({ref['naive_50'][idx]:.2f})  {dlinear[h]:>5.2f} ({ref['dl_50'][idx]:.2f})  {patchtst[h]:>5.2f} ({ref['pt_50'][idx]:.2f})")
