import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train = np.load('alibaba_10min_train.npy').astype(np.float32)
test  = np.load('alibaba_10min_test.npy').astype(np.float32)
full  = np.concatenate([train, test])

mu, sd = train.mean(), train.std()
train_n = (train - mu) / sd
full_n  = (full  - mu) / sd

LOOKBACK = 288
HORIZON  = 144

def make_windows(arr, lb, h):
    X, Y = [], []
    for i in range(len(arr) - lb - h + 1):
        X.append(arr[i:i+lb]); Y.append(arr[i+lb:i+lb+h])
    return np.array(X), np.array(Y)

Xtr, Ytr = make_windows(train_n, LOOKBACK, HORIZON)
print(f"train windows: {Xtr.shape}")

# DLinear: 直接 lookback -> horizon 的线性投影（带可学习的趋势/残差分解）
class DLinear(nn.Module):
    def __init__(self, lb=LOOKBACK, h=HORIZON, kernel=25):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=kernel//2)
        self.linear_trend = nn.Linear(lb, h)
        self.linear_seasonal = nn.Linear(lb, h)
    def forward(self, x):
        # x: (B, L)
        trend = self.avg(x.unsqueeze(1)).squeeze(1)[:, :x.size(1)]
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)

model = DLinear().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
dl = DataLoader(ds, batch_size=64, shuffle=True)

for epoch in range(30):
    model.train()
    losses = []
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    if (epoch+1) % 5 == 0:
        print(f"epoch {epoch+1:2d}  train_loss={np.mean(losses):.4f}")

# 滚动评测
model.eval()
test_start = len(train)
preds_all, trues_all = [], []
with torch.no_grad():
    for i in range(len(test) - HORIZON + 1):
        x = full_n[test_start + i - LOOKBACK : test_start + i]
        x = torch.from_numpy(x).float().unsqueeze(0).to(device)
        p = model(x).cpu().numpy()[0]
        preds_all.append(p * sd + mu)
        trues_all.append(full[test_start+i : test_start+i+HORIZON])

preds_all = np.array(preds_all); trues_all = np.array(trues_all)

# 分 horizon 看效果
print(f"\n{'Horizon':<10} {'MAE':>8} {'Naive':>8}")
print("-" * 30)
for h_eval, naive_mae in [(1, 0.86), (6, 1.51), (36, 2.90), (144, 4.27)]:
    mae = np.abs(preds_all[:, :h_eval] - trues_all[:, :h_eval]).mean()
    print(f"h={h_eval:<3}        {mae:>8.3f} {naive_mae:>8.3f}")
