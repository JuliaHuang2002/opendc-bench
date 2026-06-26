import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

train = np.load('alibaba_10min_train.npy').astype(np.float32)
test  = np.load('alibaba_10min_test.npy').astype(np.float32)
full  = np.concatenate([train, test])

# 标准化（用 train 的 mean/std）
mu, sd = train.mean(), train.std()
train_n = (train - mu) / sd
full_n  = (full  - mu) / sd

LOOKBACK = 288   # 过去 48h
HORIZON  = 144   # 预测未来 24h

def make_windows(arr, lb, h):
    X, Y = [], []
    for i in range(len(arr) - lb - h + 1):
        X.append(arr[i:i+lb])
        Y.append(arr[i+lb:i+lb+h])
    return np.array(X), np.array(Y)

Xtr, Ytr = make_windows(train_n, LOOKBACK, HORIZON)
print(f"train windows: {Xtr.shape}, {Ytr.shape}")

class LSTMForecaster(nn.Module):
    def __init__(self, hidden=64, h=HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, h)
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

model = LSTMForecaster().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
dl = DataLoader(ds, batch_size=64, shuffle=True)

for epoch in range(20):
    model.train()
    losses = []
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"epoch {epoch+1:2d}  train_loss={np.mean(losses):.4f}")

# 评测：在 test 段滚动预测
model.eval()
test_start = len(train)
preds_all, trues_all = [], []
with torch.no_grad():
    for i in range(len(test) - HORIZON + 1):
        x = full_n[test_start + i - LOOKBACK : test_start + i]
        if len(x) < LOOKBACK: continue
        x = torch.from_numpy(x).float().unsqueeze(0).to(device)
        p = model(x).cpu().numpy()[0]
        preds_all.append(p * sd + mu)
        trues_all.append(full[test_start+i : test_start+i+HORIZON])

preds_all = np.array(preds_all); trues_all = np.array(trues_all)
print(f"\nLSTM h=144 (24h)")
print(f"  MAE  = {np.abs(preds_all - trues_all).mean():.3f}  (Naive=4.272)")
print(f"  RMSE = {np.sqrt(((preds_all - trues_all)**2).mean()):.3f}  (Naive=5.659)")
