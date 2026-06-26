import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train = np.load('data/alibaba_10min_train_v2.npy').astype(np.float32)
val   = np.load('data/alibaba_10min_val_v2.npy').astype(np.float32)
test  = np.load('data/alibaba_10min_test_v2.npy').astype(np.float32)
full  = np.concatenate([train, val, test])
test_start = len(train) + len(val)
mu, sd = train.mean(), train.std()
train_n = (train - mu)/sd; full_n = (full - mu)/sd

LOOKBACK, HORIZON = 288, 144
def make_windows(arr, lb, h):
    X, Y = [], []
    for i in range(len(arr)-lb-h+1):
        X.append(arr[i:i+lb]); Y.append(arr[i+lb:i+lb+h])
    return np.array(X), np.array(Y)
Xtr, Ytr = make_windows(train_n, LOOKBACK, HORIZON)

class DLinearRevIN(nn.Module):
    def __init__(self, lb=LOOKBACK, h=HORIZON, kernel=25):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel, 1, padding=kernel//2)
        self.lt = nn.Linear(lb, h); self.ls = nn.Linear(lb, h)
    def forward(self, x):
        m = x.mean(1, keepdim=True); s = x.std(1, keepdim=True)+1e-5
        xn = (x - m)/s
        t = self.avg(xn.unsqueeze(1)).squeeze(1)[:, :xn.size(1)]
        out = self.lt(t) + self.ls(xn - t)
        return out * s + m

model = DLinearRevIN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)), 64, shuffle=True)
for epoch in range(30):
    model.train(); losses=[]
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    if (epoch+1)%10==0: print(f"epoch {epoch+1} loss={np.mean(losses):.4f}")

model.eval(); preds, trues = [], []
with torch.no_grad():
    for i in range(len(test) - HORIZON + 1):
        x = full_n[test_start+i-LOOKBACK:test_start+i]
        x = torch.from_numpy(x).float().unsqueeze(0).to(device)
        p = model(x).cpu().numpy()[0]
        preds.append(p*sd+mu)
        trues.append(full[test_start+i:test_start+i+HORIZON])
preds, trues = np.array(preds), np.array(trues)

print(f"\nDLinear+RevIN (80d):")
print(f"{'h':<6} {'MAE':>8} {'vs DLinear 50d':>15}")
for h, ref in [(1,1.14),(6,1.64),(36,2.91),(144,3.93)]:
    mae = float(np.abs(preds[:,:h]-trues[:,:h]).mean())
    diff = (mae-ref)/ref*100
    print(f"h={h:<4} {mae:>8.3f}  {diff:>+6.1f}%")
