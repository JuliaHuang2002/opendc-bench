import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train = np.load('data/alibaba_10min_train.npy').astype(np.float32)
test  = np.load('data/alibaba_10min_test.npy').astype(np.float32)
full  = np.concatenate([train, test])
mu, sd = train.mean(), train.std()
train_n = (train - mu) / sd
full_n  = (full  - mu) / sd

LOOKBACK, HORIZON, PATCH, STRIDE = 288, 144, 16, 8

def make_windows(arr, lb, h):
    X, Y = [], []
    for i in range(len(arr) - lb - h + 1):
        X.append(arr[i:i+lb]); Y.append(arr[i+lb:i+lb+h])
    return np.array(X), np.array(Y)

Xtr, Ytr = make_windows(train_n, LOOKBACK, HORIZON)

class PatchTSTv2(nn.Module):
    def __init__(self, lb=LOOKBACK, h=HORIZON, patch=PATCH, stride=STRIDE,
                 d_model=48, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.n_patches = (lb - patch) // stride + 1
        self.proj = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model*2,
                                          dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Linear(d_model * self.n_patches, h)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # 正确 RevIN: 每个样本独立标准化
        mean = x.mean(dim=1, keepdim=True)
        std  = x.std (dim=1, keepdim=True) + 1e-5
        xn = (x - mean) / std
        patches = xn.unfold(1, self.patch, self.stride)  # (B, N, P)
        z = self.proj(patches) + self.pos
        z = self.encoder(z)
        out = self.head(self.drop(z.flatten(1)))
        return out * std + mean

model = PatchTSTv2().to(device)
print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
loss_fn = nn.MSELoss()
dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                batch_size=64, shuffle=True)

# 训练集划 10% 做 val
n_val = int(len(Xtr) * 0.1)
Xval, Yval = Xtr[-n_val:], Ytr[-n_val:]
Xtr2, Ytr2 = Xtr[:-n_val], Ytr[:-n_val]
dl = DataLoader(TensorDataset(torch.from_numpy(Xtr2), torch.from_numpy(Ytr2)),
                batch_size=64, shuffle=True)
xv = torch.from_numpy(Xval).to(device)
yv = torch.from_numpy(Yval).to(device)

best_val = float('inf'); best_state = None
for epoch in range(50):
    model.train(); losses = []
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb); loss = loss_fn(pred, yb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); losses.append(loss.item())
    sched.step()
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(xv), yv).item()
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if (epoch+1) % 5 == 0:
        print(f"epoch {epoch+1:2d}  train={np.mean(losses):.4f}  val={val_loss:.4f}  best={best_val:.4f}")

model.load_state_dict(best_state)
model.eval()
test_start = len(train)
preds, trues = [], []
with torch.no_grad():
    for i in range(len(test) - HORIZON + 1):
        x = full_n[test_start + i - LOOKBACK : test_start + i]
        x = torch.from_numpy(x).float().unsqueeze(0).to(device)
        p = model(x).cpu().numpy()[0]
        preds.append(p * sd + mu)
        trues.append(full[test_start+i : test_start+i+HORIZON])

preds, trues = np.array(preds), np.array(trues)
print(f"\n{'Horizon':<10} {'PatchTSTv2':>12} {'PatchTSTv1':>12} {'DLinear':>10} {'Naive':>8}")
print("-" * 58)
for h_eval, v1, dl_mae, nv in [(1,2.19,1.14,0.86),(6,2.71,1.64,1.51),(36,3.67,2.91,2.90),(144,5.32,3.93,4.27)]:
    mae = np.abs(preds[:, :h_eval] - trues[:, :h_eval]).mean()
    print(f"h={h_eval:<4}    {mae:>12.3f} {v1:>12.3f} {dl_mae:>10.3f} {nv:>8.3f}")
