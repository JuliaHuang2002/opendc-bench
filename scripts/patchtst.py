import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

train = np.load('data/alibaba_10min_train.npy').astype(np.float32)
test  = np.load('data/alibaba_10min_test.npy').astype(np.float32)
full  = np.concatenate([train, test])

mu, sd = train.mean(), train.std()
train_n = (train - mu) / sd
full_n  = (full  - mu) / sd

LOOKBACK = 288   # 48h
HORIZON  = 144   # 24h
PATCH    = 16    # patch 长度
STRIDE   = 8     # patch 步长 -> (288-16)/8+1 = 35 个 patch

def make_windows(arr, lb, h):
    X, Y = [], []
    for i in range(len(arr) - lb - h + 1):
        X.append(arr[i:i+lb]); Y.append(arr[i+lb:i+lb+h])
    return np.array(X), np.array(Y)

Xtr, Ytr = make_windows(train_n, LOOKBACK, HORIZON)
print(f"train windows: {Xtr.shape}")

class PatchTST(nn.Module):
    def __init__(self, lb=LOOKBACK, h=HORIZON, patch=PATCH, stride=STRIDE,
                 d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.patch = patch; self.stride = stride
        self.n_patches = (lb - patch) // stride + 1
        self.proj = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model * self.n_patches, h)

    def forward(self, x):
        # x: (B, L) -> patches: (B, N, P)
        B = x.size(0)
        patches = x.unfold(dimension=1, size=self.patch, step=self.stride)
        # 实例归一化(RevIN 简化版)：每个样本减去自己的均值，提升泛化
        mean = patches.mean(dim=(1,2), keepdim=True)
        std  = patches.std (dim=(1,2), keepdim=True) + 1e-5
        patches_norm = (patches - mean) / std
        tokens = self.proj(patches_norm) + self.pos
        z = self.encoder(tokens)            # (B, N, D)
        out = self.head(z.flatten(1))       # (B, h)
        # 反归一化
        return out * std.squeeze(-1) + mean.squeeze(-1)

model = PatchTST().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
loss_fn = nn.MSELoss()

ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
dl = DataLoader(ds, batch_size=64, shuffle=True)

print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")

for epoch in range(30):
    model.train(); losses = []
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    sched.step()
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

print(f"\n{'Horizon':<10} {'PatchTST':>10} {'DLinear':>10} {'Naive':>10}")
print("-" * 45)
for h_eval, dl_mae, nv_mae in [(1, 1.14, 0.86), (6, 1.64, 1.51), (36, 2.91, 2.90), (144, 3.93, 4.27)]:
    mae = np.abs(preds_all[:, :h_eval] - trues_all[:, :h_eval]).mean()
    print(f"h={h_eval:<4}     {mae:>10.3f} {dl_mae:>10.3f} {nv_mae:>10.3f}")
