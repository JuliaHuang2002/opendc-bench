import numpy as np

train = np.load('alibaba_10min_train.npy')
test = np.load('alibaba_10min_test.npy')
full = np.concatenate([train, test])
test_start = len(train)

def mae(y, p): return np.abs(y - p).mean()
def rmse(y, p): return np.sqrt(((y - p)**2).mean())

def eval_horizon(h):
    # 滚动预测: 在 test 段每个点 t, 用 t-1 时刻信息预测 t..t+h-1
    n = len(test) - h
    y_true_h = np.array([test[i:i+h] for i in range(n)])  # (n, h)
    
    # Naive: 把 t-1 的值复制 h 次
    last_vals = full[test_start-1:test_start-1+n]
    pred_naive = np.repeat(last_vals[:, None], h, axis=1)
    
    # MA(6) 复制 h 次
    pred_ma = np.array([[full[test_start+i-6:test_start+i].mean()]*h for i in range(n)])
    
    # Seasonal 24h: 用 24h 前的对应 h 步
    pred_seas = np.array([full[test_start+i-144:test_start+i-144+h] for i in range(n)])
    
    return {
        "Naive":    (mae(y_true_h, pred_naive), rmse(y_true_h, pred_naive)),
        "MA(6)":    (mae(y_true_h, pred_ma),    rmse(y_true_h, pred_ma)),
        "Seasonal": (mae(y_true_h, pred_seas),  rmse(y_true_h, pred_seas)),
    }

print(f"{'Horizon':<10} {'Model':<12} {'MAE':>8} {'RMSE':>8}")
print("-" * 45)
for h, label in [(1,"10min"), (6,"1h"), (36,"6h"), (144,"24h")]:
    res = eval_horizon(h)
    for name, (m, r) in res.items():
        print(f"h={h:<3}({label:<5}) {name:<12} {m:>8.3f} {r:>8.3f}")
    print()
