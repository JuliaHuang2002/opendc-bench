import numpy as np

train = np.load('alibaba_10min_train.npy')
test = np.load('alibaba_10min_test.npy')

def mae(y, p): return np.abs(y - p).mean()
def rmse(y, p): return np.sqrt(((y - p)**2).mean())
def mape(y, p): return (np.abs((y - p) / np.maximum(y, 1e-6)) * 100).mean()

# 预测下一步（horizon=1）
# Baseline 1: Naive (上一时刻)
# Baseline 2: 滑动平均(过去6步=1小时)
# Baseline 3: 训练集均值
# Baseline 4: 季节性 naive (取24小时前同一时刻=144步前)

full = np.concatenate([train, test])
test_start = len(train)

y_true = test[1:]  # 真实值
pred_naive = full[test_start:test_start+len(y_true)]  # 上一步
pred_ma6 = np.array([full[test_start+i-6:test_start+i].mean() for i in range(1, len(test))])
pred_mean = np.full(len(y_true), train.mean())
pred_seasonal = full[test_start-144+1:test_start-144+1+len(y_true)]  # 24小时前

print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8}")
print("-" * 55)
for name, p in [
    ("Naive (last value)", pred_naive),
    ("MA(6) = last 1h",   pred_ma6),
    ("Global mean",        pred_mean),
    ("Seasonal naive(24h)",pred_seasonal),
]:
    print(f"{name:<25} {mae(y_true,p):>8.3f} {rmse(y_true,p):>8.3f} {mape(y_true,p):>8.2f}")
