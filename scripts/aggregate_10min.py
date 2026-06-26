import numpy as np

ts = np.load('alibaba_gpu_minute_series.npy')
print(f"原始: {len(ts)} 分钟")

# 取最后 50 天
last_50d = ts[-50*1440:]
print(f"最后50天: {len(last_50d)} 分钟")

# 10 分钟聚合（取均值）
ts_10min = last_50d.reshape(-1, 10).mean(axis=1)
print(f"10分钟聚合后: {len(ts_10min)} 步 = {len(ts_10min)/144:.1f} 天")
print(f"统计: mean={ts_10min.mean():.2f}, std={ts_10min.std():.2f}, max={ts_10min.max():.1f}, min={ts_10min.min():.1f}")

# 切分: 前 40 天训练，后 10 天测试
train = ts_10min[:40*144]
test = ts_10min[40*144:]
print(f"训练集: {len(train)} 步 = {len(train)/144:.0f} 天")
print(f"测试集: {len(test)} 步 = {len(test)/144:.0f} 天")

np.save('alibaba_10min_train.npy', train)
np.save('alibaba_10min_test.npy', test)
np.save('alibaba_10min_full.npy', ts_10min)
print("\n保存: alibaba_10min_{train,test,full}.npy")
