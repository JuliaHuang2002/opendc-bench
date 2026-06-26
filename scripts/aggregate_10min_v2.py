import numpy as np

ts = np.load('data/alibaba_gpu_minute_series.npy')
print(f"原始: {len(ts)} 分钟 = {len(ts)/1440:.1f} 天")

# 取最后 80 天
last_80d = ts[-80*1440:]
ts_10min = last_80d.reshape(-1, 10).mean(axis=1)
print(f"最后80天 10min: {len(ts_10min)} 步 = {len(ts_10min)/144:.1f} 天")
print(f"统计: mean={ts_10min.mean():.2f}, std={ts_10min.std():.2f}, max={ts_10min.max():.1f}, min={ts_10min.min():.1f}")

# 切分：前 65 天训练，中间 5 天 val，后 10 天测试
train = ts_10min[:65*144]
val   = ts_10min[65*144:70*144]
test  = ts_10min[70*144:]
print(f"训练: {len(train)} 步 = {len(train)/144:.0f} 天")
print(f"验证: {len(val)} 步 = {len(val)/144:.0f} 天")
print(f"测试: {len(test)} 步 = {len(test)/144:.0f} 天")

np.save('data/alibaba_10min_train_v2.npy', train)
np.save('data/alibaba_10min_val_v2.npy',   val)
np.save('data/alibaba_10min_test_v2.npy',  test)
np.save('data/alibaba_10min_full_v2.npy',  ts_10min)
print("\n保存: alibaba_10min_{train,val,test,full}_v2.npy")
