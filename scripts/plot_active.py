import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ts = np.load('alibaba_gpu_minute_series.npy')
print(f"总长度: {len(ts)} 分钟 = {len(ts)/1440:.1f} 天")

# 取最后 50 天
last_50d = ts[-50*1440:]
print(f"最后 50 天: mean={last_50d.mean():.1f}, std={last_50d.std():.1f}, max={last_50d.max():.0f}")

# 找一个真正活跃的"典型日"——方差最大的那天
days = last_50d.reshape(-1, 1440)
day_std = days.std(axis=1)
active_day_idx = day_std.argmax()
print(f"最活跃的一天: 第{active_day_idx}天 (在最后50天里), std={day_std[active_day_idx]:.1f}")

fig, axes = plt.subplots(3, 1, figsize=(14, 9))

axes[0].plot(last_50d, lw=0.5)
axes[0].set_title(f'Last 50 days (mean={last_50d.mean():.1f}, std={last_50d.std():.1f})')
axes[0].set_xlabel('minute'); axes[0].set_ylabel('GPU count')

axes[1].plot(days[active_day_idx])
axes[1].set_title(f'Most active day (day {active_day_idx} of last 50)')
axes[1].set_xlabel('minute of day'); axes[1].set_ylabel('GPU count')

# 该活跃日里方差最大的小时
hours = days[active_day_idx].reshape(-1, 60)
hour_std = hours.std(axis=1)
active_hr = hour_std.argmax()
axes[2].plot(hours[active_hr])
axes[2].set_title(f'Most active hour (hour {active_hr})')
axes[2].set_xlabel('minute of hour'); axes[2].set_ylabel('GPU count')

plt.tight_layout()
plt.savefig('alibaba_active_period.png', dpi=120)
print("saved: alibaba_active_period.png")
