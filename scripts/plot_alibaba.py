"""
画阿里 GPU 时序的可视化图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 服务器无图形界面用这个

ts = np.load('/home/hongshao.hzx/opendc-bench/alibaba_gpu_minute_series.npy')

fig, axes = plt.subplots(3, 1, figsize=(14, 9))

# 全局视图（149 天）
axes[0].plot(ts, linewidth=0.3, color='steelblue')
axes[0].set_title(f'Alibaba GPU usage - full timespan (149 days, {len(ts)} minutes)')
axes[0].set_xlabel('Minute')
axes[0].set_ylabel('GPU count in use')
axes[0].grid(alpha=0.3)

# 中间一天
mid = len(ts) // 2
axes[1].plot(ts[mid:mid+1440], linewidth=0.8, color='darkorange')
axes[1].set_title('A typical day (1440 minutes)')
axes[1].set_xlabel('Minute')
axes[1].set_ylabel('GPU count in use')
axes[1].grid(alpha=0.3)

# 一小时
axes[2].plot(ts[mid:mid+60], marker='o', linewidth=1, color='seagreen')
axes[2].set_title('A typical hour (60 minutes)')
axes[2].set_xlabel('Minute')
axes[2].set_ylabel('GPU count in use')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/hongshao.hzx/opendc-bench/alibaba_gpu_timeseries.png', dpi=120, bbox_inches='tight')
print("saved to ~/opendc-bench/alibaba_gpu_timeseries.png")

# 几个统计
print(f"\n=== Summary ===")
print(f"Mean: {ts.mean():.2f}, Std: {ts.std():.2f}")
print(f"Zeros (idle minutes): {(ts==0).sum()} ({(ts==0).mean()*100:.1f}%)")
print(f"P50/P90/P99: {np.percentile(ts, 50):.0f} / {np.percentile(ts, 90):.0f} / {np.percentile(ts, 99):.0f}")
