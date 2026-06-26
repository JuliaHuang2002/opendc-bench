"""
把阿里 cluster-trace-gpu-v2023 事件型数据转成"每分钟 GPU 占用"时序
"""
import pandas as pd
import numpy as np

CSV = "/home/hongshao.hzx/clusterdata/cluster-trace-gpu-v2023/csv/openb_pod_list_default.csv"
df = pd.read_csv(CSV)

print("===== 原始数据概况 =====")
print(f"总任务数: {len(df)}")
print(f"字段: {list(df.columns)}")
print(f"\n各列缺失情况:")
print(df.isna().sum())

# 关键字段：把时间从秒转成分钟
print("\n===== 时间范围 =====")
print(f"最早创建: {df['creation_time'].min()} 秒")
print(f"最晚结束: {df['deletion_time'].max()} 秒")
total_minutes = df['deletion_time'].max() // 60
print(f"总跨度: 约 {total_minutes} 分钟 = {total_minutes//60} 小时 = {total_minutes//60//24} 天")

# 统计 GPU 任务情况
print("\n===== GPU 任务情况 =====")
print(f"申请 GPU 的任务数: {(df['num_gpu'] > 0).sum()}")
print(f"GPU 任务占比: {(df['num_gpu'] > 0).mean()*100:.1f}%")
print(f"\nnum_gpu 分布:")
print(df['num_gpu'].value_counts().sort_index())

print("\n===== 把事件转成每分钟 GPU 占用时序 =====")
# 只保留有效任务（有创建和结束时间）
gpu_jobs = df[(df['num_gpu'] > 0) & (df['deletion_time'] > df['creation_time'])].copy()
gpu_jobs['gpu_count'] = gpu_jobs['num_gpu'] * (gpu_jobs['gpu_milli'].fillna(1000) / 1000)
print(f"有效 GPU 任务: {len(gpu_jobs)}")

# 构建分钟级时序：长度 = 总分钟数
total_min = int(gpu_jobs['deletion_time'].max() // 60) + 1
timeseries = np.zeros(total_min)
for _, row in gpu_jobs.iterrows():
    start_min = int(row['creation_time'] // 60)
    end_min = int(row['deletion_time'] // 60)
    timeseries[start_min:end_min] += row['gpu_count']

print(f"\n时序长度: {len(timeseries)} 分钟")
print(f"平均 GPU 占用: {timeseries.mean():.1f} 块")
print(f"最大 GPU 占用: {timeseries.max():.0f} 块")
print(f"最小 GPU 占用: {timeseries.min():.0f} 块")

# 存下来给后面用
np.save('/home/hongshao.hzx/opendc-bench/alibaba_gpu_minute_series.npy', timeseries)
print(f"\n已保存到 ~/opendc-bench/alibaba_gpu_minute_series.npy")
