"""Process MIT SuperCloud nvidia_smi.csv from S3 and convert to numpy format.

This script:
1. First probes the first 100MB to find available nodes
2. Then streams and filters for selected nodes
3. Extracts utilization_gpu_pct
4. Resamples to 10-minute granularity
5. Saves as numpy array (same format as Alibaba data)
"""
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pandas as pd
import numpy as np
from io import StringIO
import time
from collections import Counter

BUCKET = 'mit-supercloud-dataset'
KEY = '2022-hpca/nvidia_smi.csv'
OUTPUT_DIR = '/home/hongshao.hzx/opendc-bench/data/mit_supercloud'

# Configuration
GPU_INDEX = 0.0
RESAMPLE_FREQ = '10min'  # 10 minutes
PROBE_SIZE = 100 * 1024 * 1024  # 100 MB for probing


def probe_nodes():
    """Read first 100MB and list top nodes by data volume."""
    print(f"[probe] Reading first {PROBE_SIZE / 1e6:.0f} MB to find nodes...")
    
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    response = s3.get_object(Bucket=BUCKET, Key=KEY, Range=f'bytes=0-{PROBE_SIZE-1}')
    content = response['Body'].read().decode('utf-8', errors='ignore')
    
    lines = content.strip().split('\n')
    header = lines[0]
    data_lines = lines[1:]
    
    # Count nodes
    node_counter = Counter()
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split(',')
        if len(parts) >= 2:
            node = parts[0]
            node_counter[node] += 1
    
    print(f"[probe] Found {len(node_counter)} unique nodes in first {PROBE_SIZE / 1e6:.0f} MB")
    print(f"[probe] Top 10 nodes by row count:")
    for node, count in node_counter.most_common(10):
        print(f"  {node}: {count} rows")
    
    # Return the top node
    top_node = node_counter.most_common(1)[0][0]
    print(f"\n[info] Selected node: {top_node}")
    return top_node, header


def stream_and_process(target_node):
    """Stream CSV from S3 and process in chunks."""
    print(f"[info] Streaming {KEY} from S3, filtering for node={target_node}...")
    t0 = time.time()
    
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # Get object metadata
    head = s3.head_object(Bucket=BUCKET, Key=KEY)
    total_size = head['ContentLength']
    print(f"[info] Total file size: {total_size / 1e9:.2f} GB")
    
    # Stream in chunks
    chunk_size = 100 * 1024 * 1024  # 100 MB
    all_data = []
    bytes_read = 0
    row_count = 0
    
    response = s3.get_object(Bucket=BUCKET, Key=KEY)
    stream = response['Body']
    
    header = None
    chunk_lines = []
    
    print(f"[info] Processing in {chunk_size / 1e6:.0f} MB chunks...")
    
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        
        bytes_read += len(chunk)
        text = chunk.decode('utf-8', errors='ignore')
        
        # Split into lines
        lines = text.split('\n')
        
        # Handle header
        if header is None:
            header = lines[0]
            lines = lines[1:]
        
        # Filter lines for target node
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(',')
            if len(parts) < 12:
                continue
            
            node = parts[0]
            if node == target_node:
                chunk_lines.append(line)
                row_count += 1
        
        # Process chunk when we have enough lines
        if len(chunk_lines) > 10000:
            csv_text = header + '\n' + '\n'.join(chunk_lines)
            df = pd.read_csv(StringIO(csv_text))
            
            # Filter for target GPU
            df = df[df['gpu_index'] == GPU_INDEX]
            
            # Extract timestamp and utilization
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df[['timestamp_dt', 'utilization_gpu_pct']].copy()
            df = df.rename(columns={'utilization_gpu_pct': 'gpu_util'})
            
            all_data.append(df)
            chunk_lines = []
            
            progress = bytes_read / total_size * 100
            elapsed = time.time() - t0
            speed = bytes_read / elapsed / 1e6 if elapsed > 0 else 0
            print(f"[progress] {progress:.1f}% ({bytes_read/1e9:.2f}/{total_size/1e9:.2f} GB) "
                  f"speed={speed:.1f} MB/s, rows={row_count}, chunks={len(all_data)}")
    
    # Process remaining lines
    if chunk_lines:
        csv_text = header + '\n' + '\n'.join(chunk_lines)
        df = pd.read_csv(StringIO(csv_text))
        df = df[df['gpu_index'] == GPU_INDEX]
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df[['timestamp_dt', 'utilization_gpu_pct']].copy()
        df = df.rename(columns={'utilization_gpu_pct': 'gpu_util'})
        all_data.append(df)
    
    if not all_data:
        print("[error] No data collected!")
        return
    
    # Concatenate all chunks
    print(f"[info] Concatenating {len(all_data)} chunks...")
    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.sort_values('timestamp_dt')
    
    print(f"[info] Total raw points: {len(full_df)}")
    print(f"[info] Time range: {full_df['timestamp_dt'].min()} to {full_df['timestamp_dt'].max()}")
    
    # Set timestamp as index and resample
    full_df = full_df.set_index('timestamp_dt')
    resampled = full_df['gpu_util'].resample(RESAMPLE_FREQ).mean()
    
    # Drop NaN values (gaps in data)
    resampled = resampled.dropna()
    
    print(f"[info] After resampling to {RESAMPLE_FREQ}: {len(resampled)} points")
    print(f"[info] Mean: {resampled.mean():.2f}, Std: {resampled.std():.2f}")
    
    # Save as numpy
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/mit_supercloud_gpu_util.npy"
    np.save(output_path, resampled.values.astype(np.float32))
    print(f"[done] Saved to {output_path}")
    
    # Also save as CSV for inspection
    csv_path = f"{OUTPUT_DIR}/mit_supercloud_gpu_util.csv"
    resampled.to_csv(csv_path)
    print(f"[done] Also saved CSV to {csv_path}")


if __name__ == "__main__":
    # Step 1: Probe to find a good node
    target_node, header = probe_nodes()
    
    # Step 2: Stream and process
    stream_and_process(target_node)
