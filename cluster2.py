#!/usr/bin/python

from pathlib import Path

import os
import numpy as np
from hashlib import sha256

from tqdm import tqdm
import seaborn as sns



# Define the root directory and video extensions
root_dir = Path('/home/mosaic/Videos/mov/')
video_extensions = {'.mp4', '.mkv', '.avi', '.mov'}

# Use rglob to recursively find video files
video_files = [file for file in root_dir.rglob('*') if 
               file.suffix.lower() in video_extensions]

print(f"Found {len(video_files)} video files.")


def block_generator(file_paths, block_size=32):
    """Yields blocks one at a time from all files."""
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            while True:
                block = f.read(block_size)
                if not block:
                    break
                # Pad if needed
                if len(block) < block_size:
                    block += b'\x00' * (block_size - len(block))
                yield block

# def vector_generator(file_paths, block_size=32):
 #   """Converts blocks to numerical vectors on-the-fly."""
  #  for block in block_generator(file_paths, block_size):
   #     yield np.frombuffer(block, dtype=np.uint8)

def vector_generator(file_paths, block_size=64, skip_count=0):
    """Skip already processed blocks when resuming"""
    count = 0
    for block in block_generator(file_paths, block_size):
        if count < skip_count:
            count += 1
            continue
        yield np.frombuffer(block, dtype=np.uint8)
        count += 1

# part 2: incremental clustering with checkpoints...
import joblib  # For saving/loading checkpoints
from sklearn.cluster import MiniBatchKMeans

def incremental_clustering(
    file_paths,
    block_size=32,
    n_clusters=100,
    batch_size=1000,
    checkpoint_interval=10_000,
    checkpoint_dir="checkpoints"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Try to load existing checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "kmeans_checkpoint.pkl")
    if os.path.exists(checkpoint_path):
        kmeans = joblib.load(checkpoint_path)
        print("Resuming from checkpoint...")
    else:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
    
    # Process data in batches
    vector_gen = vector_generator(file_paths, block_size)
    processed_count = 0
    
    # while True:
    while processed_count < 100_000_000: # while it's less than 100k
        batch = []
        try:
            # Collect one batch
            for _ in range(batch_size):
                batch.append(next(vector_gen))
        except StopIteration:
            break  # No more data
        
        # Update model incrementally
        kmeans.partial_fit(np.array(batch))
        processed_count += len(batch)
        
        # Save checkpoint periodically
        if processed_count % checkpoint_interval == 0:
            joblib.dump(kmeans, checkpoint_path)
            print(f"\rCheckpoint saved at {processed_count} blocks", end=' ')
    
    # Final save
    print("done, saving.....")
    joblib.dump(kmeans, checkpoint_path)
    return kmeans

# part 3: Distance Calculation with Disk Streaming
'''
def save_distances(vector_gen, kmeans, output_file="distances.bin"):
    centroids = kmeans.cluster_centers_.astype(np.uint8)
    
    with open(output_file, 'ab') as f:  # Append mode
        for vec in vector_gen:
            label = kmeans.predict(vec.reshape(1, -1))[0]
            distance = np.sum(np.abs(vec - centroids[label]))
            # Save as binary (4-byte float)
            f.write(np.array(distance, dtype=np.float32).tobytes())
'''
def save_distances(vector_gen, kmeans, output_file="distances.bin", labels_file="labels.bin", buffer_size=10000):
    """Save distances AND cluster labels for all blocks."""
    centroids = kmeans.cluster_centers_.astype(np.uint8)
    centroids = centroids.reshape(kmeans.n_clusters, -1)
    
    # Resume awareness for distances and labels
    skip_dist = os.path.getsize(output_file) // 4 if os.path.exists(output_file) else 0
    skip_labels = os.path.getsize(labels_file) // 4 if os.path.exists(labels_file) else 0
    skip_count = max(skip_dist, skip_labels)
    
    # Initialize buffers
    dist_buffer = np.zeros(buffer_size, dtype=np.float32)
    label_buffer = np.zeros(buffer_size, dtype=np.int32)
    buffer_idx = 0
    total_processed = 0
    
    with open(output_file, 'ab' if skip_count > 0 else 'wb') as f_dist, \
         open(labels_file, 'ab' if skip_count > 0 else 'wb') as f_labels:
        
        progress = tqdm(total=skip_count, desc="Processing blocks", unit="blocks")
        try:
            while True:
                # Read a batch of vectors
                batch = []
                while len(batch) < 1024:
                    try:
                        batch.append(next(vector_gen))
                    except StopIteration:
                        break
                if not batch:
                    break
                batch = np.array(batch)
                
                # Skip already processed
                if total_processed < skip_count:
                    to_skip = min(len(batch), skip_count - total_processed)
                    batch = batch[to_skip:]
                    total_processed += to_skip
                    progress.update(to_skip)
                    if len(batch) == 0:
                        continue
                
                # Batch prediction and distance calculation
                labels = kmeans.predict(batch)
                distances = np.sum(np.abs(batch - centroids[labels]), axis=1)
                
                # Fill buffers
                for d, l in zip(distances, labels):
                    dist_buffer[buffer_idx] = d
                    label_buffer[buffer_idx] = l
                    buffer_idx += 1
                    
                    # Write when buffer is full
                    if buffer_idx == buffer_size:
                        f_dist.write(dist_buffer.tobytes())
                        f_labels.write(label_buffer.tobytes())
                        progress.update(buffer_size)
                        buffer_idx = 0
                
                total_processed += len(batch)
                progress.update(len(batch))
                
        finally:
            # Write remaining buffers
            if buffer_idx > 0:
                f_dist.write(dist_buffer[:buffer_idx].tobytes())
                f_labels.write(label_buffer[:buffer_idx].tobytes())
                progress.update(buffer_idx)
            progress.close()


# part 4: file process first check checkpoints
def get_unprocessed_files(all_files, checkpoint_dir="checkpoints"):
    processed_log = os.path.join(checkpoint_dir, "processed_files.txt")
    
    if os.path.exists(processed_log):
        with open(processed_log, 'r') as f:
            processed = set(f.read().splitlines())
        return [f for f in all_files if f not in processed]
    else:
        return all_files.copy()

def update_processed_files(processed_files, checkpoint_dir="checkpoints"):
    with open(os.path.join(checkpoint_dir, "processed_files.txt"), 'a') as f:
        # f.write('\n'.join(processed_files) + '\n')
        f.write('\n'.join([str(file) for file in processed_files]) + '\n')


# print centroid in hex format
def print_centroid(centroid, block_size=32):
    hex_str = ' '.join(f'{b:02x}' for b in centroid)
    print(f"Centroid (hex): {hex_str}")



# main workflow

# 2. Resume or start fresh
checkpoint_dir = "checkpoints"
unprocessed = get_unprocessed_files(video_files, checkpoint_dir)

# 3. Process in chunks (e.g., 10 files at a time)
chunk_size = 10
for i in range(0, len(unprocessed), chunk_size):
    chunk = unprocessed[i:i+chunk_size]
    
    # 3a. Train clusters incrementally
    kmeans = incremental_clustering(
        chunk,
        block_size=32,
        n_clusters=100,
        batch_size=1000,
        checkpoint_dir=checkpoint_dir
    )
    
    # 3b. Calculate distances for this chunk
    vector_gen = vector_generator(chunk, block_size=32)
    save_distances(vector_gen, kmeans, "distances.bin")
    
    # 3c. Mark files as processed
    update_processed_files(chunk, checkpoint_dir)


# 4. presentation
# Load all distances and labels
distances = np.fromfile("distances.bin", dtype=np.float32)
labels = np.fromfile("labels.bin", dtype=np.int32)

# Compute average distance per cluster
n_clusters = kmeans.n_clusters
cluster_avg_distances = [
    np.mean(distances[labels == i]) 
    for i in range(n_clusters)
]

def plot_clust_size_avgdist(labels, minlength, 
                            cluster_avg_distances, save_to="clust_avgdist.jpg"):
    # Plot cluster size vs. average distance
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    plt.scatter(cluster_sizes, cluster_avg_distances, alpha=0.5)
    plt.xlabel("Cluster Size")
    plt.ylabel("Average Distance")
    plt.title("cluster size vs average distance")
    plt.savefig(save_to, dpi=500)

plot_clust_size_avgdist(labels, n_clusters, cluster_avg_distances)

d_hist = sns.histplot(distances, bins=50)
kmeans = joblib.load("checkpoints/kmeans_checkpoint.pkl")
centroids = kmeans.cluster_centers_.astype(np.uint8)

print("the centroids looks like this, which is averaged out template....")
print_centroid(centroids[0])

# plot the centroids (comparative over different set s
def plot_centroids_comp(centroidss, labels, save_to="centroids.jpg"):
    plt.figure(figsize=(8,4))
    for i, cs in enumerate(centroidss):
        sns.kdeplot([c[0] for c in cs], label=labels[i])
    plt.xlabel('Byte value at position 0')
    plt.title('Comparative byte distributions')
    plt.legend()
    plt.savefig(save_to, dpi=500)

plot_centroids_comp((centroids,), ("centroids general",))

def plot_centroids_heatmap(centroids, save_to="heatmap_centroids.jpg"):
    # Compute byte-wise variability across all centroids
    variability = np.std(centroids, axis=0)
    plt.figure(figsize=(12, 4))
    sns.heatmap(variability.reshape(1, -1), cmap='viridis', annot=False, cbar=True)
    plt.xlabel('Byte position in block')
    plt.title('Byte Stability Across All Centroids')
    plt.savefig(save_to, dpi=500)

plot_centroids_heatmap(centroids)

def find_fixed_centroids(centroids, block_size=32):
    # Bytes that are identical in >95% of centroids
    fixed_positions = []
    for pos in range(block_size):
        unique_vals = len(np.unique(centroids[:, pos]))
        if unique_vals <= 1:  # Fully fixed
            fixed_positions.append(pos)
    print(f"Fixed bytes at positions: {fixed_positions}")

find_fixed_centroids(centroids)
