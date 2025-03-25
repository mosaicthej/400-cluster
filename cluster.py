#!/usr/bin/python

from pathlib import Path

# Define the root directory and video extensions
root_dir = Path('/home/mosaic/Videos/mov/')
video_extensions = {'.mp4', '.mkv', '.avi', '.mov'}

# Use rglob to recursively find video files
video_files = [file for file in root_dir.rglob('*') if 
               file.suffix.lower() in video_extensions]

print(f"Found {len(video_files)} video files.")


def extract_blocks(file_path, block_size=32):
    blocks = []
    with open(file_path, 'rb') as f:  # Read binary, no duplicates created
        while True:
            block = f.read(block_size)
            if not block:
                break
            # Pad the last block with zeros if needed
            if len(block) < block_size:
                block += b'\x00' * (block_size - len(block))
            blocks.append(block)
    return blocks


from hashlib import sha256

def hash_blocks(blocks):
    hashes = {}
    for idx, block in enumerate(blocks):
        block_hash = sha256(block).hexdigest()
        if block_hash not in hashes:
            hashes[block_hash] = []
        hashes[block_hash].append(idx)
    return hashes

# Example usage for a single file:
# blocks = extract_blocks(video_files[0], block_size=32)
blocks = extract_blocks(video_files[0], block_size=32)
hashes = hash_blocks(blocks)
print(f"Unique blocks: {len(hashes)} / Total blocks: {len(blocks)}")


import numpy as np

def block_to_vector(block):
    return np.frombuffer(block, dtype=np.uint8)

# Example:
block_vector = block_to_vector(blocks[0])
print(block_vector.shape)  # (32,) for 32-byte blocks


from sklearn.cluster import MiniBatchKMeans

def cluster_blocks(all_vectors, n_clusters=100, batch_size=1000):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
    kmeans.fit(all_vectors)
    return kmeans

# Collect vectors from all files (example for a subset):
all_vectors = []
for i, file in enumerate(video_files[:10]):  # Start with 10 files to test
    blocks = extract_blocks(file, block_size=32)
    all_vectors.extend([block_to_vector(b) for b in blocks])
    print("done vectorizing file", i, "out of 10") 

all_vectors = np.vstack(all_vectors)

print('all_vectors obtained')

kmeans = cluster_blocks(all_vectors, n_clusters=100)

print('found kmeans', kmeans)

def compute_distances(vectors, kmeans_model):
    # Get cluster centroids (base blocks)
    centroids = kmeans_model.cluster_centers_.astype(np.uint8)
    # Predict cluster for each vector
    labels = kmeans_model.predict(vectors)
    # Compute distance of each block to its centroid
    distances = []
    for i, vec in enumerate(vectors):
        centroid = centroids[labels[i]]
        distance = np.sum(np.abs(vec - centroid))  # L1 distance (proxy for XOR)
        distances.append(distance)
    return distances

distances = compute_distances(all_vectors, kmeans)


