import os
import random
import shutil

source_dir = "" #replace with directory where images are stored
dest_dir = "" #replace with desired directory where sample should be stored
os.makedirs(dest_dir, exist_ok=True)


all_files = [f for f in os.listdir(source_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
sample_files = random.sample(all_files, 100)

for i, fname in enumerate(sample_files, 1):
    src_path = os.path.join(source_dir, fname)
    dst_path = os.path.join(dest_dir, f"human_{i:03d}.jpg")
    shutil.copy2(src_path, dst_path)

print(f"Copied {len(sample_files)} images to {dest_dir}")
