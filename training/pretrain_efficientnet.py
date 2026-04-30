import os
import shutil
import random

random.seed(42)

SRC = "data/mvtc/leather"
DST = "data/processed"

def force_delete(func, path, exc_info):
    """Force delete read-only files."""
    os.chmod(path, 0o777)
    func(path)

# ---------- 1. Clean destination ----------
for d in [os.path.join(DST, "train"), os.path.join(DST, "val")]:
    for sub in ["good", "defective"]:
        d_sub = os.path.join(d, sub)
        if os.path.exists(d_sub):
            os.chmod(d_sub, 0o777)
            shutil.rmtree(d_sub, onerror=force_delete)
        os.makedirs(d_sub)

# ---------- 2. Good images (from train/good) ----------
train_good_src = os.path.join(SRC, "train", "good")
all_train_good = sorted(os.listdir(train_good_src))
random.shuffle(all_train_good)
n_train = int(0.8 * len(all_train_good))
train_good_files = all_train_good[:n_train]
val_good_files = all_train_good[n_train:]

for fname in train_good_files:
    shutil.copy2(os.path.join(train_good_src, fname),
                 os.path.join(DST, "train", "good", fname))
for fname in val_good_files:
    shutil.copy2(os.path.join(train_good_src, fname),
                 os.path.join(DST, "val", "good", fname))

# ---------- 3. Defective images (from test/ subdirs) ----------
defective_dirs = [
    os.path.join(SRC, "test", d) for d in os.listdir(os.path.join(SRC, "test"))
    if d != "good"
]

all_defective = []
for d_dir in defective_dirs:
    defect_type = os.path.basename(d_dir)      # 'color', 'cut', ...
    for fname in os.listdir(d_dir):
        new_name = f"{defect_type}_{fname}"    # unique name
        all_defective.append((d_dir, fname, new_name))

random.shuffle(all_defective)
n_def_train = int(0.8 * len(all_defective))
train_def = all_defective[:n_def_train]
val_def = all_defective[n_def_train:]

def safe_copy(src_path, dst_dir, dst_name):
    """Copy with overwrite protection (ensures writable)."""
    dst_path = os.path.join(dst_dir, dst_name)
    try:
        shutil.copy2(src_path, dst_path)
    except PermissionError:
        if os.path.exists(dst_path):
            os.chmod(dst_path, 0o666)
            os.remove(dst_path)
        shutil.copy2(src_path, dst_path)

for d_dir, orig_fname, new_name in train_def:
    safe_copy(os.path.join(d_dir, orig_fname),
              os.path.join(DST, "train", "defective"),
              new_name)
for d_dir, orig_fname, new_name in val_def:
    safe_copy(os.path.join(d_dir, orig_fname),
              os.path.join(DST, "val", "defective"),
              new_name)

print("Dataset split complete.")
print(f"Train good: {len(os.listdir(os.path.join(DST, 'train', 'good')))}")
print(f"Train defective: {len(os.listdir(os.path.join(DST, 'train', 'defective')))}")
print(f"Val good: {len(os.listdir(os.path.join(DST, 'val', 'good')))}")
print(f"Val defective: {len(os.listdir(os.path.join(DST, 'val', 'defective')))}")