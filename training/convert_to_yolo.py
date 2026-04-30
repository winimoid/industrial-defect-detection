import os
import cv2
import numpy as np
import shutil
import random

random.seed(42)

# ---------- Configuration ----------
SRC = "data/mvtc/leather"
DST = "data/yolo"

CLASS_ID = 0  # only one class: defect

# ---------- Helper: mask to YOLO bbox ----------
def mask_to_yolo_bbox(mask_path, class_id=0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    H, W = mask.shape
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    # Normalize to [0,1]
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    nw = w / W
    nh = h / H
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

# ---------- Clean & create YOLO directory structure ----------
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    d = os.path.join(DST, sub)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# ---------- Collect defective images ----------
defect_dirs = [
    os.path.join(SRC, "test", d) for d in os.listdir(os.path.join(SRC, "test"))
    if d != "good"
]

all_defective_pairs = []  # (image_path, mask_path, defect_type, new_name)
for d_dir in defect_dirs:
    defect_type = os.path.basename(d_dir)
    mask_dir = os.path.join(SRC, "ground_truth", defect_type)
    for fname in os.listdir(d_dir):
        img_path = os.path.join(d_dir, fname)
        # Your mask naming: e.g., 001.png -> 001_mask.png
        base, ext = os.path.splitext(fname)
        mask_path = os.path.join(mask_dir, f"{base}_mask.png")
        if os.path.exists(mask_path):
            # Create a unique filename: defecttype_originalname
            new_name = f"{defect_type}_{fname}"
            all_defective_pairs.append((img_path, mask_path, defect_type, new_name))
        else:
            print(f"Warning: mask not found for {img_path} (expected: {mask_path})")

# Split defective
random.shuffle(all_defective_pairs)
split_idx = int(0.8 * len(all_defective_pairs))
train_def = all_defective_pairs[:split_idx]
val_def = all_defective_pairs[split_idx:]

# ---------- Utility: safe copy ----------
def safe_copy(src_path, dst_dir, dst_name):
    dst_path = os.path.join(dst_dir, dst_name)
    try:
        shutil.copy2(src_path, dst_path)
    except PermissionError:
        if os.path.exists(dst_path):
            os.chmod(dst_path, 0o666)
            os.remove(dst_path)
        shutil.copy2(src_path, dst_path)

# ---------- Process defective images ----------
def process_set(pairs, img_dst_dir, lbl_dst_dir):
    for img_path, mask_path, defect_type, new_name in pairs:
        # Copy image with new unique name
        safe_copy(img_path, img_dst_dir, new_name)

        # Generate YOLO label file, name must match new_name
        annotation = mask_to_yolo_bbox(mask_path, CLASS_ID)
        if annotation is None:
            print(f"Empty mask for {img_path} - skipping")
            continue
        lbl_name = os.path.splitext(new_name)[0] + ".txt"
        with open(os.path.join(lbl_dst_dir, lbl_name), "w") as f:
            f.write(annotation + "\n")

process_set(train_def, f"{DST}/images/train", f"{DST}/labels/train")
process_set(val_def, f"{DST}/images/val", f"{DST}/labels/val")

# ---------- Process good images (test/good only) ----------
good_src = os.path.join(SRC, "test", "good")
all_good = [os.path.join(good_src, f) for f in os.listdir(good_src)]
random.shuffle(all_good)

n_good_train = int(0.8 * len(all_good))
train_good = all_good[:n_good_train]
val_good = all_good[n_good_train:]

def process_good(file_list, img_dst_dir, lbl_dst_dir):
    for img_path in file_list:
        fname = os.path.basename(img_path)
        new_name = f"good_{fname}"          # unique prefix
        safe_copy(img_path, img_dst_dir, new_name)

        # Empty label file for good images
        lbl_name = os.path.splitext(new_name)[0] + ".txt"
        with open(os.path.join(lbl_dst_dir, lbl_name), "w") as f:
            pass

process_good(train_good, f"{DST}/images/train", f"{DST}/labels/train")
process_good(val_good, f"{DST}/images/val", f"{DST}/labels/val")

print("YOLO dataset created successfully.")
print("Train images:", len(os.listdir(f"{DST}/images/train")))
print("Train labels:", len(os.listdir(f"{DST}/labels/train")))
print("Val images:", len(os.listdir(f"{DST}/images/val")))
print("Val labels:", len(os.listdir(f"{DST}/labels/val")))