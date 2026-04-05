import os
import random
import numpy as np
from PIL import Image
import glob

IMAGES_DIR = "/mnt/e/programs/convert/dataset/train/images/images"
LABELS_DIR = "/mnt/e/programs/convert/dataset/train/labels"
OUTPUT_DIR = "/mnt/e/programs/convert/calib_npy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stratified calibration: guarantee good coverage of rare classes
# Target: 500 Ates(0), 200 Fire(1), 150 FireDetection(2), 50 smoke(9), 124 random others
TARGETS = {0: 500, 1: 200, 2: 150, 9: 50}
TOTAL   = 1024

random.seed(42)

all_images = (
    glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGES_DIR, "*.png"))
)

# Group images by which classes they contain
by_class = {i: [] for i in range(10)}
no_label  = []

for img_path in all_images:
    stem     = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(LABELS_DIR, stem + ".txt")
    if not os.path.exists(lbl_path):
        no_label.append(img_path)
        continue
    with open(lbl_path) as f:
        classes = {int(l.split()[0]) for l in f if l.strip()}
    for c in classes:
        by_class[c].append(img_path)

print("Images per class:", {c: len(v) for c, v in by_class.items()})

selected = set()

# Sample targeted classes first
for cls, target in TARGETS.items():
    pool = [p for p in by_class[cls] if p not in selected]
    random.shuffle(pool)
    selected.update(pool[:target])
    print(f"Class {cls}: selected {min(target, len(pool))} / {len(pool)} available")

# Fill remainder with random images from any class
remaining = TOTAL - len(selected)
rest = [p for p in all_images if p not in selected]
random.shuffle(rest)
selected.update(rest[:remaining])

selected = list(selected)
random.shuffle(selected)
print(f"Total selected: {len(selected)}")

# Clear old calibration files
for f in glob.glob(os.path.join(OUTPUT_DIR, "*.npy")):
    os.remove(f)

saved = 0
for img_path in selected:
    try:
        img = Image.open(img_path).convert("RGB").resize((640, 640))
        arr = np.array(img, dtype=np.uint8)
        np.save(os.path.join(OUTPUT_DIR, f"calib_{saved:04d}.npy"), arr)
        saved += 1
    except Exception as e:
        print(f"Skipped {img_path}: {e}")

print(f"Saved {saved} calibration files to {OUTPUT_DIR}")
