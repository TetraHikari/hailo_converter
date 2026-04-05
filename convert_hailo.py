"""
YOLOv8n (10-class) → Hailo HEF conversion pipeline

Steps:
  1. Parse ONNX → best_10cls.har
  2. Optimize (quantize) with calibration data → best_10cls_optimized.har
  3. Compile → best_10cls_compiled.har + best_10cls.hef

Usage:
  source env/bin/activate
  python convert_hailo.py
"""

import os
import glob
import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext

# ── Config ────────────────────────────────────────────────────────────────────
WORK_DIR    = "/mnt/e/programs/convert"
ONNX_PATH   = f"{WORK_DIR}/dataset/best.onnx"
CALIB_DIR   = f"{WORK_DIR}/calib_npy"
ALLS_PATH   = f"{WORK_DIR}/collab_model.alls"   # normalization + perf settings
HW_ARCH     = "hailo8"
MODEL_NAME  = "yolov8n_10cls"

HAR_RAW      = f"{WORK_DIR}/{MODEL_NAME}.har"
HAR_OPT      = f"{WORK_DIR}/{MODEL_NAME}_optimized.har"
HAR_COMPILED = f"{WORK_DIR}/{MODEL_NAME}_compiled.har"
HEF_PATH     = f"{WORK_DIR}/{MODEL_NAME}.hef"

# YOLOv8 end nodes — cuts before the Detect-head post-processing (use ONNX node names)
END_NODES = [
    "/model.22/cv2.0/cv2.0.2/Conv",
    "/model.22/cv3.0/cv3.0.2/Conv",
    "/model.22/cv2.1/cv2.1.2/Conv",
    "/model.22/cv3.1/cv3.1.2/Conv",
    "/model.22/cv2.2/cv2.2.2/Conv",
    "/model.22/cv3.2/cv3.2.2/Conv",
]

# ── Step 1: Parse ONNX ────────────────────────────────────────────────────────
if os.path.exists(HAR_RAW):
    print(f"STEP 1: Skipping — {HAR_RAW} already exists\n")
else:
    print("=" * 60)
    print("STEP 1: Parsing ONNX → HAR")
    print("=" * 60)

    runner = ClientRunner(hw_arch=HW_ARCH)
    runner.translate_onnx_model(
        model=ONNX_PATH,
        net_name=MODEL_NAME,
        end_node_names=END_NODES,
        net_input_shapes={"images": [1, 3, 640, 640]},
    )

    # Apply normalization and performance settings
    with open(ALLS_PATH) as f:
        alls_content = f.read()
    runner.load_model_script(alls_content)

    runner.save_har(HAR_RAW)
    print(f"Saved: {HAR_RAW}\n")

# ── Step 2: Optimize (quantize) ───────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Optimizing (quantization) → optimized HAR")
print("=" * 60)

calib_files = sorted(glob.glob(os.path.join(CALIB_DIR, "*.npy")))
print(f"Using {len(calib_files)} calibration samples from {CALIB_DIR}")

# Load as NHWC uint8 [0,255] — the .alls normalization layer divides by 255 at runtime
calib_data = np.stack([np.load(f) for f in calib_files]).astype(np.uint8)
print(f"Calibration dataset shape: {calib_data.shape}, dtype: {calib_data.dtype}")

runner = ClientRunner(hw_arch=HW_ARCH, har=HAR_RAW)
with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
    runner.optimize(calib_data)

runner.save_har(HAR_OPT)
print(f"Saved: {HAR_OPT}\n")

# ── Step 3: Compile → HEF ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Compiling → HEF")
print("=" * 60)

runner = ClientRunner(hw_arch=HW_ARCH, har=HAR_OPT)
hef = runner.compile()

with open(HEF_PATH, "wb") as f:
    f.write(hef)

runner.save_har(HAR_COMPILED)
print(f"Saved HEF:  {HEF_PATH}")
print(f"Saved HAR:  {HAR_COMPILED}")
print("\nDone!")
