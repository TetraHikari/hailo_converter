# YOLOv8n → Hailo HEF Conversion Pipeline

Converts a YOLOv8n ONNX model to a Hailo Executable Format (HEF) for deployment on Raspberry Pi 5 with a Hailo-8 M.2 accelerator.

---

## Project Structure

```
convert/
├── README.md
├── .gitignore
├── camera_test.py          # Pi inference script (live camera + Hailo-8)
├── convert_hailo.py        # Full conversion pipeline (ONNX → HAR → HEF)
├── prepare_calib.py        # Stratified calibration data preparation
├── config/
│   ├── collab_model.alls   # Normalization + performance settings (active)
│   └── fire_smoke.alls     # Alternative alls config
├── setup/
│   ├── setup_hailo.sh      # Hailo DFC environment setup
│   └── nvidia_toolkit.sh   # NVIDIA container toolkit setup
├── output/
│   └── yolov8n_10cls.hef   # Compiled model (deploy this to Pi)
└── archieve/               # Previous model iterations (large files gitignored)
```

---

## Requirements

### Host Machine (Windows + WSL2)
- Ubuntu 22.04 on WSL2
- NVIDIA GPU with CUDA 12.5+
- cuDNN 9 (`libcudnn9-cuda-12`)
- Python 3.12
- Hailo Dataflow Compiler 3.33.1 (`.whl` included)

### Target Device
- Raspberry Pi 5
- Hailo-8 M.2 accelerator
- Raspberry Pi Camera Module 3 (IMX708)
- `hailo-all` runtime package installed

---

## Setup

### 1. Create Virtual Environment

```bash
python3.12 -m venv env
source env/bin/activate
```

### 2. Install Hailo Dataflow Compiler

```bash
pip install hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl
```

### 3. Install Hailo Model Zoo (optional, for reference scripts)

```bash
pip install -e hailo_model_zoo/
```

---

## Conversion

### Step 1 — Prepare Calibration Data

Calibration data is used during quantization to minimize accuracy loss. The script stratifies sampling to oversample underrepresented classes (e.g. small flames).

```bash
python prepare_calib.py
```

Output: 1024 `.npy` files saved to `calib_npy/`

### Step 2 — Run the Full Pipeline

```bash
python convert_hailo.py
```

This runs three stages automatically:

| Stage | Input | Output |
|---|---|---|
| Parse ONNX | `dataset/best.onnx` | `yolov8n_10cls.har` |
| Optimize (quantize) | `yolov8n_10cls.har` + calibration data | `yolov8n_10cls_optimized.har` |
| Compile | `yolov8n_10cls_optimized.har` | `yolov8n_10cls.hef` |

> **Note:** Optimization runs at level 2 with GPU. Compilation uses CPU only and may take 30–60 minutes.

---

## Deployment

### Copy HEF to Raspberry Pi

```bash
scp output/yolov8n_10cls.hef ai@raspberrypi:/home/ai/Desktop/projects/dub-fire/AI/models/converted/
```

### Copy Inference Script

```bash
scp camera_test.py ai@raspberrypi:/home/ai/Desktop/projects/dub-fire/AI/models/
```

### Run on Pi

```bash
python camera_test.py
```

Press `q` to quit, `s` to save a snapshot.

---

## Model Details

- **Architecture:** YOLOv8n
- **Input:** 640×640 RGB (letterboxed from camera feed)
- **Hardware target:** Hailo-8 (`hailo8`)
- **Quantization:** INT8, optimization level 2

### Classes

| ID | Name | Detection Label |
|---|---|---|
| 0 | Ates (flame) | Fire Detected |
| 1 | Fire | Fire Detected |
| 2 | Fire Detection | Fire Detected |
| 3 | api | — |
| 4 | car-crash | — |
| 5 | fight | — |
| 6 | knife | — |
| 7 | no-fight | — |
| 8 | pistol | — |
| 9 | smoke | — |

> Classes 0, 1, and 2 are grouped into a single **"Fire Detected"** label in the inference script. Other classes are ignored.

### Confidence Thresholds

| Class Group | Threshold |
|---|---|
| Fire / Ates / Fire Detection / Smoke (0,1,2,9) | 0.12 |
| All other classes | 0.20 |

---

## Troubleshooting

**Hailo SDK falls back to CPU**
The SDK rejects GPUs with >5% VRAM usage. The threshold has been patched to 90% in:
```
env/lib/python3.12/site-packages/hailo_model_optimization/acceleras/utils/nvidia_smi_gpu_selector.py
```
Re-apply this patch if the virtual environment is recreated.

**Optimization level drops to 0**
Requires 1024+ calibration samples. Run `prepare_calib.py` to regenerate.

**Compiler timeout during cluster mapping**
Normal — the compiler falls back to Auto-Merger automatically. Let it continue.
