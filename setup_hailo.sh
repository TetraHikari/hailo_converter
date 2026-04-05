#!/bin/bash
set -e

echo "=== Step 1: Install Docker (skipping if already installed) ==="
if command -v docker &> /dev/null; then
  echo "Docker already installed: $(docker --version) — skipping."
else
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc

  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

echo "=== Step 2: Install NVIDIA Container Toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart

echo "=== Step 3: Create Virtual Environment & Install Hailo DFC ==="
sudo apt install -y python3.10-venv
python3 -m venv env
source env/bin/activate
pip install hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl

echo "=== Step 4: Clone & Install Hailo Model Zoo ==="
git clone https://github.com/hailo-ai/hailo_model_zoo.git
cd hailo_model_zoo
pip install -e .
cd training/yolov8

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps (run manually):"
echo ""
echo "--- Build & run Docker for training ---"
echo "sudo docker build --build-arg timezone=\$(cat /etc/timezone) -t yolov8:v0 ."
echo "sudo docker run --name hailo_retraining -it --gpus all --ipc=host -v /path/to/local/data:/data yolov8:v0"
echo ""
echo "--- Inside Docker: Train ---"
echo "yolo detect train data=<YAML> model=<yolov8n.pt> name=<run_name> epochs=100 batch=16"
echo ""
echo "--- Inside Docker: Inference ---"
echo "yolo predict task=detect source=<image_path> model=<.pt path>"
echo ""
echo "--- Inside Docker: Export to ONNX ---"
echo "yolo export model=<.pt path> imgsz=640 format=onnx opset=11"
echo ""
echo "--- Inside Docker: Copy ONNX out ---"
echo "cp <.onnx path> /data"
echo ""
echo "--- Back on host (in venv): Compile to .hef ---"
echo "hailomz compile yolov8n --ckpt=<.onnx path> --hw-arch hailo8l --calib-path <test images folder> --classes <N> --performance"
