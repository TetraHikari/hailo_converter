#!/bin/bash
set -e

KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
LIST=/etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "Downloading toolkit list..."
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=${KEYRING}] https://#g" \
  | sudo tee ${LIST}

echo "Installing nvidia-container-toolkit..."
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo "Configuring Docker runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

echo "Done!"
