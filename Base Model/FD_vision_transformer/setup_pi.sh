#!/bin/bash
echo "Setting up Raspberry Pi..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv libatlas-base-dev python3-opencv i2c-tools
sudo raspi-config nonint do_i2c 0
mkdir -p /home/pi/solar_fault/captures
python3 -m venv ~/solar_env
source ~/solar_env/bin/activate
pip install --upgrade pip
pip install onnx onnxruntime numpy opencv-python scipy
pip install adafruit-circuitpython-mlx90640 adafruit-blinka
pip install pyserial pynmea2 timm torch
echo "Done! Run: source ~/solar_env/bin/activate"
