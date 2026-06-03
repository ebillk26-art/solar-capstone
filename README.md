# Edge-Based Multimodal Deep Learning System for Solar PV Fault Detection

**Emmanuel Bilson** | BSc Electrical and Electronics Engineering | Ashesi University, 2026  
**Supervisor:** Dr Richard Akparibo Awingot

---

## Overview

This project presents a real-time solar panel fault detection system that runs entirely on the edge — no cloud connection required. It combines RGB and thermal imaging to detect both physical and electrical faults simultaneously, using deep learning models deployed on a Raspberry Pi 5 mounted on a Parrot ANAFI Ai drone.

RGB imagery detects visible surface defects such as dust, cracks, and bird droppings. Thermal imagery identifies electrical anomalies like hotspots and diode failures. The two streams run in parallel and transmit results wirelessly to a database and user dashboard.

---

## System

**Hardware:** Raspberry Pi 5, Raspberry Pi Camera Module 3, MLX90640 thermal camera (32×24 px), NEO-6M GPS module, custom PCB HAT with 2S1P 18650 battery pack (7.4V, ~36 min runtime at full load), DC-DC buck converter. Mounted on a Parrot ANAFI Ai drone.

**Software pipeline:**
1. The RGB stream first detects a solar panel in view
2. Both RGB and thermal images are captured and passed to their respective classifiers
3. Results (fault type, confidence, GPS coordinates, timestamp) are logged locally and sent via Wi-Fi to a remote database

---

## Fault Classes

**RGB — 4 classes:** Bird-drop, Clean, Dusty, Physical-Damage

**Thermal — 6 classes:** Cell-Fault, Diode-Fault, Hot-Spot, No-Anomaly, Offline-Module, Shadowing

---

## Models and Results

Training was done on Ashesi University's JupyterHub (compute.ashesi.edu.gh). All models were trained with 224×224 input, batch size 32, AdamW optimizer, and early stopping.

A single-modality baseline using YOLOv8m on all fault types achieved 77.4% accuracy. A Vision Transformer baseline reached 86%. These results motivated the two-stream approach.

**Final RGB model — YOLOv11m:** 98.5% accuracy, 98.5% precision, 98.6% recall  
**Final thermal model — ResNet-18:** 94.3% accuracy (No-Anomaly 98.6%, Cell-Fault 95.3%, Diode-Fault 94.1%)

Other architectures evaluated: YOLOv8m-cls, YOLOv11m-cls, Vision Transformer, EfficientNet, Custom CNN, AdaptPolyKAN.

---

## Real-World Testing

The system was tested on a drone flown at approximately 6 metres. Clean panels were classified at 100% confidence, dusty panels at 93.4%, and physically damaged panels at 87.8%. The system runs at approximately 1–1.3 FPS on the Raspberry Pi 5, with a response time of about 1 second per inspection.

A key finding was that a cracked panel produced both a visible RGB defect (Physical-Damage, 87.8%) and abnormal thermal signatures (Shadowing 52%, Cell-Fault 24%), confirming that physical damage also manifests as electrical anomalies — validating the multimodal design.

---

## Repository Structure

```
solar-capstone/
├── Base Model/                  # Multimodal system and model experiments
│   ├── Solar_V2/                # CNN, ResNet, EfficientNet, YOLOv8m training
│   ├── FD_AdaptPolyKAN/         # AdaptPolyKAN experiment
│   └── FD_vision_transformer/   # ViT experiment + Raspberry Pi deployment scripts
├── rgb_fault_detect/            # RGB 4-class YOLOv11m pipeline
├── thermal_fault_detection/     # Thermal 6-class ResNet-18 pipeline
├── .gitignore
└── README.md
```

Dataset images and trained model weights are not included in this repository due to size. Download links will be added.

---

## Setup

```bash
pip install -r "Base Model/Solar_V2/requirements.txt"
```

To set up the Raspberry Pi environment:
```bash
bash "Base Model/FD_vision_transformer/setup_pi.sh"
```

To run real-time inference on the Pi:
```bash
python "Base Model/FD_vision_transformer/main.py"
```

---

## Acknowledgements

Supervised by Dr Richard Akparibo Awingot. Training compute provided by Ashesi University JupyterHub. Built with PyTorch and Ultralytics YOLO. Datasets sourced from Kaggle (PV Panel Defect Dataset) and the InfraredSolarModules dataset.
