# Multimodal Deep Learning for Real-Time Solar PV Fault Detection on Edge Devices

> **BSc Capstone Project** | Electrical and Electronics Engineering | Ashesi University | 2026

---

## Author

| Field | Details |
|-------|---------|
| **Author** | Emmanuel Bilson |
| **Programme** | BSc Electrical and Electronics Engineering |
| **Institution** | Ashesi University, Ghana |
| **Supervisor** | Dr Richard Akparibo Awingot |
| **Year** | 2026 |
| **Repository** | https://github.com/ebillk26-art/solar |

---

## Project Overview

Solar photovoltaic (PV) panels degrade over time due to faults such as hot-spots, soiling, cracking, diode failures, and physical damage. Manual inspection is time-consuming, expensive, and hazardous — especially at utility-scale installations.

This project presents a **multimodal, edge-deployed deep learning system** that fuses **RGB** and **thermal infrared** imaging to detect and classify solar PV faults in real time. The system is deployed on a **Raspberry Pi 5** mounted on a **Parrot ANAFI Ai drone**, enabling autonomous aerial inspection without a cloud connection. Detected faults are transmitted wirelessly to a database and visualised on a user dashboard.

### Key Contributions

- **Dual-modality detection pipeline**: separate RGB (4-class) and thermal (9-class) classification heads, fused at inference time
- **Comparative model study**: CNN, ResNet-18, Vision Transformer (ViT), EfficientNet, YOLOv8m, YOLOv11m, and AdaptPolyKAN evaluated on the same datasets
- **Edge deployment**: all inference runs on-device (Raspberry Pi 5) with no internet dependency
- **Drone integration**: real-time capture from a Parrot ANAFI Ai UAV platform

---

## Fault Classes

### Thermal Imaging — 9 Classes
| Label | Description |
|-------|-------------|
| `Cell-Fault` | Localised cell defect visible as thermal anomaly |
| `Cracking` | Physical micro-crack causing heat concentration |
| `Diode-Fault` | Bypass diode failure showing elevated temperature |
| `Hot-Spot` | Overheating cell due to mismatch or shadowing |
| `No-Anomaly` | Healthy panel — no fault detected |
| `Offline-Module` | Disconnected or non-operating module |
| `Shadowing` | Partial shading reducing output |
| `Soiling` | Dirt or dust accumulation on panel surface |
| `Vegetation` | Plant growth obstructing the panel |

### RGB Imaging — 4 Classes
| Label | Description |
|-------|-------------|
| `Bird-drop` | Bird dropping on panel surface |
| `Clean` | Clean panel — no visible contamination |
| `Dusty` | Dust or particulate layer on panel |
| `Physical-Damage` | Visible cracks, chips, or broken glass |

---

## System Architecture

```
Parrot ANAFI Ai Drone
        │
        ├── RGB Camera ──────────────────────────────────┐
        │                                                 │
        └── Thermal Camera (FLIR / equivalent) ──────────┤
                                                          ▼
                                               Raspberry Pi 5 (Edge Node)
                                               ┌──────────────────────────┐
                                               │  RGB Branch              │
                                               │    YOLOv11m (4-class)    │
                                               │                          │
                                               │  Thermal Branch          │
                                               │    YOLOv8m / EfficientNet│
                                               │    ResNet-18 / ViT / CNN │
                                               │    AdaptPolyKAN          │
                                               │                          │
                                               │  Multimodal Fusion       │
                                               │    Ensemble Prediction   │
                                               └──────────┬───────────────┘
                                                          │ Wi-Fi
                                                          ▼
                                               Remote Database + Dashboard
```

---

## Repository Structure

```
solar/
├── README.md
├── .gitignore
│
├── Fault_detect_V2/                        # Main multimodal / thermal 9-class system
│   ├── Solar_V2/
│   │   ├── CNN_Complete_Clean.ipynb        # Custom CNN training notebook
│   │   ├── Solar_CNN_ResNet.ipynb          # ResNet-18 training notebook
│   │   ├── train_efficientnet.py           # EfficientNet training script
│   │   ├── train_yolov8_v2.py             # YOLOv8m training script
│   │   ├── ensemble_predict.py             # Ensemble inference script
│   │   ├── evaluate_comprehensive.py       # Full evaluation & metrics
│   │   ├── export_for_deployment.py        # Export models for edge device
│   │   ├── config_v2.yaml                  # Dataset & training config
│   │   └── requirements.txt               # Python dependencies
│   │
│   ├── FD_AdaptPolyKAN/
│   │   └── Solar_AdaptPolyKAN_JupyterHub.ipynb   # KAN model experiment
│   │
│   └── FD_vision_transformer/
│       ├── Solar_Vision_Transformer.ipynb  # ViT experiment notebook
│       ├── main.py                         # Raspberry Pi inference entry-point
│       ├── setup_pi.sh                     # Raspberry Pi environment setup
│       └── test_thermal_camera.py          # Thermal camera hardware test
│
├── rgb_fault_detect/                       # RGB 4-class detection system
│   ├── RGB_4Class_YOLO_Complete.ipynb      # YOLOv11m training notebook
│   ├── raspberry_pi_rgb.py                 # Raspberry Pi RGB inference
│   ├── rgb_training_results.json           # Training metrics (JSON)
│   └── assessment_outputs/                 # Evaluation plots & confusion matrices
│       ├── dashboard.png
│       └── confusion_matrix.png
│
└── thermal_fault_detection/                # Thermal 6-class system (ResNet-18 / ViT)
    ├── Thermal_6Class_Complete_Training.ipynb
    ├── deployment_6class.json
    ├── thermal_6class_results.csv
    └── thermal_6class_plots/
```

> **Note:** Dataset images, model weight files (`.pt`, `.pth`), and training run artefacts (`runs_*/`) are **not included** in this repository due to size constraints. See [Datasets & Weights](#datasets--model-weights) below.

---

## Getting Started

### Prerequisites

#### Software
- Python 3.10+
- PyTorch 2.1+ (with CUDA if training on GPU)
- Ultralytics YOLO
- OpenCV, NumPy, Matplotlib, scikit-learn

Install all dependencies:

```bash
pip install -r Fault_detect_V2/Solar_V2/requirements.txt
```

#### Hardware (for edge deployment)
- Raspberry Pi 5 (8 GB recommended)
- FLIR Lepton or compatible thermal camera module
- RGB camera (Pi Camera Module 3 or USB camera)
- Parrot ANAFI Ai drone (for aerial capture)

---

## Training

### 1. Thermal 9-Class Models (Fault_detect_V2)

**Custom CNN / ResNet-18** — open in JupyterHub or Jupyter Lab:
```
Fault_detect_V2/Solar_V2/CNN_Complete_Clean.ipynb
Fault_detect_V2/Solar_V2/Solar_CNN_ResNet.ipynb
```

**EfficientNet** — run from command line:
```bash
python Fault_detect_V2/Solar_V2/train_efficientnet.py
```

**YOLOv8m** — run from command line:
```bash
python Fault_detect_V2/Solar_V2/train_yolov8_v2.py
```

**Vision Transformer (ViT)** — open notebook:
```
Fault_detect_V2/FD_vision_transformer/Solar_Vision_Transformer.ipynb
```

**AdaptPolyKAN** — open notebook:
```
Fault_detect_V2/FD_AdaptPolyKAN/Solar_AdaptPolyKAN_JupyterHub.ipynb
```

### 2. RGB 4-Class Model (rgb_fault_detect)

Open the notebook:
```
rgb_fault_detect/RGB_4Class_YOLO_Complete.ipynb
```

### 3. Thermal 6-Class System (thermal_fault_detection)

Open the notebook:
```
thermal_fault_detection/Thermal_6Class_Complete_Training.ipynb
```

---

## Inference & Deployment

### Raspberry Pi Setup

Run the setup script once on the Pi to install all dependencies:

```bash
bash Fault_detect_V2/FD_vision_transformer/setup_pi.sh
```

### Real-Time Thermal Inference (on Pi)

```bash
python Fault_detect_V2/FD_vision_transformer/main.py
```

### Real-Time RGB Inference (on Pi)

```bash
python rgb_fault_detect/raspberry_pi_rgb.py
```

### Test Thermal Camera

```bash
python Fault_detect_V2/FD_vision_transformer/test_thermal_camera.py
```

### Ensemble Prediction (multi-model fusion)

```bash
python Fault_detect_V2/Solar_V2/ensemble_predict.py
```

---

## Datasets & Model Weights

Dataset images and trained model weights are **not stored in this repository**.

| Artefact | Size | Download |
|----------|------|----------|
| Thermal 9-class dataset | Large | *(link to be added)* |
| RGB 4-class dataset | Large | *(link to be added)* |
| Thermal 6-class dataset | Large | *(link to be added)* |
| `best_cnn.pth` | ~50 MB | *(link to be added)* |
| `best_adaptpolykan.pth` | ~50 MB | *(link to be added)* |
| `best_vit_model.pth` | ~330 MB | *(link to be added)* |
| `best.pt` (YOLOv8m thermal) | ~50 MB | *(link to be added)* |
| `best_rgb_4class.pt` | ~50 MB | *(link to be added)* |
| `best_resnet18_6class.pth` | ~50 MB | *(link to be added)* |
| `best_vit_6class.pth` | ~330 MB | *(link to be added)* |

> Place downloaded weight files back into their original subdirectories (matching the paths used in the scripts) before running inference.

---

## Results Summary

Training was performed on Ashesi University's JupyterHub cluster (compute.ashesi.edu.gh).

| Model | Task | Classes | Top Metric |
|-------|------|---------|------------|
| YOLOv8m | Thermal detection | 9 | See `runs_v2/` |
| YOLOv11m | RGB detection | 4 | See `runs_rgb/` |
| ResNet-18 | Thermal classification | 6 | See `thermal_6class_results.csv` |
| ViT | Thermal classification | 6 | See `thermal_6class_results.csv` |
| EfficientNet | Thermal classification | 9 | See training logs |
| Custom CNN | Thermal classification | 9 | See `CNN_Complete_Clean.ipynb` |
| AdaptPolyKAN | Thermal classification | 9 | See `Solar_AdaptPolyKAN_JupyterHub.ipynb` |

Full evaluation metrics (accuracy, precision, recall, F1, confusion matrices) are available inside each notebook and in `rgb_fault_detect/assessment_outputs/`.

---

## Project Motivation

Ghana and sub-Saharan Africa are rapidly expanding solar energy capacity. Fault detection in large solar farms is critical for maximising energy yield and preventing costly panel degradation. This project demonstrates that affordable, drone-mounted edge AI systems can perform inspection tasks traditionally requiring expensive equipment or manual labour.

---

## Acknowledgements

- **Supervisor**: Dr Richard Akparibo Awingot, Ashesi University
- **Computing Resources**: Ashesi University JupyterHub (compute.ashesi.edu.gh)
- **Frameworks**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [PyTorch](https://pytorch.org/)
- **Hardware Platform**: Raspberry Pi Foundation, Parrot Drones

---

## License

This repository is submitted as an academic capstone project. All code is available for educational and research purposes. Please cite appropriately if used.

---

*Ashesi University — Department of Engineering — Capstone Project 2026*
