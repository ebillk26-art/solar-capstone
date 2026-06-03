# Design and Implementation of an Edge-Based Multimodal Deep Learning System for Fault Detection in Solar PV

> **BSc Capstone Project** | Electrical and Electronics Engineering | Ashesi University | 2026

---

## Author

| Field | Details |
|-------|---------|
| **Name** | Emmanuel Bilson |
| **Student ID** | 37402026 |
| **Programme** | BSc Electrical and Electronics Engineering |
| **Institution** | Ashesi University, Ghana |
| **Supervisor** | Dr Richard Akparibo Awingot |
| **Year** | 2026 |
| **Repository** | https://github.com/ebillk26-art/solar-capstone |

---

## Abstract

The rapid adoption of solar photovoltaic (PV) systems necessitates efficient and scalable fault detection methods. This project presents a multimodal edge-based system for real-time PV fault detection using both RGB and thermal imaging. RGB data detects physical defects such as soiling and cracks, while thermal data identifies electrical faults including hotspots and diode failures.

The proposed two-stream architecture — combining **YOLOv11m** and **ResNet-18** — outperforms single-modality baselines, achieving accuracies of **98.5%** and **94.3%** on RGB and thermal streams respectively. The system is deployed on a **Raspberry Pi 5** mounted on a **Parrot ANAFI Ai drone** for real-time aerial inspection.

---

## Problem Statement

Most existing fault detection studies rely on a single imaging modality (typically thermal) to detect both surface and electrical faults. This leads to inter-class confusion due to similar thermal signatures across different fault types. Additionally, most studies stop at reporting model accuracy without building and testing complete systems in real-world, resource-constrained conditions.

This project addresses both gaps by proposing a multimodal, two-pipeline edge system deployed on a drone platform in Ghana.

---

## System Architecture

The system is divided into three subsystems:

```
┌─────────────────────────────────────────────────────────────┐
│               DATA COLLECTION SUBSYSTEM                     │
│  RGB Camera (Pi Cam 3) + MLX90640 Thermal + NEO-6M GPS     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               COMPUTER VISION SUBSYSTEM                     │
│                  Raspberry Pi 5                             │
│  ┌──────────────────┐    ┌──────────────────────────────┐  │
│  │   RGB Stream     │    │      Thermal Stream          │  │
│  │  YOLOv11m-cls    │    │       ResNet-18              │  │
│  │  4-class:        │    │  6-class:                    │  │
│  │  Bird-drop       │    │  Cell-Fault, Diode-Fault     │  │
│  │  Clean           │    │  Hot-Spot, No-Anomaly        │  │
│  │  Dusty           │    │  Offline-Module, Shadowing   │  │
│  │  Physical-Damage │    │                              │  │
│  └──────────────────┘    └──────────────────────────────┘  │
│           Custom PCB HAT (Power + Peripherals)              │
└────────────────────────┬────────────────────────────────────┘
                         │ Wi-Fi
┌────────────────────────▼────────────────────────────────────┐
│               END USER SUBSYSTEM                            │
│            Database + Dashboard                             │
└─────────────────────────────────────────────────────────────┘
```

### Two-Stage Operation
1. **Panel Detection Stage** — The RGB stream identifies a solar panel in the camera's field of view
2. **Fault Classification Stage** — RGB and thermal images of the detected panel are passed simultaneously to the respective classification models

---

## Hardware Components

| Component | Part | Specification |
|-----------|------|---------------|
| Processing Unit | Raspberry Pi 5 | Quad-core ARM Cortex-A76, 2.4 GHz, 8 GB RAM |
| RGB Camera | Raspberry Pi Camera Module 3 | Standard visible light camera |
| Thermal Camera | MLX90640-D110 | 32×24 px, 110°×75° FOV, ±1°C to ±2.5°C accuracy |
| GPS Module | NEO-6M-0-001 | UART interface, NMEA format, trilateration positioning |
| Battery | 2S1P 18650 Li-ion | 7.4V nominal, 18.87 Wh, ~0.6 hr runtime at full load |
| Power Conversion | DC-DC Buck Converter | 5A, 85–90% efficiency, 7.4V → 5V |
| PCB | Custom Raspberry Pi HAT | 40-pin GPIO, battery management, peripheral connectors |
| Drone Platform | Parrot ANAFI Ai | Commercial UAV, tested at ~6m altitude |

**Total System Power Consumption:** 26.43 W (max, all sensors active)

---

## Fault Classes

### RGB Stream — 4 Physical Fault Classes
| Class | Description | Visual Signature |
|-------|-------------|-----------------|
| `Bird-drop` | Bird droppings on panel surface | Irregular white/dark patches |
| `Clean` | Healthy panel, no defects | Uniform, clear surface |
| `Dusty` | Dust/dirt accumulation | Thin layer covering the panel |
| `Physical-Damage` | Structural damage (cracks, broken glass) | Visible cracks or surface defects |

### Thermal Stream — 6 Electrical Fault Classes
| Class | Description | Thermal Signature |
|-------|-------------|------------------|
| `Cell-Fault` | Defect in individual solar cell | Localised thermal anomaly |
| `Diode-Fault` | Bypass diode failure | Linear/strip-like temperature patterns |
| `Hot-Spot` | Localised overheating | Intense bright region, high temperature |
| `No-Anomaly` | Healthy panel | Uniform temperature distribution |
| `Offline-Module` | Panel not producing power | Cooler or irregular thermal pattern |
| `Shadowing` | Partial sunlight obstruction | Uneven temperature with darker regions |

---

## Datasets

| Stream | Source | Original Classes | Classes Used | Images |
|--------|--------|-----------------|-------------|--------|
| RGB | Kaggle — "PV Panel Defect Dataset" | 6 | 4 (removed Electrical-damage, Snow-covered) | ~1,574 |
| Thermal | InfraredSolarModules dataset | 9 | 6 (removed Cracking, Soiling, Vegetation) | — |

> **Note:** Dataset images are **not included** in this repository due to size. Download links to be added.

---

## Models Evaluated

### Training Configuration (all models)
- Input resolution: 224×224 pixels
- Batch size: 32
- Max epochs: 100 (early stopping, patience=15)
- Optimizer: AdamW
- Fine-tuning: Full (all layers)
- Training platform: Ashesi University JupyterHub (compute.ashesi.edu.gh)

### Baseline (Single-Modality, All Fault Types)
| Model | Accuracy | Notes |
|-------|----------|-------|
| YOLOv8m | 77.4% | Struggled with hotspots and diode failures |
| Vision Transformer (ViT) | 86.0% | Better but still limited with single modality |

These baselines motivated the multimodal two-stream approach.

### RGB Stream Results
| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| YOLOv8m-cls | 93.9% | 94.2% | 94.0% |
| **YOLOv11m-cls ✓** | **98.5%** | **98.5%** | **98.6%** |

**Selected: YOLOv11m** — only 1 misclassification on test set (clean → dusty)

### Thermal Stream Results
| Model | Accuracy |
|-------|----------|
| Vision Transformer (ViT-Base) | 77.68% |
| YOLOv8m-cls | 81.79% |
| YOLOv11m-cls | 81.94% |
| **ResNet-18 ✓** | **94.29%** |

**Selected: ResNet-18** — Per-class accuracy: No-Anomaly 98.60%, Cell-Fault 95.30%, Diode-Fault 94.08%

### Additional Experiments (in `Base Model/` folder)
- EfficientNet training on thermal data
- AdaptPolyKAN (KAN-based architecture)
- Extended ViT experiments
- Ensemble prediction pipeline

---

## Repository Structure

```
solar-capstone/
├── README.md
├── .gitignore
│
├── Base Model/                             # Main multimodal system + experiments
│   ├── Solar_V2/
│   │   ├── CNN_Complete_Clean.ipynb        # Custom CNN training notebook
│   │   ├── Solar_CNN_ResNet.ipynb          # ResNet-18 training notebook
│   │   ├── train_efficientnet.py           # EfficientNet training script
│   │   ├── train_yolov8_v2.py             # YOLOv8m training script
│   │   ├── ensemble_predict.py             # Ensemble inference script
│   │   ├── evaluate_comprehensive.py       # Evaluation & metrics
│   │   ├── export_for_deployment.py        # Export models for edge device
│   │   ├── config_v2.yaml                  # Dataset and training configuration
│   │   └── requirements.txt               # Python dependencies
│   │
│   ├── FD_AdaptPolyKAN/
│   │   └── Solar_AdaptPolyKAN_JupyterHub.ipynb   # KAN architecture experiment
│   │
│   └── FD_vision_transformer/
│       ├── Solar_Vision_Transformer.ipynb  # ViT experiment notebook
│       ├── main.py                         # Raspberry Pi inference entry point
│       ├── setup_pi.sh                     # Raspberry Pi environment setup
│       └── test_thermal_camera.py          # MLX90640 hardware test
│
├── rgb_fault_detect/                       # RGB 4-class stream (YOLOv11m)
│   ├── RGB_4Class_YOLO_Complete.ipynb      # Full training notebook
│   ├── raspberry_pi_rgb.py                 # Raspberry Pi RGB inference
│   ├── rgb_training_results.json           # Training metrics
│   └── assessment_outputs/                 # Evaluation plots
│       ├── dashboard.png
│       └── confusion_matrix.png
│
└── thermal_fault_detection/                # Thermal 6-class stream (ResNet-18)
    ├── Thermal_6Class_Complete_Training.ipynb
    ├── deployment_6class.json
    └── thermal_6class_results.csv
```

> **Not included in this repo:** Dataset images, trained model weights (`.pt`, `.pth`, `.onnx`), and YOLO training run folders (`runs_*/`). See [Datasets & Weights](#datasets--model-weights) section above.

---

## Getting Started

### Prerequisites

```bash
pip install -r "Base Model/Solar_V2/requirements.txt"
```

Key dependencies: PyTorch 2.1+, Ultralytics YOLO, OpenCV, NumPy, scikit-learn, Matplotlib

### Raspberry Pi Setup

Run once on the Raspberry Pi 5:

```bash
bash "Base Model/FD_vision_transformer/setup_pi.sh"
```

### Test Thermal Camera (MLX90640)

```bash
python "Base Model/FD_vision_transformer/test_thermal_camera.py"
```

---

## Running Inference

### Real-Time Multimodal Inference on Raspberry Pi

```bash
python "Base Model/FD_vision_transformer/main.py"
```

This launches both the RGB and thermal streams simultaneously. Output includes:
- Panel detection confidence
- Physical fault class + confidence (RGB)
- Electrical fault class + confidence (Thermal)
- Temperature range from MLX90640
- GPS coordinates logged per detection

### RGB Stream Only

```bash
python rgb_fault_detect/raspberry_pi_rgb.py
```

### Ensemble Prediction

```bash
python "Base Model/Solar_V2/ensemble_predict.py"
```

---

## Training

### RGB 4-Class Model
Open and run the notebook:
```
rgb_fault_detect/RGB_4Class_YOLO_Complete.ipynb
```

### Thermal 6-Class Models (ResNet-18, ViT, YOLO)
```
thermal_fault_detection/Thermal_6Class_Complete_Training.ipynb
```

### Additional Experiments
```
Base Model/Solar_V2/CNN_Complete_Clean.ipynb
Base Model/Solar_V2/Solar_CNN_ResNet.ipynb
Base Model/FD_AdaptPolyKAN/Solar_AdaptPolyKAN_JupyterHub.ipynb
Base Model/FD_vision_transformer/Solar_Vision_Transformer.ipynb
```

### EfficientNet Training
```bash
python "Base Model/Solar_V2/train_efficientnet.py"
```

---

## Real-World Testing Results

The system was tested by mounting the device on a Parrot ANAFI Ai drone flown at approximately 6 metres altitude. Key observations:

| Test | Result |
|------|--------|
| Clean panel detection | 100.0% confidence |
| Dusty panel detection | 93.4% confidence |
| Physical damage detection | 87.8% confidence |
| Thermal stream FPS on Pi 5 | ~1.2 FPS |
| RGB stream FPS on Pi 5 | ~1.3 FPS |
| System response time | ~1 second per inspection |

**Key finding:** Physical damage to a panel produces both a visible RGB defect AND abnormal thermal signatures (Shadowing 52%, Cell-Fault 24%), validating the value of the multimodal approach.

---

## PCB Design

A custom Raspberry Pi HAT was designed to integrate power distribution, battery management, and peripheral connections:

- **Schematic:** Designed in EasyEDA
- **Design selected:** Design B (external battery ports, 1S2P configuration)
- **Features:** 40-pin GPIO header, 5A buck converter, rocker switch with reverse polarity protection (1N4007 diode), LED power indicator, connectors for GPS and thermal camera
- **Battery:** Two 18650 cells externally connected via terminal block

---

## Project Motivation

Ghana and sub-Saharan Africa are rapidly expanding solar energy capacity. As solar parks scale to hundreds of megawatts, manual inspection becomes impractical. This project demonstrates that an affordable, drone-mounted edge AI system can perform panel-level fault inspection in real time — without cloud connectivity — making reliable solar farm maintenance accessible in low-resource environments.

---

## Acknowledgements

- **Supervisor:** Dr Richard Akparibo Awingot, Department of Engineering, Ashesi University
- **Computing Resources:** Ashesi University JupyterHub (compute.ashesi.edu.gh)
- **Frameworks:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [PyTorch](https://pytorch.org/)
- **Datasets:** Kaggle PV Panel Defect Dataset, InfraredSolarModules dataset
- **Hardware:** Raspberry Pi Foundation, Parrot Drones, Waveshare (MLX90640)

---

## License

This repository is submitted as an academic capstone project at Ashesi University. Code is available for educational and research purposes. Please cite appropriately if used.

---

*Ashesi University — Department of Engineering — Capstone Project 2026*
