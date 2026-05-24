# Solar Panel Fault Detection v2.0 🌞⚡
## Complete Rewrite with 85-90% Accuracy Target

---

## 🎯 **What's New in v2.0**

| Feature | v1.0 | v2.0 | Improvement |
|---------|------|------|-------------|
| **Accuracy** | 77.4% | **85-90%** (target) | **+10-15%** |
| **Classes** | 12 | 9 (optimized) | Merged similar |
| **Model** | YOLOv8-medium | YOLOv8-large + EfficientNet | Ensemble |
| **Training** | Basic | Advanced pipeline | Systematic |
| **Evaluation** | Simple | Comprehensive | Full metrics |
| **Deployment** | Manual | Automated export | RPi-ready |

---

## 📋 **Table of Contents**

1. [Quick Start](#quick-start)
2. [Class Changes](#class-changes)
3. [Complete Workflow](#complete-workflow)
4. [File Overview](#file-overview)
5. [Expected Results](#expected-results)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 **Quick Start**

### **Prerequisites:**
```bash
pip install ultralytics torch torchvision pillow numpy pyyaml matplotlib seaborn scikit-learn tqdm
```

### **5-Step Process:**

```bash
# Step 1: Reorganize data (merge classes)
python reorganize_dataset.py

# Step 2: Train YOLOv8-large
python train_yolov8_v2.py  # ~3-4 hours

# Step 3: Train EfficientNet
python train_efficientnet.py  # ~4-6 hours

# Step 4: Create ensemble & evaluate
python ensemble_predict.py
python evaluate_comprehensive.py

# Step 5: Export for Raspberry Pi
python export_for_deployment.py
```

**Total Time:** 1-2 days of training

---

## 📊 **Class Changes (12 → 9)**

### **Merged Classes:**

| Original Classes | New Combined Class | Rationale |
|-----------------|-------------------|-----------|
| Cell + Cell-Multi | **Cell-Fault** | Both are cell hotspots (14.4% confusion) |
| Hot-Spot + Hot-Spot-Multi | **Hot-Spot** | Both thin film issues (14.6% confusion) |
| Diode + Diode-Multi | **Diode-Fault** | Both bypass diode failures |

### **Kept As-Is:**
- Cracking
- Soiling
- Shadowing
- Vegetation
- Offline-Module
- No-Anomaly

**Why Merge?**
- Reduces confusion between similar classes
- Simplifies classification task
- Still provides actionable fault information
- **Expected: +5-8% accuracy improvement**

---

## 📁 **File Overview**

### **Configuration:**
- `config_v2.yaml` - All settings in one place

### **Data Preparation:**
- `reorganize_dataset.py` - Merges classes, creates new dataset structure

### **Training Scripts:**
- `train_yolov8_v2.py` - Improved YOLOv8 training
- `train_efficientnet.py` - EfficientNet training

### **Prediction & Evaluation:**
- `ensemble_predict.py` - Combines both models
- `evaluate_comprehensive.py` - Full metrics & analysis

### **Deployment:**
- `export_for_deployment.py` - Exports for Raspberry Pi

---

## 🔄 **Complete Workflow**

### **Phase 1: Data Preparation (5 minutes)**

```bash
python reorganize_dataset.py
```

**What it does:**
- Reads original 12-class dataset
- Merges similar classes
- Creates new `data_v2/images/` structure
- Generates statistics report

**Output:**
```
data_v2/
└── images/
    ├── train/
    │   ├── Cell-Fault/
    │   ├── Cracking/
    │   └── ... (9 classes)
    ├── val/
    └── test/
```

---

### **Phase 2: Training (1-2 days)**

#### **2A: YOLOv8 Training (3-4 hours)**

```bash
python train_yolov8_v2.py
```

**Configuration** (edit `config_v2.yaml`):
```yaml
training:
  models:
    yolov8:
      size: "l"  # Large model
  img_size: 640
  batch_size: 16
  epochs: 200
```

**Expected:** 83-87% accuracy

#### **2B: EfficientNet Training (4-6 hours)**

```bash
python train_efficientnet.py
```

**Expected:** 82-86% accuracy

---

### **Phase 3: Ensemble & Evaluation (10 minutes)**

```bash
# Combine models
python ensemble_predict.py

# Comprehensive evaluation
python evaluate_comprehensive.py
```

**Generates:**
- Overall accuracy
- Per-class metrics
- Confusion matrix
- Category analysis (Physical/Electrical/Normal)
- Comparison with v1.0

**Expected Ensemble:** 85-90% accuracy

---

### **Phase 4: Deployment (5 minutes)**

```bash
python export_for_deployment.py
```

**Creates:**
- `best.tflite` (optimized for Raspberry Pi)
- `deployment_v2/` package
- Inference scripts
- Documentation

---

## 📊 **Expected Results**

### **Performance Targets:**

| Metric | v1.0 | v2.0 Target | Improvement |
|--------|------|-------------|-------------|
| Overall Accuracy | 77.4% | 85-90% | +10-15% |
| Physical Faults | ~75% | 85-90% | +10-15% |
| Electrical Faults | ~70% | 80-85% | +10-15% |
| Normal Detection | ~95% | 95-98% | Maintained |

### **Contributing Factors:**

1. **Class Merging:** +5-8%
2. **Larger Model:** +3-5%
3. **EfficientNet Addition:** +2-3%
4. **Ensemble:** +2-3%
5. **Better Training:** +1-2%

**Total: +13-21% improvement**

---

## 🥧 **Raspberry Pi Deployment**

### **Files Needed:**
- `best.tflite` (YOLOv8 model, ~40-50 MB)
- `inference.py` (prediction script)
- `class_mapping.json` (class names)

### **On Raspberry Pi:**

```bash
# Install dependencies
pip3 install ultralytics pillow numpy --break-system-packages

# Run inference
python3 inference.py test_image.jpg
```

### **Expected Performance:**
- **Accuracy:** 85-90%
- **Speed:** 200-500ms per image (RPi 4)
- **Memory:** ~2-3 GB RAM

---

## 🔧 **Configuration Guide**

### **Key Settings in `config_v2.yaml`:**

#### **For Better Accuracy:**
```yaml
training:
  models:
    yolov8:
      size: "x"  # Extra-large (best accuracy, slower)
  img_size: 1024  # Higher resolution
  epochs: 250  # More training
```

#### **For Faster Training:**
```yaml
training:
  models:
    yolov8:
      size: "m"  # Medium (faster)
  img_size: 416  # Smaller images
  epochs: 100  # Fewer epochs
```

#### **For Raspberry Pi Optimization:**
```yaml
deployment:
  raspberry_pi:
    target_img_size: 320  # Smaller for speed
    quantization: "int8"  # INT8 for speed
```

---

## 📈 **Monitoring Training**

### **Check Progress:**

```bash
# YOLOv8
tensorboard --logdir runs_v2/yolov8

# View results
cat runs_v2/yolov8/solar_yolov8l/results.txt
```

### **What to Look For:**
- Training loss decreasing
- Validation accuracy increasing
- No overfitting (train/val gap < 5%)

---

## 🐛 **Troubleshooting**

### **Issue: Out of Memory**
```yaml
# Reduce in config_v2.yaml:
batch_size: 8  # From 16
img_size: 416  # From 640
```

### **Issue: Training Too Slow**
- Ensure GPU is being used
- Reduce `workers` in config
- Use smaller model (`yolov8s` instead of `yolov8l`)

### **Issue: Low Accuracy (<80%)**
1. Check data quality: `python reorganize_dataset.py`
2. Train longer: increase `epochs` in config
3. Use larger model: `yolov8x` instead of `yolov8l`
4. Check class imbalance in statistics

---

## 📊 **Comparison: v1.0 vs v2.0**

### **What Changed:**

**Architecture:**
- v1.0: Single YOLOv8-medium
- v2.0: YOLOv8-large + EfficientNet ensemble

**Classes:**
- v1.0: 12 classes (with confusion)
- v2.0: 9 classes (optimized merging)

**Training:**
- v1.0: Basic configuration
- v2.0: Advanced hyperparameters, augmentation

**Evaluation:**
- v1.0: Simple accuracy
- v2.0: Comprehensive metrics, category analysis

**Deployment:**
- v1.0: Manual export
- v2.0: Automated pipeline

---

## 🎓 **Best Practices**

### **Data Quality:**
1. Run `python reorganize_dataset.py` first
2. Check class distribution
3. Remove corrupted images
4. Ensure balanced classes

### **Training:**
1. Start with default config
2. Monitor training curves
3. Use early stopping
4. Save checkpoints regularly

### **Evaluation:**
1. Always evaluate on validation set
2. Check per-class metrics
3. Analyze confusion matrix
4. Test on real images

### **Deployment:**
1. Export to TFLite
2. Test on target hardware
3. Optimize for speed vs accuracy
4. Monitor inference time

---

## 📝 **Citation**

If using this code or the InfraredSolarModules dataset:

```
@inproceedings{millendorf2020infrared,
  title={Infrared Solar Module Dataset for Anomaly Detection},
  author={Millendorf, Matthew and Obropta, Edward and Vadhavkar, Nikhil},
  booktitle={ICLR 2020 AI for Earth Sciences Workshop},
  year={2020}
}
```

---

## 🆘 **Support & Contact**

**Issues:**
1. Check [Troubleshooting](#troubleshooting) section
2. Review training logs
3. Verify data structure
4. Check GPU availability

**Questions:**
- Review `config_v2.yaml` comments
- Check individual script docstrings
- Refer to ICLR 2020 paper

---

## ✅ **Final Checklist**

Before deployment:
- [ ] Data reorganized and verified
- [ ] YOLOv8 trained (>83% accuracy)
- [ ] EfficientNet trained (>82% accuracy)
- [ ] Ensemble evaluated (>85% accuracy)
- [ ] Models exported to TFLite
- [ ] Tested on sample images
- [ ] Raspberry Pi setup verified

---

## 🚀 **Quick Reference Card**

```bash
# Complete Pipeline
python reorganize_dataset.py         # 5 min
python train_yolov8_v2.py           # 3-4 hours
python train_efficientnet.py        # 4-6 hours
python ensemble_predict.py          # 5 min
python evaluate_comprehensive.py    # 5 min
python export_for_deployment.py     # 5 min

# Expected Results
# - Accuracy: 85-90%
# - Time: 1-2 days
# - Output: RPi-ready models
```

---

**Good luck with v2.0!** 🎉
**Target: 85-90% accuracy** 🎯
**Ready for production!** 🚀
