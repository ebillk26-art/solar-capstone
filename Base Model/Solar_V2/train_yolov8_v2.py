"""
Solar Panel Fault Detection v2.0 - YOLOv8 Training
===================================================

Improved YOLOv8 training with:
- Larger model (yolov8l-cls)
- Merged classes (9 instead of 12)
- Better hyperparameters
- Comprehensive logging

Expected: 83-87% accuracy
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
from datetime import datetime
import json
import numpy as np

class ImprovedYOLOv8Trainer:
    """Improved YOLOv8 trainer for solar panel fault detection"""
    
    def __init__(self, config_path='config_v2.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = self.config['data']['path']
        self.model_size = self.config['training']['models']['yolov8']['size']
        self.device = self.check_gpu()
        
        # Training settings
        self.img_size = self.config['training']['img_size']
        self.batch_size = self.config['training']['batch_size']
        self.epochs = self.config['training']['epochs']
        self.patience = self.config['training']['patience']
        
        print("="*80)
        print("YOLOV8 TRAINING - v2.0")
        print("="*80)
        print(f"Model: yolov8{self.model_size}-cls")
        print(f"Data: {self.data_path}")
        print(f"Classes: {self.config['data']['merged_classes']}")
        print(f"Image Size: {self.img_size}px")
        print(f"Epochs: {self.epochs}")
        print("="*80)
    
    def check_gpu(self):
        """Check GPU availability"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            return 0
        else:
            print("\n⚠️  No GPU detected!")
            return 'cpu'
    
    def analyze_dataset(self):
        """Analyze new merged dataset"""
        print("\n📊 Analyzing Dataset...")
        
        data_path = Path(self.data_path)
        train_path = data_path / 'train'
        val_path = data_path / 'val'
        
        stats = {}
        total_train = 0
        total_val = 0
        
        print(f"\n{'Class':<30} | {'Train':>8} | {'Val':>8} | {'Total':>8}")
        print("-"*70)
        
        for class_dir in sorted(train_path.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                train_count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
                
                val_class_dir = val_path / class_name
                val_count = 0
                if val_class_dir.exists():
                    val_count = len(list(val_class_dir.glob('*.jpg'))) + len(list(val_class_dir.glob('*.png')))
                
                total = train_count + val_count
                stats[class_name] = {'train': train_count, 'val': val_count, 'total': total}
                
                total_train += train_count
                total_val += val_count
                
                print(f"{class_name:<30} | {train_count:>8,} | {val_count:>8,} | {total:>8,}")
        
        print("-"*70)
        print(f"{'TOTAL':<30} | {total_train:>8,} | {total_val:>8,} | {total_train+total_val:>8,}")
        print("-"*70)
        
        return stats
    
    def calculate_class_weights(self, stats):
        """Calculate class weights for imbalanced dataset"""
        if not self.config['training']['use_class_weights']:
            return None
        
        print("\n⚖️  Calculating class weights...")
        
        # Get training counts
        train_counts = {cls: data['train'] for cls, data in stats.items()}
        total = sum(train_counts.values())
        n_classes = len(train_counts)
        
        # Inverse frequency weighting
        weights = {}
        for cls, count in train_counts.items():
            if count > 0:
                weights[cls] = total / (n_classes * count)
            else:
                weights[cls] = 0.0
        
        # Normalize
        max_weight = max(weights.values())
        weights = {cls: w/max_weight for cls, w in weights.items()}
        
        print("\nClass Weights:")
        for cls in sorted(weights.keys()):
            print(f"   {cls:<30}: {weights[cls]:.3f}")
        
        return weights
    
    def train(self):
        """Train YOLOv8 model"""
        
        # Analyze dataset
        stats = self.analyze_dataset()
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(stats)
        
        # Initialize model
        model_name = f'yolov8{self.model_size}-cls.pt'
        print(f"\n🤖 Loading {model_name}...")
        model = YOLO(model_name)
        
        print(f"\n🚀 Starting Training...")
        print(f"   Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Training configuration
        cfg = self.config['training']
        aug = cfg['augmentation']
        
        results = model.train(
            # Data
            data=self.data_path,
            
            # Training params
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
            
            # Optimizer
            optimizer=cfg['optimizer'],
            lr0=cfg['lr_initial'],
            lrf=cfg['lr_final'] / cfg['lr_initial'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
            
            # Augmentation (thermal-optimized)
            hsv_h=aug['hsv_h'],
            hsv_s=aug['hsv_s'],
            hsv_v=aug['hsv_v'],
            degrees=aug['degrees'],
            translate=aug['translate'],
            scale=aug['scale'],
            shear=aug['shear'],
            perspective=aug['perspective'],
            flipud=aug['flipud'],
            fliplr=aug['fliplr'],
            mosaic=aug['mosaic'],
            mixup=aug['mixup'],
            copy_paste=aug['copy_paste'],
            
            # Learning rate schedule
            warmup_epochs=cfg['lr_schedule']['warmup_epochs'],
            warmup_momentum=0.8,
            warmup_bias_lr=cfg['lr_schedule']['warmup_bias_lr'],
            
            # Training settings
            patience=self.patience,
            save=True,
            save_period=10,
            
            # Validation
            val=True,
            plots=True,
            
            # Performance
            workers=self.config['hardware']['workers'],
            amp=self.config['hardware']['mixed_precision'],
            cache=False,
            
            # Logging
            project='runs_v2/yolov8',
            name='solar_yolov8l',
            exist_ok=True,
            verbose=True,
        )
        
        print(f"\n✅ Training Complete!")
        print(f"   End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Best model: runs_v2/yolov8/solar_yolov8l/weights/best.pt")
        
        return model, results
    
    def quick_evaluation(self, model):
        """Quick evaluation on validation set"""
        print("\n📊 Quick Validation Check...")
        
        val_path = Path(self.data_path) / 'val'
        classes = sorted([d.name for d in val_path.iterdir() if d.is_dir()])
        
        correct = 0
        total = 0
        
        for class_idx, class_name in enumerate(classes):
            class_dir = val_path / class_name
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            for img_path in images[:50]:  # Sample 50 per class
                try:
                    result = model(str(img_path), verbose=False)
                    probs = result[0].probs.data.cpu().numpy()
                    pred_idx = np.argmax(probs)
                    
                    if pred_idx == class_idx:
                        correct += 1
                    total += 1
                except:
                    pass
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n   Quick Val Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return accuracy


def main():
    """Main execution"""
    
    # Initialize trainer
    trainer = ImprovedYOLOv8Trainer(config_path='config_v2.yaml')
    
    # Train
    model, results = trainer.train()
    
    # Quick evaluation
    if model:
        accuracy = trainer.quick_evaluation(model)
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Model: YOLOv8-large")
        print(f"Quick Validation: {accuracy:.2%}")
        print(f"Model saved: runs_v2/yolov8/solar_yolov8l/weights/best.pt")
        print("\n📊 Next Steps:")
        print("   1. Run: python train_efficientnet.py")
        print("   2. Run: python ensemble_predict.py")
        print("   3. Run: python evaluate_comprehensive.py")
        print("="*80)


if __name__ == "__main__":
    main()
