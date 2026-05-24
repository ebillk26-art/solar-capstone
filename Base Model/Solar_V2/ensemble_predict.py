"""
Solar Panel Fault Detection v2.0 - Ensemble Prediction
=======================================================

Combines YOLOv8 and EfficientNet predictions for best accuracy

Strategy: Weighted average of probabilities
Expected: +2-3% improvement over single models
"""

from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from pathlib import Path
from PIL import Image
import numpy as np
import yaml
from tqdm import tqdm


class EnsemblePredictor:
    """Ensemble of YOLOv8 and EfficientNet"""
    
    def __init__(self, 
                 yolo_path='runs_v2/yolov8/solar_yolov8l/weights/best.pt',
                 efficientnet_path='runs_v2/efficientnet/best_model.pth',
                 config_path='config_v2.yaml'):
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.num_classes = self.config['data']['merged_classes']
        self.class_names = self.config['data']['class_names']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensemble weights
        weights = self.config['ensemble']['weights']
        self.yolo_weight = weights['yolov8']
        self.efficientnet_weight = weights['efficientnet']
        
        print("="*80)
        print("ENSEMBLE PREDICTOR - v2.0")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"YOLOv8 weight: {self.yolo_weight}")
        print(f"EfficientNet weight: {self.efficientnet_weight}")
        
        # Load models
        self.load_models(yolo_path, efficientnet_path)
    
    def load_models(self, yolo_path, efficientnet_path):
        """Load both models"""
        print("\n📦 Loading models...")
        
        # Load YOLOv8
        if Path(yolo_path).exists():
            self.yolo_model = YOLO(yolo_path)
            print(f"   ✅ YOLOv8 loaded from {yolo_path}")
        else:
            print(f"   ❌ YOLOv8 not found at {yolo_path}")
            self.yolo_model = None
        
        # Load EfficientNet
        if Path(efficientnet_path).exists():
            checkpoint = torch.load(efficientnet_path, map_location=self.device)
            
            self.efficientnet_model = efficientnet_v2_s()
            num_ftrs = self.efficientnet_model.classifier[1].in_features
            self.efficientnet_model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)
            self.efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
            self.efficientnet_model = self.efficientnet_model.to(self.device)
            self.efficientnet_model.eval()
            print(f"   ✅ EfficientNet loaded from {efficientnet_path}")
        else:
            print(f"   ❌ EfficientNet not found at {efficientnet_path}")
            self.efficientnet_model = None
        
        # Transform for EfficientNet
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_single(self, image_path):
        """Predict single image with ensemble"""
        
        predictions = {}
        
        # YOLOv8 prediction
        if self.yolo_model:
            result = self.yolo_model(image_path, verbose=False)[0]
            yolo_probs = result.probs.data.cpu().numpy()
            predictions['yolo'] = yolo_probs
        
        # EfficientNet prediction
        if self.efficientnet_model:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.efficientnet_model(image_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            predictions['efficientnet'] = probs
        
        # Ensemble (weighted average)
        if 'yolo' in predictions and 'efficientnet' in predictions:
            ensemble_probs = (
                self.yolo_weight * predictions['yolo'] +
                self.efficientnet_weight * predictions['efficientnet']
            )
        elif 'yolo' in predictions:
            ensemble_probs = predictions['yolo']
        elif 'efficientnet' in predictions:
            ensemble_probs = predictions['efficientnet']
        else:
            return None
        
        # Get prediction
        pred_idx = np.argmax(ensemble_probs)
        pred_class = self.class_names[pred_idx]
        confidence = ensemble_probs[pred_idx]
        
        return {
            'class': pred_class,
            'confidence': float(confidence),
            'index': int(pred_idx),
            'probabilities': ensemble_probs.tolist(),
            'individual_predictions': {
                'yolo': predictions.get('yolo', None),
                'efficientnet': predictions.get('efficientnet', None)
            }
        }
    
    def evaluate(self, data_path='data_v2/images', split='val'):
        """Evaluate ensemble on validation set"""
        print(f"\n📊 Evaluating ensemble on {split} set...")
        
        val_path = Path(data_path) / split
        
        y_true = []
        y_pred = []
        
        for class_idx, class_name in enumerate(tqdm(self.class_names, desc="Processing")):
            class_dir = val_path / class_name
            if not class_dir.exists():
                continue
            
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            for img_path in images:
                try:
                    result = self.predict_single(str(img_path))
                    if result:
                        y_true.append(class_idx)
                        y_pred.append(result['index'])
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Calculate accuracy
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        accuracy = np.mean(y_true == y_pred)
        
        print(f"\n{'='*80}")
        print(f"ENSEMBLE RESULTS")
        print(f"{'='*80}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Evaluated: {len(y_true):,} images")
        print(f"{'='*80}")
        
        return accuracy, y_true, y_pred


def main():
    """Main execution"""
    
    predictor = EnsemblePredictor()
    
    # Evaluate
    accuracy, y_true, y_pred = predictor.evaluate()
    
    print("\n" + "="*80)
    print("ENSEMBLE SUMMARY")
    print("="*80)
    print(f"Ensemble Accuracy: {accuracy*100:.2f}%")
    print("\nThis combines:")
    print("  - YOLOv8-large")
    print("  - EfficientNetV2-S")
    print("\n📊 Next Steps:")
    print("   1. Run: python evaluate_comprehensive.py")
    print("   2. Run: python export_for_deployment.py")
    print("="*80)


if __name__ == "__main__":
    main()
