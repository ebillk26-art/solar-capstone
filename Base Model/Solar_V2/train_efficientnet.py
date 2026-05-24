"""
Solar Panel Fault Detection v2.0 - EfficientNet Training
=========================================================

Trains EfficientNetV2 for comparison with YOLOv8
State-of-the-art architecture for image classification

Expected: 82-86% accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from pathlib import Path
from PIL import Image
import yaml
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json

class SolarDataset(Dataset):
    """Custom dataset for solar panel images"""
    
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = Path(data_path) / split
        self.transform = transform
        
        # Get all images and labels
        self.images = []
        self.labels = []
        self.classes = sorted([d.name for d in self.data_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = self.data_path / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(class_idx)
            for img_path in class_dir.glob('*.png'):
                self.images.append(img_path)
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class EfficientNetTrainer:
    """EfficientNet trainer"""
    
    def __init__(self, config_path='config_v2.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = self.config['data']['path']
        self.num_classes = self.config['data']['merged_classes']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training params
        self.img_size = self.config['training']['img_size']
        self.batch_size = self.config['training']['batch_size']
        self.epochs = self.config['training']['epochs']
        self.lr = self.config['training']['lr_initial']
        
        print("="*80)
        print("EFFICIENTNET TRAINING - v2.0")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Classes: {self.num_classes}")
        print(f"Image Size: {self.img_size}px")
        print(f"Batch Size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print("="*80)
    
    def get_transforms(self, split='train'):
        """Get data transforms"""
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def create_dataloaders(self):
        """Create data loaders"""
        print("\n📊 Creating dataloaders...")
        
        train_dataset = SolarDataset(
            self.data_path, 
            split='train',
            transform=self.get_transforms('train')
        )
        
        val_dataset = SolarDataset(
            self.data_path,
            split='val',
            transform=self.get_transforms('val')
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"   Train: {len(train_dataset):,} images")
        print(f"   Val: {len(val_dataset):,} images")
        
        return train_loader, val_loader, train_dataset.classes
    
    def create_model(self):
        """Create EfficientNet model"""
        print("\n🤖 Creating EfficientNetV2...")
        
        # Load pretrained model
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        # Modify final layer
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)
        
        model = model.to(self.device)
        
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def train(self):
        """Train model"""
        
        # Create dataloaders
        train_loader, val_loader, classes = self.create_dataloaders()
        
        # Create model
        model = self.create_model()
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=self.config['training']['lr_final']
        )
        
        print(f"\n🚀 Training Started...")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'\nEpoch {epoch+1}/{self.epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                Path('runs_v2/efficientnet').mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_acc,
                    'classes': classes,
                }, 'runs_v2/efficientnet/best_model.pth')
                print(f'  ✅ New best model saved! (Val Acc: {val_acc:.2f}%)')
        
        print(f"\n✅ Training Complete!")
        print(f"   Best Val Accuracy: {best_acc:.2f}%")
        print(f"   Model saved: runs_v2/efficientnet/best_model.pth")
        
        # Save history
        with open('runs_v2/efficientnet/history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return model, best_acc, history


def main():
    """Main execution"""
    
    trainer = EfficientNetTrainer()
    model, best_acc, history = trainer.train()
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Model: EfficientNetV2-S")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved: runs_v2/efficientnet/best_model.pth")
    print("\n📊 Next Steps:")
    print("   1. Run: python ensemble_predict.py")
    print("   2. Run: python evaluate_comprehensive.py")
    print("="*80)


if __name__ == "__main__":
    main()
