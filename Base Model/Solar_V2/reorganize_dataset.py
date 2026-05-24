"""
Solar Panel Fault Detection v2.0 - Dataset Reorganization
==========================================================

Merges similar classes for better accuracy:
- 12 classes → 9 optimized classes
- Better class balance
- Higher expected accuracy (85-90%)

MERGED CLASSES:
- Cell + Cell-Multi → Cell-Fault
- Hot-Spot + Hot-Spot-Multi → Hot-Spot
- Diode + Diode-Multi → Diode-Fault
"""

import shutil
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime

class DatasetReorganizer:
    """Reorganizes dataset with merged classes"""
    
    def __init__(self, 
                 source_data='data/images',
                 target_data='data_v2/images'):
        self.source_data = Path(source_data)
        self.target_data = Path(target_data)
        
        # Define class merging strategy
        self.class_mapping = {
            # Merge Cell classes
            'Cell': 'Cell-Fault',
            'Cell-Multi': 'Cell-Fault',
            
            # Merge Hot-Spot classes (thin film modules)
            'Hot-Spot': 'Hot-Spot',
            'Hot-Spot-Multi': 'Hot-Spot',
            
            # Merge Diode classes
            'Diode': 'Diode-Fault',
            'Diode-Multi': 'Diode-Fault',
            
            # Keep these as-is
            'Cracking': 'Cracking',
            'Soiling': 'Soiling',
            'Shadowing': 'Shadowing',
            'Vegetation': 'Vegetation',
            'Offline-Module': 'Offline-Module',
            'No-Anomaly': 'No-Anomaly',
        }
        
        self.stats = defaultdict(lambda: defaultdict(int))
        
    def create_directory_structure(self):
        """Create new directory structure"""
        print("\n📁 Creating new directory structure...")
        
        for split in ['train', 'val', 'test']:
            for new_class in set(self.class_mapping.values()):
                target_dir = self.target_data / split / new_class
                target_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ Directory structure created!")
        
    def reorganize_split(self, split='train'):
        """Reorganize one split (train/val/test)"""
        print(f"\n🔄 Reorganizing {split} split...")
        
        source_split = self.source_data / split
        if not source_split.exists():
            print(f"⚠️  {split} directory not found, skipping...")
            return
        
        # Process each original class
        for old_class_dir in sorted(source_split.iterdir()):
            if not old_class_dir.is_dir():
                continue
            
            old_class = old_class_dir.name
            
            # Get new class name
            if old_class not in self.class_mapping:
                print(f"⚠️  Unknown class: {old_class}, skipping...")
                continue
            
            new_class = self.class_mapping[old_class]
            
            # Get all images
            images = list(old_class_dir.glob('*.jpg')) + list(old_class_dir.glob('*.png'))
            
            # Copy to new location
            target_class_dir = self.target_data / split / new_class
            
            for img_path in images:
                # Create unique filename to avoid collisions
                new_filename = f"{old_class}_{img_path.name}"
                target_path = target_class_dir / new_filename
                
                shutil.copy2(img_path, target_path)
                self.stats[split][new_class] += 1
            
            print(f"   {old_class:20s} → {new_class:20s}: {len(images):5d} images")
        
    def print_statistics(self):
        """Print reorganization statistics"""
        print("\n" + "="*80)
        print("REORGANIZATION SUMMARY")
        print("="*80)
        
        for split in ['train', 'val', 'test']:
            if split not in self.stats:
                continue
            
            print(f"\n{split.upper()}:")
            print("-"*80)
            print(f"{'Class':<30} | {'Images':>10}")
            print("-"*80)
            
            total = 0
            for class_name in sorted(self.stats[split].keys()):
                count = self.stats[split][class_name]
                total += count
                print(f"{class_name:<30} | {count:>10,}")
            
            print("-"*80)
            print(f"{'TOTAL':<30} | {total:>10,}")
        
        # Class comparison
        print("\n" + "="*80)
        print("CLASS COMPARISON: Before vs After")
        print("="*80)
        print(f"\n{'Original Classes':<30} | {'New Class':<30}")
        print("-"*80)
        
        for old_class, new_class in sorted(self.class_mapping.items()):
            marker = "→" if old_class != new_class else "  "
            print(f"{old_class:<30} {marker} {new_class:<30}")
        
        print("-"*80)
        print(f"\nTotal Classes: 12 → {len(set(self.class_mapping.values()))}")
        
    def save_mapping(self):
        """Save class mapping for reference"""
        mapping_info = {
            'timestamp': datetime.now().isoformat(),
            'class_mapping': self.class_mapping,
            'new_classes': sorted(set(self.class_mapping.values())),
            'statistics': dict(self.stats),
        }
        
        output_file = self.target_data / 'class_mapping.json'
        with open(output_file, 'w') as f:
            json.dump(mapping_info, f, indent=2)
        
        print(f"\n📄 Class mapping saved to: {output_file}")
        
    def run(self):
        """Run complete reorganization"""
        print("="*80)
        print("DATASET REORGANIZATION - v2.0")
        print("="*80)
        print(f"\nSource: {self.source_data}")
        print(f"Target: {self.target_data}")
        
        # Create structure
        self.create_directory_structure()
        
        # Reorganize each split
        for split in ['train', 'val', 'test']:
            self.reorganize_split(split)
        
        # Print statistics
        self.print_statistics()
        
        # Save mapping
        self.save_mapping()
        
        print("\n" + "="*80)
        print("✅ REORGANIZATION COMPLETE!")
        print("="*80)
        print("\n📊 Next Steps:")
        print("   1. Review statistics above")
        print("   2. Check data_v2/images/ directory")
        print("   3. Run: python train_yolov8_v2.py")
        print("="*80)


def main():
    """Main execution"""
    reorganizer = DatasetReorganizer(
        source_data='data/images',      # Original data
        target_data='data_v2/images'    # New reorganized data
    )
    
    reorganizer.run()


if __name__ == "__main__":
    main()
