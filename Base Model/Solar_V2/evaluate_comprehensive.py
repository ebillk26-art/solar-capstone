"""
Solar Panel Fault Detection v2.0 - Comprehensive Evaluation
============================================================

Complete evaluation with:
- Per-class metrics
- Confusion matrix
- Category analysis (Physical vs Electrical)
- Comparison with v1.0
"""

from ensemble_predict import EnsemblePredictor
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime


class ComprehensiveEvaluator:
    """Complete evaluation suite"""
    
    def __init__(self):
        self.predictor = EnsemblePredictor()
        self.class_names = self.predictor.class_names
        
        # Category mapping
        self.category_map = {
            'Cell-Fault': 'Physical',
            'Cracking': 'Physical',
            'Soiling': 'Physical',
            'Shadowing': 'Physical',
            'Vegetation': 'Physical',
            'Hot-Spot': 'Electrical',
            'Diode-Fault': 'Electrical',
            'Offline-Module': 'Electrical',
            'No-Anomaly': 'Normal',
        }
    
    def evaluate_all(self):
        """Run complete evaluation"""
        
        print("="*80)
        print("COMPREHENSIVE EVALUATION - v2.0")
        print("="*80)
        
        # Get predictions
        accuracy, y_true, y_pred = self.predictor.evaluate()
        
        # Overall metrics
        self.print_overall_metrics(accuracy, y_true, y_pred)
        
        # Per-class metrics
        self.print_per_class_metrics(y_true, y_pred)
        
        # Category analysis
        self.analyze_categories(y_true, y_pred)
        
        # Generate confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        # Compare with v1.0
        self.compare_with_v1()
        
        # Save results
        self.save_results(accuracy, y_true, y_pred)
    
    def print_overall_metrics(self, accuracy, y_true, y_pred):
        """Print overall metrics"""
        print(f"\n📊 OVERALL METRICS")
        print("-"*80)
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Total Samples: {len(y_true):,}")
    
    def print_per_class_metrics(self, y_true, y_pred):
        """Print per-class metrics"""
        print(f"\n📋 PER-CLASS METRICS")
        print("-"*80)
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4,
            output_dict=True
        )
        
        print(f"{'Class':<20} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Support':>8}")
        print("-"*80)
        
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name:<20} | {metrics['precision']:>10.4f} | "
                      f"{metrics['recall']:>10.4f} | {metrics['f1-score']:>10.4f} | "
                      f"{int(metrics['support']):>8}")
        
        print("-"*80)
        print(f"{'Macro Avg':<20} | {report['macro avg']['precision']:>10.4f} | "
              f"{report['macro avg']['recall']:>10.4f} | {report['macro avg']['f1-score']:>10.4f} | "
              f"{int(report['macro avg']['support']):>8}")
        print(f"{'Weighted Avg':<20} | {report['weighted avg']['precision']:>10.4f} | "
              f"{report['weighted avg']['recall']:>10.4f} | {report['weighted avg']['f1-score']:>10.4f} | "
              f"{int(report['weighted avg']['support']):>8}")
    
    def analyze_categories(self, y_true, y_pred):
        """Analyze by fault category"""
        print(f"\n🔍 CATEGORY ANALYSIS (Physical vs Electrical vs Normal)")
        print("-"*80)
        
        category_correct = {'Physical': 0, 'Electrical': 0, 'Normal': 0}
        category_total = {'Physical': 0, 'Electrical': 0, 'Normal': 0}
        
        for true_idx, pred_idx in zip(y_true, y_pred):
            true_class = self.class_names[true_idx]
            pred_class = self.class_names[pred_idx]
            
            true_category = self.category_map[true_class]
            pred_category = self.category_map[pred_class]
            
            category_total[true_category] += 1
            if true_category == pred_category:
                category_correct[true_category] += 1
        
        print(f"{'Category':<20} | {'Accuracy':>12} | {'Correct':>10} | {'Total':>10}")
        print("-"*80)
        
        for category in ['Physical', 'Electrical', 'Normal']:
            if category_total[category] > 0:
                acc = category_correct[category] / category_total[category]
                print(f"{category:<20} | {acc:>11.2%} | {category_correct[category]:>10} | "
                      f"{category_total[category]:>10}")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Generate confusion matrix visualization"""
        print(f"\n📈 Generating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix - Raw Counts (v2.0)', fontsize=14)
        ax1.set_xlabel('Predicted', fontsize=12)
        ax1.set_ylabel('True', fontsize=12)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title('Confusion Matrix - Normalized (v2.0)', fontsize=14)
        ax2.set_xlabel('Predicted', fontsize=12)
        ax2.set_ylabel('True', fontsize=12)
        
        plt.tight_layout()
        
        output_path = Path('results_v2')
        output_path.mkdir(exist_ok=True)
        plt.savefig(output_path / 'confusion_matrix_v2.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved to: results_v2/confusion_matrix_v2.png")
        plt.close()
    
    def compare_with_v1(self):
        """Compare with v1.0 results"""
        print(f"\n📊 COMPARISON: v1.0 vs v2.0")
        print("-"*80)
        
        # v1.0 results (from your original training)
        v1_accuracy = 0.774  # 77.4%
        v1_classes = 12
        
        # v2.0 results (current)
        v2_accuracy = self.predictor.evaluate()[0]
        v2_classes = len(self.class_names)
        
        print(f"{'Metric':<30} | {'v1.0':<15} | {'v2.0':<15} | {'Change':<15}")
        print("-"*80)
        print(f"{'Classes':<30} | {v1_classes:<15} | {v2_classes:<15} | {v2_classes-v1_classes:<15}")
        print(f"{'Accuracy':<30} | {v1_accuracy:<15.2%} | {v2_accuracy:<15.2%} | "
              f"{(v2_accuracy-v1_accuracy)*100:+.1f}%")
        print(f"{'Model':<30} | {'YOLOv8-medium':<15} | {'Ensemble':<15} | {'Upgraded':<15}")
        print(f"{'Image Size':<30} | {'640px':<15} | {'640px':<15} | {'Same':<15}")
        
        print("\n💡 Improvements in v2.0:")
        print("   ✅ Merged similar classes (12 → 9)")
        print("   ✅ Larger model (medium → large)")
        print("   ✅ Added EfficientNet")
        print("   ✅ Ensemble prediction")
        print(f"   ✅ Accuracy improved: {(v2_accuracy-v1_accuracy)*100:+.1f}%")
    
    def save_results(self, accuracy, y_true, y_pred):
        """Save evaluation results"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            'overall_accuracy': float(accuracy),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'total_samples': len(y_true),
            'improvements': {
                'v1_accuracy': 0.774,
                'v2_accuracy': float(accuracy),
                'improvement': float(accuracy - 0.774),
            }
        }
        
        output_path = Path('results_v2')
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / 'evaluation_results_v2.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Results saved to: results_v2/evaluation_results_v2.json")


def main():
    """Main execution"""
    
    evaluator = ComprehensiveEvaluator()
    evaluator.evaluate_all()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("   - results_v2/confusion_matrix_v2.png")
    print("   - results_v2/evaluation_results_v2.json")
    print("\n📊 Next Steps:")
    print("   1. Review confusion matrix")
    print("   2. Run: python export_for_deployment.py")
    print("   3. Deploy to Raspberry Pi!")
    print("="*80)


if __name__ == "__main__":
    main()
