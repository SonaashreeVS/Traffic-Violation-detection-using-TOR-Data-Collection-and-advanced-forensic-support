# inspect_all_datasets.py - ENHANCED VERSION with YOLO analysis
import os
import yaml
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def inspect_datasets():
    """Comprehensive inspection of both helmet (YOLO) and signal datasets"""
    base_path = r"D:\Traffic Violation"
    
    print("🔍 COMPREHENSIVE DATASET INSPECTION")
    print("=" * 60)
    
    # 1. Check Helmet Dataset (YOLO format)
    print("\n🛵 HELMET DATASET (YOLO Format):")
    print("-" * 40)
    
    helmet_stats = inspect_yolo_dataset(base_path, "helmet")
    print_helmet_stats(helmet_stats)
    
    # 2. Check Signal Dataset
    print("\n🚦 SIGNAL COLOR DATASET:")
    print("-" * 40)
    
    signal_stats = inspect_signal_dataset(base_path)
    print_signal_stats(signal_stats)
    
    # 3. Dataset Summary for Training
    print("\n📊 TRAINING READINESS SUMMARY:")
    print("-" * 40)
    print_training_summary(helmet_stats, signal_stats)
    
    return helmet_stats, signal_stats

def inspect_yolo_dataset(base_path, dataset_name):
    """Inspect YOLO format dataset with detailed analysis"""
    stats = {
        'train_images': 0,
        'train_labels': 0,
        'val_images': 0, 
        'val_labels': 0,
        'classes': [],
        'class_distribution': {},
        'image_sizes': [],
        'label_files_checked': 0,
        'valid_labels': 0
    }
    
    # Paths
    train_images = os.path.join(base_path, "train", "images")
    train_labels = os.path.join(base_path, "train", "labels") 
    val_images = os.path.join(base_path, "val", "images")
    val_labels = os.path.join(base_path, "val", "labels")
    
    # Count files
    stats['train_images'] = len(os.listdir(train_images)) if os.path.exists(train_images) else 0
    stats['train_labels'] = len(os.listdir(train_labels)) if os.path.exists(train_labels) else 0
    stats['val_images'] = len(os.listdir(val_images)) if os.path.exists(val_images) else 0
    stats['val_labels'] = len(os.listdir(val_labels)) if os.path.exists(val_labels) else 0
    
    # Get classes
    classes_file = os.path.join(base_path, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            stats['classes'] = [line.strip() for line in f.readlines()]
    
    # Analyze label distribution
    if os.path.exists(train_labels):
        label_files = os.listdir(train_labels)[:10]  # Sample first 10
        stats['label_files_checked'] = len(label_files)
        
        class_counts = Counter()
        for label_file in label_files:
            label_path = os.path.join(train_labels, label_file)
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if lines:  # Only count if file has content
                        stats['valid_labels'] += 1
                    for line in lines:
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
            except:
                continue
        
        # Convert to class names if available
        if stats['classes']:
            stats['class_distribution'] = {stats['classes'].get(k, f'class_{k}'): v 
                                         for k, v in class_counts.items()}
        else:
            stats['class_distribution'] = dict(class_counts)
    
    # Check image sizes (sample)
    if os.path.exists(train_images):
        image_files = os.listdir(train_images)[:5]  # Sample first 5
        for img_file in image_files:
            img_path = os.path.join(train_images, img_file)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    stats['image_sizes'].append(img.shape)
            except:
                pass
    
    return stats

def inspect_signal_dataset(base_path):
    """Inspect signal color classification dataset"""
    stats = {
        'train_images': 0,
        'test_images': 0,
        'color_distribution': {},
        'image_sizes': [],
        'sample_files': []
    }
    
    signal_train = os.path.join(base_path, "train_dataset", "train_images")
    signal_test = os.path.join(base_path, "test_dataset", "test_images")
    
    # Count files
    stats['train_images'] = len(os.listdir(signal_train)) if os.path.exists(signal_train) else 0
    stats['test_images'] = len(os.listdir(signal_test)) if os.path.exists(signal_test) else 0
    
    # Analyze color distribution from filenames
    if os.path.exists(signal_train):
        all_files = os.listdir(signal_train)
        stats['sample_files'] = all_files[:5]
        
        # Try to infer colors from filenames
        color_keywords = {
            'red': ['red', 'r_', '_r'],
            'green': ['green', 'g_', '_g'], 
            'yellow': ['yellow', 'y_', '_y']
        }
        
        color_counts = Counter()
        for file in all_files:
            file_lower = file.lower()
            for color, keywords in color_keywords.items():
                if any(keyword in file_lower for keyword in keywords):
                    color_counts[color] += 1
                    break
            else:
                color_counts['unknown'] += 1
        
        stats['color_distribution'] = dict(color_counts)
    
    # Check image sizes
    if os.path.exists(signal_train):
        sample_files = os.listdir(signal_train)[:3]
        for img_file in sample_files:
            img_path = os.path.join(signal_train, img_file)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    stats['image_sizes'].append(img.shape)
            except:
                pass
    
    return stats

def print_helmet_stats(stats):
    """Print detailed helmet dataset statistics"""
    print(f"📁 Train Images: {stats['train_images']}")
    print(f"📁 Train Labels: {stats['train_labels']}")
    print(f"📁 Val Images: {stats['val_images']}")
    print(f"📁 Val Labels: {stats['val_labels']}")
    
    if stats['classes']:
        print(f"🎯 Classes: {stats['classes']}")
    
    if stats['class_distribution']:
        print(f"📊 Class Distribution: {stats['class_distribution']}")
    
    print(f"🔍 Labels Checked: {stats['label_files_checked']}")
    print(f"✅ Valid Labels: {stats['valid_labels']}")
    
    if stats['image_sizes']:
        unique_sizes = set(stats['image_sizes'])
        print(f"🖼️ Sample Image Sizes: {list(unique_sizes)[:3]}")
    
    # Data quality assessment
    if stats['train_images'] > 0 and stats['train_labels'] > 0:
        imbalance = abs(stats['train_images'] - stats['train_labels'])
        if imbalance > 10:
            print(f"⚠️ Warning: Image-Label mismatch: {imbalance} files")
        else:
            print("✅ Good: Image-Label pairs match well")

def print_signal_stats(stats):
    """Print detailed signal dataset statistics"""
    print(f"📁 Train Images: {stats['train_images']}")
    print(f"📁 Test Images: {stats['test_images']}")
    
    if stats['color_distribution']:
        print(f"🎨 Color Distribution: {stats['color_distribution']}")
        
        # Check class balance
        if len(stats['color_distribution']) >= 2:
            counts = list(stats['color_distribution'].values())
            max_count = max(counts)
            min_count = min(counts)
            if min_count > 0:
                imbalance_ratio = max_count / min_count
                if imbalance_ratio > 3:
                    print(f"⚠️ Class imbalance detected: ratio {imbalance_ratio:.1f}")
    
    if stats['sample_files']:
        print(f"📄 Sample Files: {stats['sample_files']}")
    
    if stats['image_sizes']:
        unique_sizes = set(stats['image_sizes'])
        print(f"🖼️ Sample Image Sizes: {list(unique_sizes)[:3]}")

def print_training_summary(helmet_stats, signal_stats):
    """Print overall training readiness assessment"""
    print("🎯 HELMET DETECTION TRAINING:")
    helmet_ready = True
    
    if helmet_stats['train_images'] < 100:
        print(f"❌ Insufficient training images: {helmet_stats['train_images']}")
        helmet_ready = False
    else:
        print(f"✅ Sufficient training images: {helmet_stats['train_images']}")
    
    if helmet_stats['train_labels'] < 100:
        print(f"❌ Insufficient training labels: {helmet_stats['train_labels']}")
        helmet_ready = False
    else:
        print(f"✅ Sufficient training labels: {helmet_stats['train_labels']}")
    
    if not helmet_stats['classes']:
        print("❌ No classes defined")
        helmet_ready = False
    else:
        print(f"✅ Classes defined: {helmet_stats['classes']}")
    
    print("\n🎯 SIGNAL COLOR TRAINING:")
    signal_ready = True
    
    if signal_stats['train_images'] < 100:
        print(f"❌ Insufficient training images: {signal_stats['train_images']}")
        signal_ready = False
    else:
        print(f"✅ Sufficient training images: {signal_stats['train_images']}")
    
    if len(signal_stats['color_distribution']) < 2:
        print(f"❌ Insufficient color variety: {signal_stats['color_distribution']}")
        signal_ready = False
    else:
        print(f"✅ Good color variety: {list(signal_stats['color_distribution'].keys())}")
    
    print(f"\n📈 OVERALL ASSESSMENT:")
    if helmet_ready and signal_ready:
        print("🎉 EXCELLENT! Both datasets are ready for training!")
        print("Next steps: Run your training scripts for both models")
    elif helmet_ready:
        print("✅ Helmet dataset ready, ❌ Signal dataset needs work")
    elif signal_ready:
        print("✅ Signal dataset ready, ❌ Helmet dataset needs work") 
    else:
        print("❌ Both datasets need more data preparation")

def generate_dataset_report():
    """Generate a detailed report for model training"""
    helmet_stats, signal_stats = inspect_datasets()
    
    report = {
        'helmet_dataset': helmet_stats,
        'signal_dataset': signal_stats,
        'recommendations': []
    }
    
    # Generate recommendations
    if helmet_stats['train_images'] < 500:
        report['recommendations'].append("Collect more helmet training images (target: 500+)")
    
    if signal_stats['train_images'] < 300:
        report['recommendations'].append("Collect more signal training images (target: 300+)")
    
    if not helmet_stats['class_distribution']:
        report['recommendations'].append("Check helmet label files for proper formatting")
    
    return report

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Dataset Analysis...\n")
    
    # Run inspection
    helmet_stats, signal_stats = inspect_datasets()
    
    # Generate report
    report = generate_dataset_report()
    
    print("\n" + "=" * 60)
    print("📋 DATASET REPORT SUMMARY")
    print("=" * 60)
    
    for rec in report['recommendations']:
        print(f"• {rec}")
    
    print(f"\n💡 Next Steps:")
    print("1. Review the recommendations above")
    print("2. Run your training scripts:")
    print("   - For helmet detection: Use YOLO training")
    print("   - For signal colors: Use TensorFlow/Keras training") 
    print("3. Test trained models with your AI processor")
    
    print(f"\n🎯 Your system is configured for:")
    print(f"   • YOLO vehicle detection (pretrained)")
    print(f"   • Custom helmet detection (needs training)") 
    print(f"   • Custom signal color detection (needs training)")
    print(f"   • License plate recognition (Haar cascade + OCR)")