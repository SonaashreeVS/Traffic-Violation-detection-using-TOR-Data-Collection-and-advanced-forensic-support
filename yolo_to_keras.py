# yolo_to_keras.py - Convert YOLO format to Keras training format
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class YOLOToKerasConverter:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_size = (224, 224)
        
    def convert_helmet_dataset(self):
        """Convert YOLO helmet dataset to Keras format"""
        print("🛵 Converting helmet dataset...")
        
        train_images_dir = os.path.join(self.dataset_path, "train", "images")
        train_labels_dir = os.path.join(self.dataset_path, "train", "labels")
        
        images = []
        labels = []
        
        if not os.path.exists(train_images_dir):
            print("❌ Train images directory not found")
            return np.array(images), np.array(labels)
        
        # Get class mapping
        class_file = os.path.join(self.dataset_path, "classes.txt")
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            print(f"📋 Classes: {classes}")
        else:
            classes = ['no_helmet', 'helmet']  # Default classes
            print("⚠️  Using default classes: ['no_helmet', 'helmet']")
        
        image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📁 Found {len(image_files)} images")
        
        for img_file in image_files[:100]:  # Process first 100 images for demo
            img_path = os.path.join(train_images_dir, img_file)
            label_path = os.path.join(train_labels_dir, os.path.splitext(img_file)[0] + '.txt')
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, self.img_size)
                img = img.astype('float32') / 255.0
                images.append(img)
                
                # Load label
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # For helmet detection, we use the first object's class
                            first_line = lines[0].strip().split()
                            if first_line:
                                class_id = int(first_line[0])
                                labels.append(class_id)
                                continue
                
                # If no label found, use default (no_helmet)
                labels.append(0)
        
        print(f"✅ Converted {len(images)} helmet samples")
        return np.array(images), np.array(labels)
    
    def convert_signal_dataset(self):
        """Convert signal color dataset to Keras format"""
        print("🚦 Converting signal color dataset...")
        
        signal_train_dir = os.path.join(self.dataset_path, "train_dataset", "train_images")
        
        images = []
        labels = []
        
        if not os.path.exists(signal_train_dir):
            print("❌ Signal train directory not found")
            return np.array(images), np.array(labels)
        
        color_mapping = {'red': 0, 'yellow': 1, 'green': 2}
        image_files = [f for f in os.listdir(signal_train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📁 Found {len(image_files)} signal images")
        
        for img_file in image_files[:100]:  # Process first 100 images
            img_path = os.path.join(signal_train_dir, img_file)
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, self.img_size)
                img = img.astype('float32') / 255.0
                images.append(img)
                
                # Infer color from filename
                img_lower = img_file.lower()
                if 'red' in img_lower:
                    labels.append(color_mapping['red'])
                elif 'yellow' in img_lower:
                    labels.append(color_mapping['yellow'])
                elif 'green' in img_lower:
                    labels.append(color_mapping['green'])
                else:
                    labels.append(0)  # Default to red
        
        print(f"✅ Converted {len(images)} signal samples")
        return np.array(images), np.array(labels)