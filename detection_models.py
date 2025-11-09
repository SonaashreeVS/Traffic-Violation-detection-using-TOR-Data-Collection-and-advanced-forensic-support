# detection_models.py - YOLO-INTEGRATED VERSION
import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2
import numpy as np
import os
from ultralytics import YOLO

class HelmetDetectionModel:
    """Helmet detection model for motorcycles - YOLO Integrated"""
    def __init__(self, model_path=None):
        self.model = None
        self.input_size = (224, 224)
        self.class_names = ['no_helmet', 'helmet']
        
        # YOLO for motorcycle detection
        self.yolo_model = YOLO('yolov8n.pt')
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.create_model()
    
    def create_model(self):
        """Create helmet detection model with improved architecture"""
        # Use EfficientNet for better feature extraction
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        self.model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(2, activation='softmax', name='helmet_classifier')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Helmet detection model created!")
    
    def load_model(self, model_path):
        """Load pre-trained helmet model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Helmet detection model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading helmet model: {e}")
            self.create_model()
    
    def preprocess_image(self, image):
        """Preprocess image for helmet detection"""
        # Resize image
        image = cv2.resize(image, self.input_size)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        image = image.astype('float32') / 255.0
        
        return np.expand_dims(image, axis=0)
    
    def detect_helmet(self, image, confidence_threshold=0.7):
        """Detect helmet in image with confidence score"""
        if self.model is None:
            return 'unknown', 0.0, None
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get class name
            class_name = self.class_names[predicted_class_idx]
            
            # Only return confident predictions
            if confidence >= confidence_threshold:
                return class_name, confidence, predictions[0]
            else:
                return 'unknown', confidence, predictions[0]
                
        except Exception as e:
            print(f"❌ Helmet detection error: {e}")
            return 'unknown', 0.0, None
    
    def detect_motorcycles_with_helmets(self, frame, confidence_threshold=0.5):
        """Detect motorcycles and check helmets using YOLO + CNN"""
        try:
            # Detect motorcycles using YOLO
            results = self.yolo_model(frame, verbose=False)
            motorcycles = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = box.conf[0].item()
                        
                        # Filter for motorcycles (class 3 in COCO)
                        if class_id == 3 and confidence > confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Extract motorcycle ROI
                            motorcycle_roi = frame[int(y1):int(y2), int(x1):int(x2)]
                            
                            if motorcycle_roi.size > 0:
                                # Extract rider region (upper 40% for helmet)
                                rider_height = int((y2 - y1) * 0.4)
                                rider_roi = frame[int(y1):int(y1) + rider_height, int(x1):int(x2)]
                                
                                if rider_roi.size > 0:
                                    # Detect helmet
                                    helmet_class, helmet_confidence, _ = self.detect_helmet(rider_roi)
                                    
                                    motorcycles.append({
                                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                        'rider_bbox': [int(x1), int(y1), int(x2), int(y1) + rider_height],
                                        'class_name': 'motorcycle',
                                        'confidence': confidence,
                                        'helmet_status': helmet_class,
                                        'helmet_confidence': helmet_confidence,
                                        'has_helmet': helmet_class == 'helmet',
                                        'violation': helmet_class == 'no_helmet'
                                    })
            
            return motorcycles
            
        except Exception as e:
            print(f"❌ Motorcycle detection error: {e}")
            return []

class SignalColorDetectionModel:
    """Traffic signal color classification model - YOLO Enhanced"""
    def __init__(self, model_path=None):
        self.model = None
        self.input_size = (224, 224)
        self.class_names = ['red', 'yellow', 'green']
        self.color_rgb = {
            'red': (255, 0, 0),
            'yellow': (255, 255, 0),
            'green': (0, 255, 0)
        }
        
        # YOLO for traffic light detection
        self.yolo_model = YOLO('yolov8n.pt')
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.create_model()
    
    def create_model(self):
        """Create signal color classification model with improved architecture"""
        self.model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.1),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Fifth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.5),
            
            # Classification Head
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(3, activation='softmax', name='signal_classifier')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Signal color detection model created!")
    
    def load_model(self, model_path):
        """Load pre-trained signal color model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Signal color detection model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading signal model: {e}")
            self.create_model()
    
    def preprocess_image(self, image):
        """Preprocess image for signal color detection"""
        # Resize image
        image = cv2.resize(image, self.input_size)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype('float32') / 255.0
        
        return np.expand_dims(image, axis=0)
    
    def detect_signal_color(self, image, confidence_threshold=0.6):
        """Detect traffic signal color with confidence score"""
        if self.model is None:
            return 'unknown', 0.0, None
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get class name
            class_name = self.class_names[predicted_class_idx]
            
            # Only return confident predictions
            if confidence >= confidence_threshold:
                return class_name, confidence, predictions[0]
            else:
                return 'unknown', confidence, predictions[0]
                
        except Exception as e:
            print(f"❌ Signal color detection error: {e}")
            return 'unknown', 0.0, None
    
    def detect_traffic_lights_with_colors(self, frame, confidence_threshold=0.5):
        """Detect traffic lights and their colors using YOLO + CNN"""
        try:
            # Use traditional method to find traffic light regions
            traffic_light_regions = self._find_traffic_light_regions(frame)
            
            traffic_lights = []
            
            for region in traffic_light_regions:
                x1, y1, x2, y2 = region
                traffic_light_roi = frame[y1:y2, x1:x2]
                
                if traffic_light_roi.size > 0:
                    # Detect color using CNN
                    color, confidence, _ = self.detect_signal_color(traffic_light_roi)
                    
                    traffic_lights.append({
                        'bbox': [x1, y1, x2, y2],
                        'color': color,
                        'confidence': confidence,
                        'rgb_color': self.get_color_rgb(color),
                        'status': self._get_signal_status(color)
                    })
            
            return traffic_lights
            
        except Exception as e:
            print(f"❌ Traffic light detection error: {e}")
            return []
    
    def _find_traffic_light_regions(self, frame):
        """Find traffic light regions using traditional CV + YOLO"""
        regions = []
        
        # Method 1: Use YOLO to detect traffic lights (if trained)
        try:
            results = self.yolo_model(frame, verbose=False)
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Note: YOLO COCO doesn't have traffic lights by default
                        # You would need a custom model for this
                        # For now, use traditional method
                        pass
        except:
            pass
        
        # Method 2: Traditional computer vision approach
        height, width = frame.shape[:2]
        
        # Common traffic light positions (you can customize these)
        potential_regions = [
            # Top-right corner (common position)
            [width-100, 50, width-20, 200],
            # Top-left corner
            [50, 50, 150, 200],
            # Center top
            [width//2 - 50, 50, width//2 + 50, 200]
        ]
        
        # Filter regions that are within frame bounds
        for region in potential_regions:
            x1, y1, x2, y2 = region
            if (0 <= x1 < width and 0 <= y1 < height and 
                0 <= x2 <= width and 0 <= y2 <= height):
                regions.append(region)
        
        return regions
    
    def _get_signal_status(self, color):
        """Get traffic signal status"""
        status_map = {
            'red': 'STOP',
            'yellow': 'CAUTION', 
            'green': 'GO',
            'unknown': 'UNKNOWN'
        }
        return status_map.get(color, 'UNKNOWN')
    
    def get_color_rgb(self, color_name):
        """Get RGB color for visualization"""
        return self.color_rgb.get(color_name, (128, 128, 128))

class TrafficDetectionSystem:
    """Complete traffic detection system with YOLO integration"""
    def __init__(self, helmet_model_path=None, signal_model_path=None):
        self.helmet_detector = HelmetDetectionModel(helmet_model_path)
        self.signal_detector = SignalColorDetectionModel(signal_model_path)
        
        # Main YOLO model for vehicle detection
        self.yolo_model = YOLO('yolov8n.pt')
        
        print("🚦 YOLO Traffic Detection System Initialized!")
    
    def process_complete_frame(self, frame):
        """Complete frame processing with all detections"""
        results = {
            'vehicles': [],
            'motorcycles': [],
            'traffic_lights': [],
            'violations': [],
            'traffic_metrics': {}
        }
        
        try:
            # 1. Detect all vehicles using YOLO
            vehicles = self._detect_all_vehicles(frame)
            results['vehicles'] = vehicles
            
            # 2. Detect motorcycles with helmets
            motorcycles = self.helmet_detector.detect_motorcycles_with_helmets(frame)
            results['motorcycles'] = motorcycles
            
            # 3. Detect traffic lights with colors
            traffic_lights = self.signal_detector.detect_traffic_lights_with_colors(frame)
            results['traffic_lights'] = traffic_lights
            
            # 4. Detect violations
            violations = self._detect_violations(vehicles, motorcycles, traffic_lights)
            results['violations'] = violations
            
            # 5. Calculate traffic metrics
            metrics = self._calculate_traffic_metrics(vehicles, frame.shape)
            results['traffic_metrics'] = metrics
            
            return results
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return results
    
    def _detect_all_vehicles(self, frame, confidence_threshold=0.5):
        """Detect all vehicles using YOLO"""
        try:
            results = self.yolo_model(frame, verbose=False)
            vehicles = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = box.conf[0].item()
                        
                        # Filter for vehicles (COCO classes: car=2, motorcycle=3, bus=5, truck=7)
                        if class_id in [2, 3, 5, 7] and confidence > confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            class_names = {
                                2: 'car', 
                                3: 'motorcycle', 
                                5: 'bus', 
                                7: 'truck'
                            }
                            
                            vehicles.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'class_name': class_names[class_id],
                                'class_id': class_id,
                                'confidence': confidence,
                                'center_x': int((x1 + x2) / 2),
                                'center_y': int((y1 + y2) / 2),
                                'area': (x2 - x1) * (y2 - y1)
                            })
            
            return vehicles
            
        except Exception as e:
            print(f"❌ Vehicle detection error: {e}")
            return []
    
    def _detect_violations(self, vehicles, motorcycles, traffic_lights):
        """Detect various traffic violations"""
        violations = []
        
        # Helmet violations
        for motorcycle in motorcycles:
            if motorcycle.get('violation', False):
                violations.append({
                    'type': 'NO_HELMET',
                    'vehicle_id': len(violations) + 1,
                    'confidence': motorcycle['helmet_confidence'],
                    'bbox': motorcycle['bbox'],
                    'timestamp': np.datetime64('now')
                })
        
        # Red light violations (simplified - check if vehicles are moving when light is red)
        red_light = next((tl for tl in traffic_lights if tl['color'] == 'red'), None)
        if red_light and red_light['confidence'] > 0.7:
            # Check if any vehicles are in intersection area (simplified)
            for vehicle in vehicles:
                if self._is_in_intersection(vehicle, red_light):
                    violations.append({
                        'type': 'RED_LIGHT_VIOLATION',
                        'vehicle_id': len(violations) + 1,
                        'vehicle_type': vehicle['class_name'],
                        'confidence': min(vehicle['confidence'], red_light['confidence']),
                        'bbox': vehicle['bbox'],
                        'timestamp': np.datetime64('now')
                    })
        
        return violations
    
    def _is_in_intersection(self, vehicle, traffic_light):
        """Check if vehicle is in intersection area (simplified)"""
        # This is a simplified check - in real system, you'd use line crossing detection
        vehicle_center_y = vehicle['center_y']
        traffic_light_y = traffic_light['bbox'][1]  # Top of traffic light
        
        # If vehicle is above traffic light position, it might be in intersection
        return vehicle_center_y < traffic_light_y + 100
    
    def _calculate_traffic_metrics(self, vehicles, frame_shape):
        """Calculate traffic metrics"""
        frame_area = frame_shape[0] * frame_shape[1]
        total_vehicle_area = sum(v['area'] for v in vehicles)
        
        tor = total_vehicle_area / frame_area if frame_area > 0 else 0
        
        return {
            'vehicle_count': len(vehicles),
            'tor': round(tor, 3),
            'congestion_level': 'HIGH' if tor > 0.3 else 'MEDIUM' if tor > 0.15 else 'LOW',
            'vehicle_breakdown': {
                'cars': len([v for v in vehicles if v['class_name'] == 'car']),
                'motorcycles': len([v for v in vehicles if v['class_name'] == 'motorcycle']),
                'trucks': len([v for v in vehicles if v['class_name'] == 'truck']),
                'buses': len([v for v in vehicles if v['class_name'] == 'bus'])
            }
        }

# Usage example
def main():
    # Initialize the detection system
    detector = TrafficDetectionSystem(
        helmet_model_path='models/helmet_detection_model.h5',
        signal_model_path='models/signal_color_model.h5'
    )
    
    # Test with a sample image
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Process frame
    results = detector.process_complete_frame(sample_image)
    
    print("🎯 YOLO Detection Results:")
    print(f"🚗 Vehicles detected: {len(results['vehicles'])}")
    print(f"🏍️ Motorcycles with helmets: {len(results['motorcycles'])}")
    print(f"🚦 Traffic lights: {len(results['traffic_lights'])}")
    print(f"🚨 Violations found: {len(results['violations'])}")
    
    # Print traffic metrics
    metrics = results['traffic_metrics']
    print(f"📊 Traffic metrics: {metrics['vehicle_count']} vehicles, TOR: {metrics['tor']}")
    
    # Print violations
    for violation in results['violations']:
        print(f"   - {violation['type']} (confidence: {violation['confidence']:.2f})")

if __name__ == "__main__":
    main()