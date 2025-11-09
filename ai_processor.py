# ai_processor.py - YOLO-BASED VERSION (SQLite ONLY)
import tensorflow as tf
import cv2
import numpy as np
import json
import time
import requests
from typing import List, Dict, Any
from datetime import datetime
import sqlite3
import os
import re
import pytesseract
from PIL import Image
from collections import deque
from ultralytics import YOLO

print(f"🚀 TensorFlow {tf.__version__} loaded successfully!")
print(f"🎯 YOLO Ultralytics loaded successfully!")

class YOLOVehicleDetector:
    """YOLO-based vehicle detection"""
    def __init__(self, model_size='n'):
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = self.model.names
        print("✅ YOLO Vehicle Detector initialized!")
    
    def detect_vehicles(self, frame, confidence_threshold=0.5):
        """Detect vehicles in frame using YOLO"""
        try:
            # Run YOLO inference
            results = self.model(frame, conf=confidence_threshold, verbose=False)
            
            vehicles = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = box.conf[0].item()
                        
                        # Filter only vehicles
                        if class_id in self.vehicle_classes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            vehicles.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'class_id': class_id,
                                'class_name': self.class_names[class_id],
                                'confidence': confidence,
                                'center_x': int((x1 + x2) / 2),
                                'center_y': int((y1 + y2) / 2),
                                'area': (x2 - x1) * (y2 - y1)
                            })
            
            return vehicles
            
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            return []
    
    def draw_detections(self, frame, vehicles):
        """Draw bounding boxes and labels on frame"""
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            class_name = vehicle['class_name']
            confidence = vehicle['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame

class SignalColorClassifier:
    """CNN model for traffic signal color classification"""
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = ['red', 'yellow', 'green']
        self.input_size = (224, 224)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.create_model()
    
    def create_model(self):
        """Create signal color classification model"""
        try:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print("✅ Signal color model created successfully!")
        except Exception as e:
            print(f"❌ Error creating signal model: {e}")
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Signal color model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading signal model: {e}")
            self.create_model()
    
    def preprocess_image(self, image):
        """Preprocess image for signal classification"""
        image = cv2.resize(image, self.input_size)
        image = image.astype('float32') / 255.0
        return np.expand_dims(image, axis=0)
    
    def predict_color(self, image):
        """Predict traffic signal color"""
        if self.model is None:
            return 'unknown', 0.0
        
        try:
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            return self.class_names[predicted_class], float(confidence)
        except Exception as e:
            print(f"❌ Signal prediction error: {e}")
            return 'unknown', 0.0

class HelmetDetector:
    """Helmet detection for motorcycles"""
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = ['no_helmet', 'helmet']
        self.input_size = (224, 224)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.create_model()
    
    def create_model(self):
        """Create helmet detection model using transfer learning"""
        try:
            base_model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            base_model.trainable = False
            
            self.model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print("✅ Helmet detection model created successfully!")
        except Exception as e:
            print(f"❌ Error creating helmet model: {e}")
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Helmet detection model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading helmet model: {e}")
            self.create_model()
    
    def preprocess_image(self, image):
        """Preprocess image for helmet detection"""
        image = cv2.resize(image, self.input_size)
        image = image.astype('float32') / 255.0
        return np.expand_dims(image, axis=0)
    
    def detect_helmet(self, image):
        """Detect helmet in motorcycle image"""
        if self.model is None:
            return 'unknown', 0.0
        
        try:
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]  
            return self.class_names[predicted_class], float(confidence)
        except Exception as e:
            print(f"❌ Helmet detection error: {e}")
            return 'unknown', 0.0

class LineDetector:
    def __init__(self, num_frames_avg=10):
        self.y_start_queue = deque(maxlen=num_frames_avg)
        self.y_end_queue = deque(maxlen=num_frames_avg)
    
    def detect_white_line(self, frame, color, 
                          slope1=0.03, intercept1=920, slope2=0.03, intercept2=770, slope3=-0.8, intercept3=2420):
        
        def get_color_code(color_name):
            color_codes = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'yellow': (0, 255, 255)
            }
            return color_codes.get(color_name.lower())

        frame_org = frame.copy()
        height, width, _ = frame.shape
        
        # Line equations for ROI
        def line1(x): return slope1 * x + intercept1
        def line2(x): return slope2 * x + intercept2
        def line3(x): return slope3 * x + intercept3

        # Create masks
        mask1 = frame.copy()
        for x in range(width):
            y_line = line1(x)
            mask1[int(y_line):, x] = 0

        mask2 = mask1.copy()
        for x in range(width):
            y_line = line2(x)
            mask2[:int(y_line), x] = 0

        mask3 = mask2.copy()
        for y in range(height):
            x_line = line3(y)
            mask3[y, :int(x_line)] = 0

        # Convert to grayscale and process
        gray = cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(blurred_gray)

        # Edge detection and line detection
        edges = cv2.Canny(gray, 30, 100)
        dilated_edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.erode(dilated_edges, None, iterations=1)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=160, maxLineGap=5)

        x_start, x_end = 0, width - 1

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + np.finfo(float).eps)
                intercept = y1 - slope * x1

                y_start = int(slope * x_start + intercept)
                y_end = int(slope * x_end + intercept)
                
                self.y_start_queue.append(y_start)
                self.y_end_queue.append(y_end)

        # Compute averages
        avg_y_start = int(sum(self.y_start_queue) / len(self.y_start_queue)) if self.y_start_queue else 0
        avg_y_end = int(sum(self.y_end_queue) / len(self.y_end_queue)) if self.y_end_queue else 0

        # Draw the line
        line_start_ratio = 0.32
        x_start_adj = x_start + int(line_start_ratio * (x_end - x_start))
        avg_y_start_adj = avg_y_start + int(line_start_ratio * (avg_y_end - avg_y_start))

        mask = np.zeros_like(frame)
        cv2.line(mask, (x_start_adj, avg_y_start_adj), (x_end, avg_y_end), (255, 255, 255), 4)

        # Apply color based on traffic light
        color_code = get_color_code(color)
        if color_code == (0, 255, 0):  # Green
            channel_indices = [1]
        elif color_code == (0, 0, 255):  # Red
            channel_indices = [2]
        elif color_code == (0, 255, 255):  # Yellow
            channel_indices = [1, 2]
        else:
            channel_indices = []

        for channel_index in channel_indices:
            frame[mask[:,:,channel_index] == 255, channel_index] = 255

        # Create mask for license plate detection
        slope_avg = (avg_y_end - avg_y_start) / (x_end - x_start + np.finfo(float).eps)
        intercept_avg = avg_y_start - slope_avg * x_start

        mask_line = np.copy(frame_org)
        for x in range(width):
            y_line = slope_avg * x + intercept_avg - 35
            mask_line[:int(y_line), x] = 0

        return frame, mask_line

class SQLiteViolationDB:
    """SQLite database for traffic violations"""
    def __init__(self, db_path='traffic_violations.db'):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # License plates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS license_plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL UNIQUE,
                total_violations INTEGER DEFAULT 0,
                red_light_violations INTEGER DEFAULT 0,
                helmet_violations INTEGER DEFAULT 0,
                speeding_violations INTEGER DEFAULT 0,
                other_violations INTEGER DEFAULT 0,
                first_violation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_violation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Traffic violations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT,
                violation_type TEXT,
                vehicle_type TEXT,
                detected_speed INTEGER DEFAULT 0,
                speed_limit INTEGER DEFAULT 0,
                confidence REAL,
                location_lat REAL,
                location_lng REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                evidence_data TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_plate ON traffic_violations(plate_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_type ON traffic_violations(violation_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON traffic_violations(timestamp)')
        
        conn.commit()
        conn.close()
        print("✅ SQLite database initialized successfully!")
    
    def add_violation(self, violation_data):
        """Add a violation to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            plate_number = violation_data.get('license_plate', 'UNKNOWN')
            violation_type = violation_data.get('type', 'UNKNOWN')
            vehicle_type = violation_data.get('vehicle_type', 'Unknown')
            confidence = violation_data.get('confidence', 0.0)
            
            # Update license_plates table
            cursor.execute('''
                INSERT OR REPLACE INTO license_plates 
                (plate_number, total_violations, red_light_violations, helmet_violations, speeding_violations, last_violation)
                VALUES (?, 
                    COALESCE((SELECT total_violations FROM license_plates WHERE plate_number = ?), 0) + 1,
                    COALESCE((SELECT red_light_violations FROM license_plates WHERE plate_number = ?), 0) + ?,
                    COALESCE((SELECT helmet_violations FROM license_plates WHERE plate_number = ?), 0) + ?,
                    COALESCE((SELECT speeding_violations FROM license_plates WHERE plate_number = ?), 0) + ?,
                    CURRENT_TIMESTAMP
                )
            ''', (
                plate_number, plate_number, plate_number,
                1 if violation_type == 'RED_LIGHT_VIOLATION' else 0,
                plate_number,
                1 if violation_type == 'NO_HELMET' else 0,
                plate_number,
                1 if violation_type == 'SPEEDING' else 0
            ))
            
            # Insert into traffic_violations table
            cursor.execute('''
                INSERT INTO traffic_violations 
                (plate_number, violation_type, vehicle_type, detected_speed, speed_limit, confidence, evidence_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                plate_number,
                violation_type,
                vehicle_type,
                violation_data.get('detected_speed', 0),
                violation_data.get('speed_limit', 0),
                confidence,
                json.dumps(violation_data)
            ))
            
            conn.commit()
            print(f"✅ Violation recorded in SQLite: {plate_number} - {violation_type}")
            
        except Exception as e:
            print(f"❌ Database error: {e}")
        finally:
            conn.close()
    
    def get_statistics(self):
        """Get violation statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Overall statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_violations,
                    COUNT(DISTINCT plate_number) as unique_offenders,
                    SUM(red_light_violations) as total_red_light,
                    SUM(helmet_violations) as total_helmet,
                    SUM(speeding_violations) as total_speeding
                FROM license_plates
            ''')
            stats = cursor.fetchone()
            
            # Top offenders
            cursor.execute('''
                SELECT plate_number, total_violations 
                FROM license_plates 
                ORDER BY total_violations DESC 
                LIMIT 10
            ''')
            top_offenders = cursor.fetchall()
            
            return {
                'total_violations': stats[0] if stats else 0,
                'unique_offenders': stats[1] if stats else 0,
                'violation_breakdown': {
                    'red_light': stats[2] if stats else 0,
                    'no_helmet': stats[3] if stats else 0,
                    'speeding': stats[4] if stats else 0
                },
                'top_offenders': [{'plate': row[0], 'count': row[1]} for row in top_offenders]
            }
            
        except Exception as e:
            print(f"❌ Statistics error: {e}")
            return {}
        finally:
            conn.close()

class EnhancedRedLightViolationDetector:
    def __init__(self, db_path='traffic_violations.db'):
        self.db_path = db_path
        
        # Initialize detectors
        self.line_detector = LineDetector()
        self.signal_classifier = SignalColorClassifier('models/signal_color_model.h5')
        self.helmet_detector = HelmetDetector('models/helmet_detection_model.h5')
        
        # Load license plate cascade
        self.license_plate_cascade = self._load_license_plate_cascade()
        
        # Traffic light position
        self.traffic_light_rect = (1700, 40, 100, 250)
        
        # Initialize SQLite database
        self.db = SQLiteViolationDB(db_path)
        print("✅ Enhanced Red Light Violation Detector with SQLite initialized!")
    
    def _load_license_plate_cascade(self):
        """Load Haar cascade for license plate detection"""
        try:
            cascade_path = 'haarcascade_russian_plate_number.xml'
            if not os.path.exists(cascade_path):
                print("📥 Downloading license plate cascade...")
                url = "https://raw.githubusercontent.com/FarzadNekouee/Traffic-Violation-Detection/master/haarcascade_russian_plate_number.xml"
                response = requests.get(url)
                with open(cascade_path, 'wb') as file:
                    file.write(response.content)
            
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                print("❌ Failed to load license plate cascade")
                return None
                
            print("✅ License plate cascade loaded successfully")
            return cascade
            
        except Exception as e:
            print(f"⚠️  Could not load license plate cascade: {e}")
            return None

    def detect_traffic_light_color_enhanced(self, frame):
        """Enhanced traffic light detection using both methods"""
        # Method 1: Traditional HSV-based detection
        hsv_color, hsv_confidence = self._detect_traffic_light_hsv(frame)
        
        # Method 2: CNN-based classification
        if self.traffic_light_rect:
            x, y, w, h = self.traffic_light_rect
            traffic_light_roi = frame[y:y+h, x:x+w]
            if traffic_light_roi.size > 0:
                cnn_color, cnn_confidence = self.signal_classifier.predict_color(traffic_light_roi)
            else:
                cnn_color, cnn_confidence = 'unknown', 0.0
        else:
            cnn_color, cnn_confidence = 'unknown', 0.0
        
        # Combine results (prefer CNN if confidence is high)
        if cnn_confidence > 0.8:
            final_color = cnn_color
            final_confidence = cnn_confidence
            method = 'cnn'
        else:
            final_color = hsv_color
            final_confidence = hsv_confidence
            method = 'hsv'
        
        # Add annotation to frame
        self._annotate_traffic_light(frame, final_color, final_confidence, method)
        
        return frame, final_color, final_confidence

    def _detect_traffic_light_hsv(self, image):
        """Traditional HSV-based traffic light detection"""
        if self.traffic_light_rect is None:
            return 'unknown', 0.0
            
        x, y, w, h = self.traffic_light_rect
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 'unknown', 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Color ranges
        red_lower = np.array([0, 120, 70])
        red_upper = np.array([10, 255, 255])
        
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])

        green_lower = np.array([40, 50, 50])
        green_upper = np.array([90, 255, 255])

        # Create masks
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Count pixels
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Determine color with confidence
        total_pixels = w * h
        if total_pixels == 0:
            return 'unknown', 0.0
            
        red_confidence = red_pixels / total_pixels
        yellow_confidence = yellow_pixels / total_pixels
        green_confidence = green_pixels / total_pixels
        
        max_confidence = max(red_confidence, yellow_confidence, green_confidence)
        
        if max_confidence < 0.1:  # Threshold
            return 'unknown', max_confidence
        elif red_confidence == max_confidence:
            return 'red', red_confidence
        elif yellow_confidence == max_confidence:
            return 'yellow', yellow_confidence
        else:
            return 'green', green_confidence

    def _annotate_traffic_light(self, image, color, confidence, method):
        """Add traffic light annotation to image"""
        message = f"Traffic Light: {color.upper()} ({method}) - {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if color == 'red':
            text_color = (0, 0, 255)
            status = "STOP"
        elif color == 'yellow':
            text_color = (0, 255, 255)
            status = "CAUTION"
        elif color == 'green':
            text_color = (0, 255, 0)
            status = "GO"
        else:
            text_color = (255, 255, 255)
            status = "UNKNOWN"
        
        cv2.putText(image, message, (15, 70), font, 0.7, text_color, 2, cv2.LINE_AA)
        cv2.putText(image, status, (15, 100), font, 0.7, text_color, 2, cv2.LINE_AA)

    def detect_helmet_violations(self, vehicles, frame):
        """Detect helmet violations for motorcycles"""
        helmet_violations = []
        
        for vehicle in vehicles:
            if vehicle['class_name'] == 'motorcycle':
                x1, y1, x2, y2 = vehicle['bbox']
                # Extract rider region (upper part of motorcycle bounding box)
                rider_height = int((y2 - y1) * 0.4)  # Top 40% for rider
                rider_roi = frame[y1:y1 + rider_height, x1:x2]
                
                if rider_roi.size > 0:
                    helmet_status, confidence = self.helmet_detector.detect_helmet(rider_roi)
                    
                    if helmet_status == 'no_helmet' and confidence > 0.7:
                        violation = {
                            'type': 'NO_HELMET',
                            'vehicle_id': vehicle['id'],
                            'confidence': confidence,
                            'timestamp': datetime.now().isoformat(),
                            'severity': 'HIGH'
                        }
                        helmet_violations.append(violation)
                        
                        # Update database
                        self._log_helmet_violation(vehicle, confidence)
        
        return helmet_violations

    def _log_helmet_violation(self, vehicle, confidence):
        """Log helmet violation to SQLite database"""
        try:
            violation_data = {
                'license_plate': vehicle.get('license_plate', 'UNKNOWN'),
                'type': 'NO_HELMET',
                'confidence': confidence,
                'vehicle_type': vehicle.get('class_name', 'Unknown')
            }
            self.db.add_violation(violation_data)
            
        except Exception as e:
            print(f"❌ Helmet violation logging error: {e}")

    def detect_red_light_violations(self, frame, vehicles):
        """Enhanced red light violation detection with real vehicles"""
        violations = []
        license_plates = []
        annotated_frame = frame.copy()
        
        try:
            # Detect traffic light color with enhanced method
            annotated_frame, color, confidence = self.detect_traffic_light_color_enhanced(annotated_frame)
            
            # Only process if light is red with sufficient confidence
            if color == 'red' and confidence > 0.6:
                # Detect crossing line
                annotated_frame, mask_line = self.line_detector.detect_white_line(annotated_frame, color)
                
                # Check for vehicles crossing the line
                for vehicle in vehicles:
                    if self._is_vehicle_crossing_line(vehicle, mask_line):
                        # Extract license plate for this vehicle
                        plate_text = self._extract_license_plate_from_vehicle(frame, vehicle)
                        
                        if plate_text and self.validate_license_plate(plate_text):
                            violation_data = {
                                'type': 'RED_LIGHT_VIOLATION',
                                'license_plate': plate_text,
                                'vehicle_type': vehicle['class_name'],
                                'confidence': confidence * vehicle['confidence'],
                                'timestamp': datetime.now().isoformat(),
                                'severity': 'HIGH'
                            }
                            violations.append(violation_data)
                            license_plates.append(plate_text)
                            
                            # Update database
                            self.update_database_with_violation(plate_text)
                            print(f"🚨 Red light violation detected: {plate_text} ({vehicle['class_name']})")
            
            return {
                'violations': violations,
                'license_plates': license_plates,
                'traffic_light_color': color,
                'traffic_light_confidence': confidence,
                'annotated_frame': annotated_frame
            }
            
        except Exception as e:
            print(f"❌ Enhanced red light detection error: {e}")
            return {
                'violations': [], 
                'license_plates': [], 
                'traffic_light_color': 'unknown', 
                'traffic_light_confidence': 0.0,
                'annotated_frame': annotated_frame
            }

    def _is_vehicle_crossing_line(self, vehicle, mask_line):
        """Check if vehicle is crossing the stop line"""
        x1, y1, x2, y2 = vehicle['bbox']
        center_x = vehicle['center_x']
        center_y = vehicle['center_y']
        
        # Check if vehicle center is below the line in the mask
        if 0 <= center_x < mask_line.shape[1] and 0 <= center_y < mask_line.shape[0]:
            # Check if the pixel at vehicle center is white (above line)
            pixel_value = mask_line[center_y, center_x]
            if np.any(pixel_value > 0):  # Vehicle is crossing the line
                return True
        
        return False

    def _extract_license_plate_from_vehicle(self, frame, vehicle):
        """Extract license plate from specific vehicle"""
        if self.license_plate_cascade is None:
            return None
            
        try:
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_roi = frame[y1:y2, x1:x2]
            
            if vehicle_roi.size == 0:
                return None
                
            gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Detect license plates in vehicle ROI
            license_plates = self.license_plate_cascade.detectMultiScale(
                gray, scaleFactor=1.07, minNeighbors=10, minSize=(20, 20)
            )
            
            for (x_plate, y_plate, w_plate, h_plate) in license_plates:
                license_plate_image = gray[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
                text = self.apply_ocr_to_image(license_plate_image)
                if text and self.validate_license_plate(text):
                    return text
        
        except Exception as e:
            print(f"❌ License plate extraction error: {e}")
        
        return None

    def validate_license_plate(self, text):
        patterns = [
            r"^[A-Z]{2}\s[0-9]{3,4}$",
            r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$",
            r"^[0-9]{2}[A-Z]{2}[0-9]{4}$",
        ]
        
        for pattern in patterns:
            if re.match(pattern, text.strip()):
                return True
        return False

    def apply_ocr_to_image(self, license_plate_image):    
        try:
            _, img = cv2.threshold(license_plate_image, 120, 255, cv2.THRESH_BINARY)
            height, width = img.shape
            if width > 0 and height > 0:
                img = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            
            pil_img = Image.fromarray(img)
            full_text = pytesseract.image_to_string(pil_img, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
            return full_text.strip()
            
        except Exception as e:
            print(f"❌ OCR error: {e}")
            return ""

    def update_database_with_violation(self, plate_number):
        try:
            violation_data = {
                'license_plate': plate_number,
                'type': 'RED_LIGHT_VIOLATION'
            }
            self.db.add_violation(violation_data)
            
        except Exception as e:
            print(f"❌ Database error: {e}")

class CompleteTrafficAI:
    def __init__(self, model_path: str = 'yolov8n.pt', google_api_key: str = None):
        self.model_path = model_path
        self.google_api_key = google_api_key
        
        # Initialize YOLO detector
        self.yolo_detector = YOLOVehicleDetector('n')
        
        # Enhanced red light detector with signal classification and helmet detection
        self.red_light_detector = EnhancedRedLightViolationDetector()
        
        # Tracking
        self.track_history = {}
        self.forensic_db = "traffic_forensics.db"
        
        self._init_forensic_database()
        print("✅ Complete Traffic AI System Initialized with YOLO!")
        
    def _init_forensic_database(self):
        """Initialize forensic evidence database"""
        try:
            conn = sqlite3.connect(self.forensic_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT,
                    vehicle_id INTEGER,
                    speed REAL,
                    speed_limit REAL,
                    location TEXT,
                    timestamp DATETIME,
                    confidence REAL,
                    evidence_data TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS traffic_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    description TEXT,
                    vehicles_involved TEXT,
                    timestamp DATETIME,
                    tor REAL,
                    location TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Forensic database initialized")
        except Exception as e:
            print(f"⚠️  Forensic database error: {e}")

    def process_image(self, image: np.ndarray, location: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Process single image for traffic violations
        """
        start_time = time.time()
        
        # 1. Vehicle detection using YOLO
        vehicles = self.yolo_detector.detect_vehicles(image)
        
        # 2. Update tracking and calculate speeds
        vehicles_with_tracking = self._update_tracking(vehicles)
        
        # 3. Enhanced red light violation detection
        red_light_results = self.red_light_detector.detect_red_light_violations(image.copy(), vehicles_with_tracking)
        
        # 4. Helmet violation detection
        helmet_violations = self.red_light_detector.detect_helmet_violations(vehicles_with_tracking, image.copy())
        
        # 5. Speed violations detection
        speed_violations = self._detect_speed_violations(vehicles_with_tracking)
        
        # 6. Combine all violations
        all_violations = speed_violations + red_light_results['violations'] + helmet_violations
        
        # 7. Calculate metrics
        tor = self._calculate_tor(vehicles_with_tracking, image.shape)
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'processing_time': processing_time,
            'vehicles_detected': len(vehicles_with_tracking),
            'violations': all_violations,
            'traffic_metrics': {
                'tor': tor,
                'vehicle_count': len(vehicles_with_tracking),
                'average_speed': np.mean([v.get('speed', 0) for v in vehicles_with_tracking]) if vehicles_with_tracking else 0,
                'congestion_level': self._get_congestion_level(tor),
                'traffic_light_status': red_light_results['traffic_light_color'],
                'traffic_light_confidence': red_light_results['traffic_light_confidence']
            },
            'red_light_detection': {
                'violations_detected': len(red_light_results['violations']),
                'license_plates': red_light_results['license_plates'],
                'traffic_light_color': red_light_results['traffic_light_color'],
                'confidence': red_light_results['traffic_light_confidence']
            },
            'helmet_detection': {
                'violations_detected': len(helmet_violations),
                'motorcycles_checked': len([v for v in vehicles_with_tracking if v['class_name'] == 'motorcycle'])
            },
            'speed_detection': {
                'violations_detected': len(speed_violations),
                'vehicles_checked': len(vehicles_with_tracking)
            },
            'vehicle_breakdown': {
                'cars': len([v for v in vehicles_with_tracking if v['class_name'] == 'car']),
                'motorcycles': len([v for v in vehicles_with_tracking if v['class_name'] == 'motorcycle']),
                'trucks': len([v for v in vehicles_with_tracking if v['class_name'] == 'truck']),
                'buses': len([v for v in vehicles_with_tracking if v['class_name'] == 'bus'])
            },
            'timestamp': datetime.now().isoformat(),
            'mode': 'real_yolo'
        }

    def _update_tracking(self, vehicles: List[Dict]) -> List[Dict]:
        """Update vehicle tracking and calculate speeds"""
        current_time = time.time()
        updated_vehicles = []
        
        for vehicle in vehicles:
            center = (vehicle['center_x'], vehicle['center_y'])
            vehicle_id = None
            
            # Find matching track
            for track_id, track_data in self.track_history.items():
                if track_data['positions']:
                    last_pos = track_data['positions'][-1]
                    distance = np.sqrt((center[0]-last_pos[0])**2 + (center[1]-last_pos[1])**2)
                    if distance < 50:  # Matching threshold
                        vehicle_id = track_id
                        
                        # Calculate speed
                        if len(track_data['positions']) > 1:
                            time_diff = current_time - track_data['last_time']
                            if time_diff > 0:
                                movement = np.sqrt((center[0]-last_pos[0])**2 + (center[1]-last_pos[1])**2)
                                vehicle['speed'] = int(movement / time_diff * 3.6)  # km/h
                        break
            
            # Create new track if no match
            if vehicle_id is None:
                vehicle_id = len(self.track_history) + 1
                self.track_history[vehicle_id] = {'positions': [], 'last_time': current_time}
                vehicle['speed'] = 0
            
            vehicle['id'] = vehicle_id
            self.track_history[vehicle_id]['positions'].append(center)
            self.track_history[vehicle_id]['last_time'] = current_time
            
            # Keep only last 10 positions
            if len(self.track_history[vehicle_id]['positions']) > 10:
                self.track_history[vehicle_id]['positions'].pop(0)
            
            updated_vehicles.append(vehicle)
        
        # Clean old tracks
        self._clean_old_tracks(current_time)
        return updated_vehicles

    def _clean_old_tracks(self, current_time: float):
        """Remove tracks that haven't been updated recently"""
        expired = [tid for tid, data in self.track_history.items() 
                  if current_time - data['last_time'] > 5.0]
        for tid in expired: 
            del self.track_history[tid]

    def _detect_speed_violations(self, vehicles: List[Dict]) -> List[Dict]:
        """Detect speeding violations"""
        violations = []
        speed_limit = 50  # Default speed limit
        
        for vehicle in vehicles:
            if vehicle.get('speed', 0) > speed_limit:
                violations.append({
                    'type': 'SPEEDING',
                    'vehicle_id': vehicle['id'],
                    'detected_speed': vehicle['speed'],
                    'speed_limit': speed_limit,
                    'confidence': vehicle['confidence'] * 0.9,
                    'severity': 'HIGH' if vehicle['speed'] > speed_limit + 20 else 'MEDIUM',
                    'timestamp': datetime.now().isoformat(),
                    'vehicle_type': vehicle['class_name']
                })
        
        return violations

    def _calculate_tor(self, vehicles: List[Dict], frame_shape: tuple) -> float:
        """Calculate Traffic Occupancy Rate"""
        frame_area = frame_shape[0] * frame_shape[1]
        if frame_area == 0:
            return 0.0
        
        total_vehicle_area = sum(v['area'] for v in vehicles)
        return min(total_vehicle_area / frame_area, 1.0)

    def _get_congestion_level(self, tor: float) -> str:
        """Get congestion level based on TOR"""
        if tor > 0.3: 
            return 'HIGH'
        elif tor > 0.15: 
            return 'MEDIUM'
        else: 
            return 'LOW'

    def analyze_road_image(self, image_path: str) -> Dict[str, Any]:
        """
        Main function to analyze a road image and detect violations
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not read image'}
            
            # Process image
            results = self.process_image(image)
            
            # Format violations for easy reading
            violation_summary = self._format_violation_summary(results)
            
            return {
                'success': True,
                'image_path': image_path,
                'violation_summary': violation_summary,
                'detailed_results': results,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _format_violation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format violations into a readable summary"""
        violations = results.get('violations', [])
        
        summary = {
            'total_violations': len(violations),
            'violations_by_type': {},
            'red_light_violators': [],
            'speeding_vehicles': [],
            'helmet_violators': []
        }
        
        for violation in violations:
            v_type = violation['type']
            
            # Count by type
            summary['violations_by_type'][v_type] = summary['violations_by_type'].get(v_type, 0) + 1
            
            # Categorize violations
            if v_type == 'RED_LIGHT_VIOLATION':
                summary['red_light_violators'].append({
                    'license_plate': violation.get('license_plate', 'UNKNOWN'),
                    'confidence': violation['confidence'],
                    'vehicle_type': violation.get('vehicle_type', 'unknown')
                })
            elif v_type == 'SPEEDING':
                summary['speeding_vehicles'].append({
                    'vehicle_id': violation['vehicle_id'],
                    'speed': violation['detected_speed'],
                    'limit': violation['speed_limit'],
                    'severity': violation['severity']
                })
            elif v_type == 'NO_HELMET':
                summary['helmet_violators'].append({
                    'vehicle_id': violation['vehicle_id'],
                    'confidence': violation['confidence']
                })
        
        return summary

    def get_detailed_violation_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed text report of violations"""
        violations = results.get('violations', [])
        
        if not violations:
            return "✅ No violations detected in this image."
        
        report = ["🚨 TRAFFIC VIOLATION REPORT", "=" * 40]
        
        # Red light violations
        red_light_violations = [v for v in violations if v['type'] == 'RED_LIGHT_VIOLATION']
        if red_light_violations:
            report.append("\n🔴 RED LIGHT VIOLATIONS:")
            for viol in red_light_violations:
                report.append(f"   • Plate: {viol['license_plate']} | Confidence: {viol['confidence']:.1%}")
        
        # Speeding violations
        speeding_violations = [v for v in violations if v['type'] == 'SPEEDING']
        if speeding_violations:
            report.append("\n🚀 SPEEDING VIOLATIONS:")
            for viol in speeding_violations:
                report.append(f"   • Vehicle {viol['vehicle_id']} | Speed: {viol['detected_speed']}km/h (Limit: {viol['speed_limit']}km/h)")
        
        # Helmet violations
        helmet_violations = [v for v in violations if v['type'] == 'NO_HELMET']
        if helmet_violations:
            report.append("\n👷 NO HELMET VIOLATIONS:")
            for viol in helmet_violations:
                report.append(f"   • Motorcycle {viol['vehicle_id']} | Confidence: {viol['confidence']:.1%}")
        
        # Traffic metrics
        metrics = results.get('traffic_metrics', {})
        report.append(f"\n📊 TRAFFIC METRICS:")
        report.append(f"   • Vehicles detected: {metrics.get('vehicle_count', 0)}")
        report.append(f"   • Traffic light: {metrics.get('traffic_light_status', 'unknown').upper()}")
        report.append(f"   • Congestion level: {metrics.get('congestion_level', 'unknown')}")
        
        return "\n".join(report)

    def visualize_detections(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Create visualization with detections and violations
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print("❌ Could not read image")
                return None
            
            # Process image to get results
            results = self.process_image(image)
            vehicles = results.get('vehicles_detected', 0)
            violations = results.get('violations', [])
            
            # Draw vehicle detections
            annotated_image = self.yolo_detector.draw_detections(image.copy(), 
                [v for v in results.get('detailed_results', {}).get('vehicles', [])])
            
            # Add violation annotations
            for violation in violations:
                if violation['type'] == 'RED_LIGHT_VIOLATION':
                    # Find vehicle with this license plate
                    for vehicle in results.get('detailed_results', {}).get('vehicles', []):
                        if vehicle.get('license_plate') == violation.get('license_plate'):
                            x1, y1, x2, y2 = vehicle['bbox']
                            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(annotated_image, "RED LIGHT VIOLATION", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            break
            
            # Add summary text
            summary_text = [
                f"Vehicles: {vehicles}",
                f"Violations: {len(violations)}",
                f"Traffic Light: {results['traffic_metrics']['traffic_light_status']}",
                f"Congestion: {results['traffic_metrics']['congestion_level']}"
            ]
            
            for i, text in enumerate(summary_text):
                cv2.putText(annotated_image, text, (10, 30 + i*30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save if output path provided
            if output_path:
                cv2.imwrite(output_path, annotated_image)
                print(f"✅ Visualization saved to: {output_path}")
            
            return annotated_image
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return None

    def get_forensic_report(self, start_time: str, end_time: str) -> Dict[str, Any]:
        """Generate forensic report for time period"""
        try:
            conn = sqlite3.connect(self.forensic_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM violations 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (start_time, end_time))
            
            violations = []
            for row in cursor.fetchall():
                violations.append({
                    'id': row[0],
                    'type': row[1],
                    'vehicle_id': row[2],
                    'speed': row[3],
                    'speed_limit': row[4],
                    'location': row[5],
                    'timestamp': row[6],
                    'confidence': row[7],
                    'evidence_data': row[8]
                })
            
            conn.close()
            
            return {
                'period': f"{start_time} to {end_time}",
                'violation_count': len(violations),
                'violations_by_type': self._count_violations_by_type(violations),
                'summary': {
                    'total_vehicles_processed': len(violations),
                    'peak_violation_time': self._find_peak_violation_time(violations)
                }
            }
        except Exception as e:
            return {'error': f'Forensic report error: {e}'}

    def _count_violations_by_type(self, violations: List[Dict]) -> Dict[str, int]:
        """Count violations by type"""
        counts = {}
        for violation in violations:
            v_type = violation['type']
            counts[v_type] = counts.get(v_type, 0) + 1
        return counts

    def _find_peak_violation_time(self, violations: List[Dict]) -> str:
        """Find time with most violations"""
        if not violations:
            return "No violations"
        
        hour_counts = {}
        for violation in violations:
            if isinstance(violation['timestamp'], str):
                hour = violation['timestamp'][11:13]
            else:
                hour = violation['timestamp'].strftime('%H')
            
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        if hour_counts:
            peak_hour = max(hour_counts, key=hour_counts.get)
            return f"{peak_hour}:00 - {int(peak_hour)+1}:00"
        return "No data"

# === USAGE EXAMPLES ===
def main():
    """Main function to demonstrate the system"""
    print("🚀 Starting YOLO-based Traffic Violation Detection System...")
    
    # Initialize the system
    ai_system = CompleteTrafficAI()
    
    # Example 1: Analyze a single image
    image_path = "test_road_image.jpg"
    
    if os.path.exists(image_path):
        print(f"\n📷 Analyzing image: {image_path}")
        results = ai_system.analyze_road_image(image_path)
        
        if results['success']:
            # Print detailed report
            report = ai_system.get_detailed_violation_report(results['detailed_results'])
            print(report)
            
            # Create visualization
            ai_system.visualize_detections(image_path, "output_with_violations.jpg")
        else:
            print(f"❌ Error: {results['error']}")
    else:
        print(f"⚠️  Test image not found: {image_path}")
        print("💡 Please provide a road image to analyze")

def test_with_sample_image():
    """Test with a sample image"""
    ai_system = CompleteTrafficAI()
    
    # Create a sample test image
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite("sample_test.jpg", sample_image)
    
    print("🧪 Testing with sample image...")
    results = ai_system.analyze_road_image("sample_test.jpg")
    
    if results['success']:
        print("✅ System test completed successfully!")
        print(f"🚗 Vehicles detected: {results['detailed_results']['vehicles_detected']}")
        print(f"🚨 Violations found: {len(results['detailed_results']['violations'])}")
    else:
        print(f"❌ Test failed: {results['error']}")

if __name__ == "__main__":
    # Install required packages first:
    # pip install ultralytics opencv-python tensorflow pytesseract pillow
    
    main()
    # test_with_sample_image()