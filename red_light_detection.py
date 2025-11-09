# red_light_detection.py - COMPLETE VIOLATION DETECTOR (SQLite ONLY VERSION)
import cv2
import numpy as np
import pytesseract
import re
import json
from PIL import Image
from collections import deque
import sqlite3
from datetime import datetime
from ultralytics import YOLO

class CompleteViolationDetector:
    def __init__(self, db_path='traffic_violations.db'):
        self.db_path = db_path
        
        # Initialize detectors
        self.line_detector = LineDetector()
        
        # Load license plate cascade
        self.license_plate_cascade = self._load_license_plate_cascade()
        
        # YOLO for vehicle detection
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Traffic light position
        self.traffic_light_rect = (1700, 40, 100, 250)
        
        # Initialize SQLite database
        self._init_sqlite_database()
        
        print("✅ Complete Violation Detector with SQLite initialized!")
    
    def _init_sqlite_database(self):
        """Initialize SQLite database for ALL violation types"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main license plates table (tracks all violations per plate)
            cursor.execute("""
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
            """)
            
            # Detailed violations table
            cursor.execute("""
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
            """)
            
            # Helmet violations detail table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS helmet_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT,
                    vehicle_type TEXT,
                    confidence REAL,
                    rider_detected BOOLEAN DEFAULT TRUE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_plate ON traffic_violations(plate_number)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON traffic_violations(violation_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON traffic_violations(timestamp)")
            
            conn.commit()
            conn.close()
            print("✅ SQLite database initialized successfully for all violation types")
            
        except Exception as e:
            print(f"⚠️ SQLite database initialization error: {e}")

    def _load_license_plate_cascade(self):
        """Load Haar cascade for license plate detection"""
        try:
            cascade_path = 'haarcascade_russian_plate_number.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                print("❌ Failed to load license plate cascade")
                return None
            print("✅ License plate cascade loaded successfully")
            return cascade
        except Exception as e:
            print(f"⚠️ Could not load license plate cascade: {e}")
            return None
    
    def detect_all_violations(self, frame, location=None):
        """Detect ALL types of traffic violations and extract license plates"""
        all_violations = []
        detected_vehicles = []
        
        try:
            # 1. Detect vehicles using YOLO
            vehicles = self._detect_vehicles_yolo(frame)
            detected_vehicles = vehicles
            
            # 2. Detect red light violations
            red_light_results = self.detect_red_light_violations_with_yolo(frame, vehicles)
            
            # 3. Detect helmet violations
            helmet_violations = self.detect_helmet_violations(frame, vehicles)
            
            # 4. Detect speeding violations
            speeding_violations = self.detect_speeding_violations(vehicles)
            
            # 5. Extract license plates for ALL violating vehicles
            for violation_list in [red_light_results['violations'], helmet_violations, speeding_violations]:
                for violation in violation_list:
                    if 'vehicle_id' in violation:
                        # Find the vehicle that committed the violation
                        vehicle = next((v for v in vehicles if v.get('id') == violation['vehicle_id']), None)
                        if vehicle:
                            # Extract license plate for this vehicle
                            plate_text = self._extract_license_plate_from_vehicle(frame, vehicle)
                            if plate_text and self.validate_license_plate(plate_text):
                                violation['license_plate'] = plate_text
                                
                                # Update database with violation
                                self._update_violation_database(violation, plate_text, vehicle, location)
                            
                            all_violations.append(violation)
            
            return {
                'success': True,
                'violations': all_violations,
                'vehicles_detected': len(vehicles),
                'violation_summary': {
                    'red_light': len(red_light_results['violations']),
                    'no_helmet': len(helmet_violations),
                    'speeding': len(speeding_violations),
                    'total': len(all_violations)
                },
                'traffic_light_status': red_light_results.get('traffic_light_color', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Complete violation detection error: {e}")
            return {
                'success': False,
                'violations': [],
                'vehicles_detected': 0,
                'error': str(e)
            }

    def detect_red_light_violations_with_yolo(self, frame, vehicles=None):
        """Detect red light violations with license plate extraction"""
        violations = []
        annotated_frame = frame.copy()
        
        try:
            # Detect traffic light color
            annotated_frame, color, confidence = self.detect_traffic_light_color_enhanced(annotated_frame)
            
            # Use provided vehicles or detect new ones
            if vehicles is None:
                vehicles = self._detect_vehicles_yolo(frame)
            
            # Only process if light is red with sufficient confidence
            if color == 'red' and confidence > 0.6:
                # Detect crossing line
                annotated_frame, mask_line = self.line_detector.detect_white_line(annotated_frame, color)
                
                # Check for vehicles crossing the line
                for vehicle in vehicles:
                    if self._is_vehicle_crossing_line(vehicle, mask_line):
                        violation_data = {
                            'type': 'RED_LIGHT_VIOLATION',
                            'vehicle_id': vehicle.get('id'),
                            'vehicle_type': vehicle['class_name'],
                            'confidence': confidence * vehicle['confidence'],
                            'severity': 'HIGH',
                            'timestamp': datetime.now().isoformat()
                        }
                        violations.append(violation_data)
            
            return {
                'violations': violations,
                'traffic_light_color': color,
                'traffic_light_confidence': confidence,
                'annotated_frame': annotated_frame
            }
            
        except Exception as e:
            print(f"❌ Red light detection error: {e}")
            return {
                'violations': [], 
                'traffic_light_color': 'unknown', 
                'traffic_light_confidence': 0.0
            }

    def detect_helmet_violations(self, frame, vehicles=None):
        """Detect helmet violations for motorcycles"""
        violations = []
        
        try:
            # Use provided vehicles or detect new ones
            if vehicles is None:
                vehicles = self._detect_vehicles_yolo(frame)
            
            # Find motorcycles
            motorcycles = [v for v in vehicles if v['class_name'] == 'motorcycle']
            
            for motorcycle in motorcycles:
                # Simple helmet check (in real system, use your helmet detection model)
                has_helmet = self._check_helmet_simple(frame, motorcycle)
                
                if not has_helmet:
                    violation_data = {
                        'type': 'NO_HELMET',
                        'vehicle_id': motorcycle.get('id'),
                        'vehicle_type': 'motorcycle',
                        'confidence': motorcycle['confidence'] * 0.9,  # Adjust confidence
                        'severity': 'HIGH',
                        'timestamp': datetime.now().isoformat()
                    }
                    violations.append(violation_data)
            
            return violations
            
        except Exception as e:
            print(f"❌ Helmet detection error: {e}")
            return []

    def detect_speeding_violations(self, vehicles, speed_limit=50):
        """Detect speeding violations"""
        violations = []
        
        try:
            for vehicle in vehicles:
                # Simulate speed detection (in real system, use tracking)
                speed = self._estimate_speed(vehicle)
                
                if speed > speed_limit:
                    violation_data = {
                        'type': 'SPEEDING',
                        'vehicle_id': vehicle.get('id'),
                        'vehicle_type': vehicle['class_name'],
                        'detected_speed': speed,
                        'speed_limit': speed_limit,
                        'confidence': vehicle['confidence'] * 0.8,
                        'severity': 'HIGH' if speed > speed_limit + 20 else 'MEDIUM',
                        'timestamp': datetime.now().isoformat()
                    }
                    violations.append(violation_data)
            
            return violations
            
        except Exception as e:
            print(f"❌ Speeding detection error: {e}")
            return []

    def _check_helmet_simple(self, frame, motorcycle):
        """Simple helmet check (replace with your trained model)"""
        # This is a placeholder - use your actual helmet detection model
        # For now, return random result for demonstration
        import random
        return random.choice([True, False, True])  # More likely to have helmet

    def _estimate_speed(self, vehicle):
        """Estimate vehicle speed (simplified - use proper tracking in real system)"""
        # This is a placeholder - use proper speed estimation from tracking
        import random
        return random.randint(30, 80)  # Random speed for demonstration

    def _detect_vehicles_yolo(self, frame, confidence_threshold=0.5):
        """Detect vehicles using YOLO"""
        try:
            results = self.yolo_model(frame, conf=confidence_threshold, verbose=False)
            vehicles = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = box.conf[0].item()
                        
                        # Filter for vehicles
                        if class_id in [2, 3, 5, 7] and confidence > confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            class_names = {
                                2: 'car', 
                                3: 'motorcycle', 
                                5: 'bus', 
                                7: 'truck'
                            }
                            
                            vehicles.append({
                                'id': len(vehicles) + 1,  # Simple ID assignment
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'class_name': class_names[class_id],
                                'class_id': class_id,
                                'confidence': confidence,
                                'center_x': int((x1 + x2) / 2),
                                'center_y': int((y1 + y2) / 2)
                            })
            
            return vehicles
            
        except Exception as e:
            print(f"❌ YOLO vehicle detection error: {e}")
            return []

    def _is_vehicle_crossing_line(self, vehicle, mask_line):
        """Check if vehicle is crossing the stop line"""
        center_x = vehicle['center_x']
        center_y = vehicle['center_y']
        
        if 0 <= center_x < mask_line.shape[1] and 0 <= center_y < mask_line.shape[0]:
            pixel_value = mask_line[center_y, center_x]
            if np.any(pixel_value > 0):
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
            print(f"❌ License plate extraction error for vehicle {vehicle.get('id')}: {e}")
        
        return None

    def _update_violation_database(self, violation, plate_number, vehicle, location=None):
        """Update SQLite database with violation details"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            violation_type = violation['type']
            vehicle_type = vehicle['class_name']
            confidence = violation.get('confidence', 0.0)
            
            # Update license_plates table
            cursor.execute("""
                INSERT OR REPLACE INTO license_plates 
                (plate_number, total_violations, red_light_violations, helmet_violations, speeding_violations, last_violation)
                VALUES (?, 
                    COALESCE((SELECT total_violations FROM license_plates WHERE plate_number = ?), 0) + 1,
                    COALESCE((SELECT red_light_violations FROM license_plates WHERE plate_number = ?), 0) + ?,
                    COALESCE((SELECT helmet_violations FROM license_plates WHERE plate_number = ?), 0) + ?,
                    COALESCE((SELECT speeding_violations FROM license_plates WHERE plate_number = ?), 0) + ?,
                    CURRENT_TIMESTAMP
                )
            """, (
                plate_number, plate_number, plate_number,
                1 if violation_type == 'RED_LIGHT_VIOLATION' else 0,
                plate_number,
                1 if violation_type == 'NO_HELMET' else 0,
                plate_number,
                1 if violation_type == 'SPEEDING' else 0
            ))
            
            # Insert into detailed violations table
            cursor.execute("""
                INSERT INTO traffic_violations 
                (plate_number, violation_type, vehicle_type, detected_speed, speed_limit, confidence, location_lat, location_lng, evidence_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plate_number,
                violation_type,
                vehicle_type,
                violation.get('detected_speed', 0),
                violation.get('speed_limit', 0),
                confidence,
                location.get('lat') if location else None,
                location.get('lng') if location else None,
                json.dumps(violation)
            ))
            
            # For helmet violations, also add to helmet_violations table
            if violation_type == 'NO_HELMET':
                cursor.execute("""
                    INSERT INTO helmet_violations (plate_number, vehicle_type, confidence)
                    VALUES (?, ?, ?)
                """, (plate_number, vehicle_type, confidence))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Violation recorded in SQLite: {plate_number} - {violation_type}")
            
        except Exception as e:
            print(f"❌ SQLite database error: {e}")

    def detect_traffic_light_color_enhanced(self, frame):
        """Enhanced traffic light detection"""
        hsv_color, hsv_confidence = self._detect_traffic_light_hsv(frame)
        
        # Add annotation
        self._annotate_traffic_light(frame, hsv_color, hsv_confidence, 'hsv')
        
        return frame, hsv_color, hsv_confidence

    def _detect_traffic_light_hsv(self, image):
        """HSV-based traffic light detection"""
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

        # Create masks and determine color
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        total_pixels = w * h
        if total_pixels == 0:
            return 'unknown', 0.0
            
        red_confidence = red_pixels / total_pixels
        yellow_confidence = yellow_pixels / total_pixels
        green_confidence = green_pixels / total_pixels
        
        max_confidence = max(red_confidence, yellow_confidence, green_confidence)
        
        if max_confidence < 0.1:
            return 'unknown', max_confidence
        elif red_confidence == max_confidence:
            return 'red', red_confidence
        elif yellow_confidence == max_confidence:
            return 'yellow', yellow_confidence
        else:
            return 'green', green_confidence

    def _annotate_traffic_light(self, image, color, confidence, method):
        """Add traffic light annotation"""
        message = f"Traffic Light: {color.upper()} - {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if color == 'red':
            text_color = (0, 0, 255)
        elif color == 'yellow':
            text_color = (0, 255, 255)
        elif color == 'green':
            text_color = (0, 255, 0)
        else:
            text_color = (255, 255, 255)
        
        cv2.putText(image, message, (15, 70), font, 0.7, text_color, 2, cv2.LINE_AA)

    def validate_license_plate(self, text):
        """Validate license plate format"""
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
        """Apply OCR to extract text from license plate"""
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

    def get_violation_statistics(self):
        """Get comprehensive violation statistics from SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_violations,
                    COUNT(DISTINCT plate_number) as unique_offenders,
                    SUM(red_light_violations) as total_red_light,
                    SUM(helmet_violations) as total_helmet,
                    SUM(speeding_violations) as total_speeding
                FROM license_plates
            """)
            stats = cursor.fetchone()
            
            # Top offenders
            cursor.execute("""
                SELECT plate_number, total_violations 
                FROM license_plates 
                ORDER BY total_violations DESC 
                LIMIT 10
            """)
            top_offenders = cursor.fetchall()
            
            # Recent violations
            cursor.execute("""
                SELECT violation_type, COUNT(*) as count
                FROM traffic_violations 
                WHERE timestamp >= datetime('now', '-1 day')
                GROUP BY violation_type
            """)
            recent_violations = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_violations': stats[0] if stats else 0,
                'unique_offenders': stats[1] if stats else 0,
                'violation_breakdown': {
                    'red_light': stats[2] if stats else 0,
                    'no_helmet': stats[3] if stats else 0,
                    'speeding': stats[4] if stats else 0
                },
                'top_offenders': [{'plate': row[0], 'count': row[1]} for row in top_offenders],
                'recent_violations': {row[0]: row[1] for row in recent_violations}
            }
            
        except Exception as e:
            print(f"❌ SQLite database error: {e}")
            return {}

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

# Test function
def test_violation_detector():
    """Test the violation detector"""
    print("🧪 Testing Complete Violation Detector...")
    
    detector = CompleteViolationDetector()
    
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect violations
    results = detector.detect_all_violations(test_image)
    
    print(f"🎯 Detection Results:")
    print(f"   Success: {results['success']}")
    print(f"   Vehicles detected: {results['vehicles_detected']}")
    print(f"   Violations found: {len(results['violations'])}")
    print(f"   Traffic light: {results['traffic_light_status']}")
    
    # Get statistics
    stats = detector.get_violation_statistics()
    print(f"📊 Database Statistics: {stats}")
    
    return results

if __name__ == "__main__":
    test_violation_detector()