# yolo_detector.py - ENHANCED VERSION
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time

class YOLOVehicleDetector:
    def __init__(self, model_size='n'):
        """
        Initialize YOLO model
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        """
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = self.model.names
        
        # Tracking and analytics
        self.track_history = defaultdict(lambda: [])
        self.detection_count = 0
        self.start_time = time.time()
        
        print(f"✅ YOLOv8{model_size.upper()} Vehicle Detector initialized!")
        print(f"🎯 Tracking classes: {[self.class_names[cls] for cls in self.vehicle_classes]}")
    
    def detect_vehicles(self, frame, confidence_threshold=0.5, track_vehicles=True):
        """
        Detect vehicles in frame with optional tracking
        Returns: list of vehicle detections with tracking IDs
        """
        try:
            # Run YOLO inference
            results = self.model.track(frame, 
                                     conf=confidence_threshold, 
                                     persist=True,  # For tracking across frames
                                     verbose=False)
            
            vehicles = []
            current_time = time.time()
            
            for result in results:
                if result.boxes is not None and result.boxes.id is not None:
                    # With tracking
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    class_ids = result.boxes.cls.int().cpu().tolist()
                    confidences = result.boxes.conf.float().cpu().tolist()
                    
                    for box, track_id, class_id, confidence in zip(boxes, track_ids, class_ids, confidences):
                        if class_id in self.vehicle_classes:
                            x_center, y_center, width, height = box.tolist()
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            
                            # Update track history
                            track = self.track_history[track_id]
                            track.append((float(x_center), float(y_center)))
                            if len(track) > 30:  # Keep last 30 points
                                track.pop(0)
                            
                            vehicles.append({
                                'track_id': track_id,
                                'bbox': [x1, y1, x2, y2],
                                'class_id': class_id,
                                'class_name': self.class_names[class_id],
                                'confidence': confidence,
                                'center_x': x_center,
                                'center_y': y_center,
                                'width': width,
                                'height': height,
                                'area': width * height,
                                'speed': self._calculate_speed(track_id),
                                'track_history': track.copy()
                            })
                else:
                    # Fallback to detection without tracking
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = box.conf[0].item()
                        
                        if class_id in self.vehicle_classes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            vehicles.append({
                                'track_id': len(vehicles) + 1,  # Simple ID
                                'bbox': [x1, y1, x2, y2],
                                'class_id': class_id,
                                'class_name': self.class_names[class_id],
                                'confidence': confidence,
                                'center_x': (x1 + x2) / 2,
                                'center_y': (y1 + y2) / 2,
                                'width': x2 - x1,
                                'height': y2 - y1,
                                'area': (x2 - x1) * (y2 - y1),
                                'speed': 0,
                                'track_history': []
                            })
            
            self.detection_count += len(vehicles)
            return vehicles
            
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            return []
    
    def _calculate_speed(self, track_id):
        """Calculate approximate speed based on track history"""
        track = self.track_history[track_id]
        if len(track) < 2:
            return 0
        
        # Calculate distance moved between last two points
        (x1, y1), (x2, y2) = track[-2], track[-1]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Simple speed estimation (pixels per frame)
        # In real system, you'd convert to km/h using camera calibration
        return distance * 10  # Scale factor for demonstration
    
    def draw_detections(self, frame, vehicles, show_tracking=True, show_speed=True):
        """Draw bounding boxes, labels, and tracking on frame"""
        for vehicle in vehicles:
            x1, y1, x2, y2 = map(int, vehicle['bbox'])
            class_name = vehicle['class_name']
            confidence = vehicle['confidence']
            track_id = vehicle.get('track_id', 0)
            
            # Choose color based on vehicle type
            color = self._get_vehicle_color(class_name)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label_parts = [f"{class_name} {confidence:.2f}"]
            if track_id:
                label_parts.append(f"ID:{track_id}")
            if show_speed and vehicle.get('speed', 0) > 0:
                label_parts.append(f"{vehicle['speed']:.1f}px/f")
            
            label = " | ".join(label_parts)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw tracking history
            if show_tracking and vehicle.get('track_history'):
                tracks = vehicle['track_history']
                for i in range(1, len(tracks)):
                    cv2.line(frame, 
                            (int(tracks[i-1][0]), int(tracks[i-1][1])),
                            (int(tracks[i][0]), int(tracks[i][1])),
                            color, 2)
        
        return frame
    
    def _get_vehicle_color(self, class_name):
        """Get distinct colors for different vehicle types"""
        color_map = {
            'car': (0, 255, 0),      # Green
            'motorcycle': (255, 0, 0), # Blue  
            'bus': (0, 255, 255),    # Yellow
            'truck': (255, 0, 255),  # Magenta
        }
        return color_map.get(class_name, (0, 255, 0))  # Default green
    
    def get_detection_statistics(self, vehicles):
        """Get statistics about current detections"""
        stats = {
            'total_vehicles': len(vehicles),
            'by_type': {},
            'average_confidence': 0,
            'total_area': 0
        }
        
        if vehicles:
            confidences = [v['confidence'] for v in vehicles]
            areas = [v['area'] for v in vehicles]
            
            stats['average_confidence'] = np.mean(confidences)
            stats['total_area'] = np.sum(areas)
            
            # Count by vehicle type
            for vehicle in vehicles:
                v_type = vehicle['class_name']
                stats['by_type'][v_type] = stats['by_type'].get(v_type, 0) + 1
        
        # Add system statistics
        stats['total_detections'] = self.detection_count
        stats['uptime'] = time.time() - self.start_time
        stats['active_tracks'] = len(self.track_history)
        
        return stats
    
    def draw_statistics(self, frame, vehicles):
        """Draw detection statistics on frame"""
        stats = self.get_detection_statistics(vehicles)
        
        # Create statistics text
        lines = [
            f"Vehicles: {stats['total_vehicles']}",
            f"Confidence: {stats['average_confidence']:.2f}",
            f"Tracks: {stats['active_tracks']}",
            "Types: " + ", ".join([f"{k}:{v}" for k, v in stats['by_type'].items()])
        ]
        
        # Draw background for text
        y_start = 30
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (10, y_start + i*30 - 20), 
                         (10 + text_size[0] + 10, y_start + i*30 + 5), 
                         (0, 0, 0), -1)
        
        # Draw text
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (15, y_start + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def detect_specific_vehicles(self, frame, vehicle_types=None, confidence_threshold=0.5):
        """
        Detect specific types of vehicles
        vehicle_types: list of vehicle types to detect (e.g., ['car', 'motorcycle'])
        """
        if vehicle_types is None:
            vehicle_types = ['car', 'motorcycle', 'bus', 'truck']
        
        # Convert vehicle names to class IDs
        type_to_id = {
            'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7
        }
        
        target_classes = [type_to_id[v_type] for v_type in vehicle_types if v_type in type_to_id]
        
        vehicles = self.detect_vehicles(frame, confidence_threshold)
        filtered_vehicles = [v for v in vehicles if v['class_id'] in target_classes]
        
        return filtered_vehicles
    
    def clear_tracking(self):
        """Clear tracking history"""
        self.track_history.clear()
        print("✅ Tracking history cleared")

# Example usage and testing
def test_yolo_detector():
    """Test the YOLO detector with a sample image"""
    detector = YOLOVehicleDetector('n')
    
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect vehicles
    vehicles = detector.detect_vehicles(test_image)
    
    print(f"🎯 Detected {len(vehicles)} vehicles")
    
    # Draw detections
    annotated_frame = detector.draw_detections(test_image.copy(), vehicles)
    annotated_frame = detector.draw_statistics(annotated_frame, vehicles)
    
    # Show statistics
    stats = detector.get_detection_statistics(vehicles)
    print(f"📊 Statistics: {stats}")
    
    return annotated_frame, vehicles

if __name__ == "__main__":
    print("🧪 Testing YOLO Vehicle Detector...")
    annotated_frame, vehicles = test_yolo_detector()
    print("✅ YOLO Detector test completed!")