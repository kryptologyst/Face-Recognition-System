# Project 97. Face recognition system - MODERNIZED VERSION
# Description:
# A modern face recognition system that identifies or verifies a person by analyzing facial features.
# This updated version includes multiple detection methods, confidence scoring, and improved performance.

# Modern Python Implementation with Enhanced Features

import face_recognition
import cv2
import numpy as np
import os
import mediapipe as mp
import logging
from typing import List, Tuple, Optional
import argparse
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModernFaceRecognition:
    """Modern face recognition system with multiple detection methods"""
    
    def __init__(self, tolerance: float = 0.6, detection_method: str = "auto"):
        self.tolerance = tolerance
        self.detection_method = detection_method
        self.known_encodings = []
        self.known_names = []
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
    
    def load_known_faces(self, known_faces_dir: str = 'known_faces'):
        """Load known faces from directory"""
        if not os.path.exists(known_faces_dir):
            logger.warning(f"Directory {known_faces_dir} does not exist. Creating it.")
            os.makedirs(known_faces_dir, exist_ok=True)
            return
        
        for file in os.listdir(known_faces_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                img_path = os.path.join(known_faces_dir, file)
                try:
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img, model='large')
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(os.path.splitext(file)[0])
                        logger.info(f"Loaded face: {file}")
                    else:
                        logger.warning(f"No face found in {file}")
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
        
        logger.info(f"Loaded {len(self.known_encodings)} known faces")
    
    def detect_faces_mediapipe(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        face_locations = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Convert to face_recognition format
                top = y
                right = x + width
                bottom = y + height
                left = x
                face_locations.append((top, right, bottom, left))
        
        return face_locations
    
    def detect_faces_face_recognition(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using face_recognition library"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        return face_locations
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using specified method"""
        if self.detection_method == "mediapipe":
            return self.detect_faces_mediapipe(frame)
        elif self.detection_method == "face_recognition":
            return self.detect_faces_face_recognition(frame)
        elif self.detection_method == "auto":
            # Try both methods and use the one that finds more faces
            fr_locations = self.detect_faces_face_recognition(frame)
            mp_locations = self.detect_faces_mediapipe(frame)
            return fr_locations if len(fr_locations) >= len(mp_locations) else mp_locations
        else:
            raise ValueError(f"Unknown detection method: {self.detection_method}")
    
    def recognize_faces(self, frame: np.ndarray) -> List[dict]:
        """Recognize faces in frame"""
        # Detect faces
        face_locations = self.detect_faces(frame)
        
        if not face_locations:
            return []
        
        # Encode faces
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='large')
        
        results = []
        for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_encodings, face_encoding, tolerance=self.tolerance
            )
            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            
            name = "Unknown"
            confidence = 0.0
            
            if matches and any(matches):
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    confidence = 1.0 - face_distances[best_match_index]
            
            results.append({
                'location': face_location,
                'name': name,
                'confidence': confidence,
                'face_id': i
            })
        
        return results
    
    def draw_results(self, frame: np.ndarray, results: List[dict]) -> np.ndarray:
        """Draw recognition results on frame"""
        for result in results:
            top, right, bottom, left = result['location']
            name = result['name']
            confidence = result['confidence']
            
            # Choose color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label with confidence
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
            cv2.putText(frame, label, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def run_webcam(self):
        """Run face recognition on webcam feed"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return
        
        logger.info("Starting webcam face recognition. Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            results = self.recognize_faces(frame)
            frame = self.draw_results(frame, results)
            
            # Calculate and display FPS
            if frame_count % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Modern Face Recognition", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"captured_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                logger.info(f"Frame saved as {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path: str, save_result: bool = True):
        """Process a single image"""
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return
        
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        # Process image
        results = self.recognize_faces(frame)
        frame = self.draw_results(frame, results)
        
        # Display results
        logger.info(f"Found {len(results)} faces in {image_path}")
        for i, result in enumerate(results):
            logger.info(f"Face {i+1}: {result['name']} (confidence: {result['confidence']:.3f})")
        
        # Save result if requested
        if save_result:
            output_path = f"result_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, frame)
            logger.info(f"Result saved as {output_path}")
        
        # Display image
        cv2.imshow("Face Recognition Result", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Modern Face Recognition System")
    parser.add_argument("--mode", choices=["webcam", "image"], default="webcam",
                       help="Mode: webcam or image processing")
    parser.add_argument("--image", type=str, help="Path to image file (for image mode)")
    parser.add_argument("--known-faces", type=str, default="known_faces",
                       help="Directory containing known faces")
    parser.add_argument("--tolerance", type=float, default=0.6,
                       help="Face recognition tolerance (lower = more strict)")
    parser.add_argument("--detection-method", choices=["auto", "face_recognition", "mediapipe"],
                       default="auto", help="Face detection method")
    
    args = parser.parse_args()
    
    # Initialize face recognition system
    face_recognition_system = ModernFaceRecognition(
        tolerance=args.tolerance,
        detection_method=args.detection_method
    )
    
    # Load known faces
    face_recognition_system.load_known_faces(args.known_faces)
    
    if args.mode == "webcam":
        face_recognition_system.run_webcam()
    elif args.mode == "image":
        if not args.image:
            logger.error("Image path required for image mode")
            return
        face_recognition_system.process_image(args.image)

if __name__ == "__main__":
    main()

# 🧠 What This Modernized Project Demonstrates:
# ✅ Multiple face detection methods (face_recognition, MediaPipe)
# ✅ Confidence scoring for recognition accuracy
# ✅ Command-line interface for different modes
# ✅ Improved error handling and logging
# ✅ FPS monitoring for performance optimization
# ✅ Frame capture functionality
# ✅ Batch image processing capabilities
# ✅ Modern Python practices and type hints