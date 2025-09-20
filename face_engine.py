"""
Modern Face Recognition Engine with multiple detection methods
"""
import face_recognition
import cv2
import numpy as np
import mediapipe as mp
import insightface
import onnxruntime as ort
from typing import List, Tuple, Optional, Dict, Any
import logging
from PIL import Image
import pickle
import os
from config import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class FaceRecognitionEngine:
    """Modern face recognition engine with multiple detection methods"""
    
    def __init__(self):
        self.tolerance = config.FACE_RECOGNITION_TOLERANCE
        self.detection_model = config.FACE_DETECTION_MODEL
        self.encoding_model = config.FACE_ENCODING_MODEL
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Initialize InsightFace (if available)
        try:
            self.insightface_app = insightface.app.FaceAnalysis()
            self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            self.insightface_available = True
        except Exception as e:
            logger.warning(f"InsightFace not available: {e}")
            self.insightface_available = False
    
    def detect_faces_mediapipe(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        face_locations = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Convert to face_recognition format (top, right, bottom, left)
                top = y
                right = x + width
                bottom = y + height
                left = x
                face_locations.append((top, right, bottom, left))
        
        return face_locations
    
    def detect_faces_insightface(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using InsightFace"""
        if not self.insightface_available:
            return []
        
        try:
            faces = self.insightface_app.get(image)
            face_locations = []
            
            for face in faces:
                bbox = face.bbox.astype(int)
                # Convert to face_recognition format
                top, left, bottom, right = bbox[1], bbox[0], bbox[3], bbox[2]
                face_locations.append((top, right, bottom, left))
            
            return face_locations
        except Exception as e:
            logger.error(f"InsightFace detection error: {e}")
            return []
    
    def detect_faces_face_recognition(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using face_recognition library"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(
            rgb_image, model=self.detection_model
        )
        return face_locations
    
    def detect_faces(self, image: np.ndarray, method: str = "auto") -> List[Tuple[int, int, int, int]]:
        """Detect faces using specified method"""
        if method == "mediapipe":
            return self.detect_faces_mediapipe(image)
        elif method == "insightface" and self.insightface_available:
            return self.detect_faces_insightface(image)
        elif method == "face_recognition":
            return self.detect_faces_face_recognition(image)
        elif method == "auto":
            # Try multiple methods and return the best result
            methods = ["face_recognition", "mediapipe"]
            if self.insightface_available:
                methods.append("insightface")
            
            best_locations = []
            max_faces = 0
            
            for method_name in methods:
                try:
                    locations = getattr(self, f"detect_faces_{method_name}")(image)
                    if len(locations) > max_faces:
                        max_faces = len(locations)
                        best_locations = locations
                except Exception as e:
                    logger.warning(f"Method {method_name} failed: {e}")
            
            return best_locations
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def encode_face(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Encode a face from image and location"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(
                rgb_image, [face_location], model=self.encoding_model
            )
            return face_encodings[0] if face_encodings else None
        except Exception as e:
            logger.error(f"Face encoding error: {e}")
            return None
    
    def encode_faces(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Encode multiple faces"""
        encodings = []
        for location in face_locations:
            encoding = self.encode_face(image, location)
            if encoding is not None:
                encodings.append(encoding)
        return encodings
    
    def compare_faces(self, known_encodings: List[np.ndarray], face_encoding: np.ndarray) -> Tuple[List[bool], List[float]]:
        """Compare face encoding with known encodings"""
        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=self.tolerance
        )
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        return matches, distances
    
    def recognize_face(self, image: np.ndarray, known_encodings: List[np.ndarray], 
                      known_names: List[str], method: str = "auto") -> List[Dict[str, Any]]:
        """Recognize faces in image"""
        face_locations = self.detect_faces(image, method)
        face_encodings = self.encode_faces(image, face_locations)
        
        results = []
        for i, (location, encoding) in enumerate(zip(face_locations, face_encodings)):
            matches, distances = self.compare_faces(known_encodings, encoding)
            
            name = "Unknown"
            confidence = 0.0
            
            if matches:
                best_match_index = np.argmin(distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    confidence = 1.0 - distances[best_match_index]
            
            results.append({
                "location": location,
                "name": name,
                "confidence": confidence,
                "face_id": i
            })
        
        return results
    
    def save_encoding(self, encoding: np.ndarray, filepath: str) -> bool:
        """Save face encoding to file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(encoding, f)
            return True
        except Exception as e:
            logger.error(f"Error saving encoding: {e}")
            return False
    
    def load_encoding(self, filepath: str) -> Optional[np.ndarray]:
        """Load face encoding from file"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading encoding: {e}")
            return None
    
    def batch_process_images(self, image_paths: List[str], known_encodings: List[np.ndarray], 
                           known_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Process multiple images for face recognition"""
        results = {}
        
        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    results[image_path] = self.recognize_face(image, known_encodings, known_names)
                else:
                    logger.warning(f"Could not load image: {image_path}")
                    results[image_path] = []
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results[image_path] = []
        
        return results
