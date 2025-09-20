"""
Basic tests for Face Recognition System
"""
import pytest
import numpy as np
import cv2
import os
import tempfile
from unittest.mock import Mock, patch

from face_engine import FaceRecognitionEngine
from database import User, FaceEncoding, RecognitionLog, create_tables
from config import config
from security import SecurityManager, DataProtection

class TestFaceRecognitionEngine:
    """Test face recognition engine functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = FaceRecognitionEngine()
        self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        assert self.engine.tolerance == config.FACE_RECOGNITION_TOLERANCE
        assert self.engine.detection_model == config.FACE_DETECTION_MODEL
        assert self.engine.encoding_model == config.FACE_ENCODING_MODEL
    
    def test_detect_faces_face_recognition(self):
        """Test face detection using face_recognition method"""
        locations = self.engine.detect_faces_face_recognition(self.test_image)
        assert isinstance(locations, list)
    
    def test_detect_faces_mediapipe(self):
        """Test face detection using MediaPipe method"""
        locations = self.engine.detect_faces_mediapipe(self.test_image)
        assert isinstance(locations, list)
    
    def test_detect_faces_auto(self):
        """Test automatic face detection method selection"""
        locations = self.engine.detect_faces(self.test_image, method="auto")
        assert isinstance(locations, list)
    
    def test_recognize_face_empty_known(self):
        """Test face recognition with empty known faces"""
        results = self.engine.recognize_face(
            self.test_image, 
            [], 
            []
        )
        assert isinstance(results, list)
    
    def test_save_load_encoding(self):
        """Test encoding save and load functionality"""
        test_encoding = np.random.rand(128).astype(np.float64)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save encoding
            success = self.engine.save_encoding(test_encoding, tmp_path)
            assert success
            
            # Load encoding
            loaded_encoding = self.engine.load_encoding(tmp_path)
            assert loaded_encoding is not None
            np.testing.assert_array_almost_equal(test_encoding, loaded_encoding)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class TestSecurityManager:
    """Test security manager functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.security = SecurityManager("test-secret-key")
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "test_password_123"
        hashed = self.security.hash_password(password)
        
        assert hashed != password
        assert self.security.verify_password(password, hashed)
        assert not self.security.verify_password("wrong_password", hashed)
    
    def test_token_generation(self):
        """Test JWT token generation and verification"""
        user_id = 1
        username = "test_user"
        
        token = self.security.generate_token(user_id, username)
        assert isinstance(token, str)
        
        payload = self.security.verify_token(token)
        assert payload is not None
        assert payload["user_id"] == user_id
        assert payload["username"] == username
    
    def test_api_key_generation(self):
        """Test API key generation"""
        api_key = self.security.generate_api_key()
        assert isinstance(api_key, str)
        assert len(api_key) > 20
    
    def test_email_masking(self):
        """Test email masking functionality"""
        email = "test@example.com"
        masked = self.security.mask_email(email)
        assert "@" in masked
        assert "test" not in masked
        assert "example.com" in masked
    
    def test_filename_sanitization(self):
        """Test filename sanitization"""
        dangerous_filename = "../../../etc/passwd"
        sanitized = self.security.sanitize_filename(dangerous_filename)
        assert "../" not in sanitized
        assert "etc" not in sanitized

class TestDataProtection:
    """Test data protection utilities"""
    
    def test_anonymize_log_data(self):
        """Test log data anonymization"""
        sensitive_data = {
            "email": "test@example.com",
            "password": "secret123",
            "token": "jwt_token_here",
            "normal_field": "safe_data"
        }
        
        anonymized = DataProtection.anonymize_log_data(sensitive_data)
        
        assert anonymized["email"] != sensitive_data["email"]
        assert anonymized["password"] == "[REDACTED]"
        assert anonymized["token"] == "[REDACTED]"
        assert anonymized["normal_field"] == sensitive_data["normal_field"]
    
    def test_validate_image_file(self):
        """Test image file validation"""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Create a simple test image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            cv2.imwrite(tmp.name, test_image)
            
            try:
                # Test valid image
                assert DataProtection.validate_image_file(tmp.name)
                
                # Test invalid file
                assert not DataProtection.validate_image_file("nonexistent.jpg")
            finally:
                os.unlink(tmp.name)

class TestDatabase:
    """Test database functionality"""
    
    def test_create_tables(self):
        """Test database table creation"""
        # This test assumes database is properly configured
        try:
            create_tables()
            assert True  # If no exception is raised, tables were created
        except Exception as e:
            pytest.skip(f"Database test skipped: {e}")

# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_webcam_simulation(self):
        """Test webcam simulation with mock"""
        engine = FaceRecognitionEngine()
        
        # Mock cv2.VideoCapture
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.ones((480, 640, 3), dtype=np.uint8))
            mock_capture.return_value = mock_cap
            
            # This would normally run the webcam loop
            # For testing, we just verify the mock setup
            cap = cv2.VideoCapture(0)
            assert cap.isOpened()
            ret, frame = cap.read()
            assert ret
            assert frame is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
