"""
Configuration management for Face Recognition System
"""
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Config:
    """Application configuration"""
    
    # Basic settings
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./face_recognition.db")
    
    # File upload settings
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "uploads")
    KNOWN_FACES_FOLDER: str = os.getenv("KNOWN_FACES_FOLDER", "known_faces")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    ALLOWED_EXTENSIONS: set = set(os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png,gif").split(","))
    
    # Face recognition settings
    FACE_RECOGNITION_TOLERANCE: float = float(os.getenv("FACE_RECOGNITION_TOLERANCE", "0.6"))
    FACE_DETECTION_MODEL: str = os.getenv("FACE_DETECTION_MODEL", "hog")  # hog or cnn
    FACE_ENCODING_MODEL: str = os.getenv("FACE_ENCODING_MODEL", "large")  # small or large
    
    # Security settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "jwt-secret-key-change-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_HOURS: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "face_recognition.log")
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.KNOWN_FACES_FOLDER, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

# Global config instance
config = Config()
