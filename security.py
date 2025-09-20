"""
Security utilities for Face Recognition System
"""
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
import logging
import os
import json

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SecurityManager:
    """Security manager for authentication and data protection"""
    
    def __init__(self, secret_key: str, jwt_algorithm: str = "HS256", jwt_expiration_hours: int = 24):
        self.secret_key = secret_key
        self.jwt_algorithm = jwt_algorithm
        self.jwt_expiration_hours = jwt_expiration_hours
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def generate_token(self, user_id: int, username: str) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user_id,
            "username": username,
            "exp": datetime.utcnow() + timedelta(hours=self.jwt_expiration_hours),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(32)
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for storage"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def mask_email(self, email: str) -> str:
        """Mask email address for privacy"""
        if '@' not in email:
            return email
        local, domain = email.split('@', 1)
        masked_local = local[0] + '*' * (len(local) - 2) + local[-1] if len(local) > 2 else local
        return f"{masked_local}@{domain}"
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks"""
        import re
        # Remove any path components
        filename = os.path.basename(filename)
        # Remove or replace dangerous characters
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        # Ensure it's not empty and not too long
        if not filename or len(filename) > 255:
            filename = f"file_{secrets.token_hex(8)}"
        return filename

class DataProtection:
    """Data protection utilities"""
    
    @staticmethod
    def anonymize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize sensitive data in logs"""
        sensitive_fields = ['email', 'password', 'token', 'api_key', 'face_encoding']
        anonymized = data.copy()
        
        for field in sensitive_fields:
            if field in anonymized:
                if field == 'email':
                    anonymized[field] = SecurityManager.mask_email(anonymized[field])
                else:
                    anonymized[field] = '[REDACTED]'
        
        return anonymized
    
    @staticmethod
    def validate_image_file(file_path: str, max_size: int = 10 * 1024 * 1024) -> bool:
        """Validate image file for security"""
        import os
        from PIL import Image
        
        try:
            # Check file size
            if os.path.getsize(file_path) > max_size:
                return False
            
            # Check if it's a valid image
            with Image.open(file_path) as img:
                img.verify()
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def clean_temp_files(temp_dir: str = "temp"):
        """Clean temporary files"""
        import os
        import shutil
        import time
        
        if not os.path.exists(temp_dir):
            return
        
        current_time = time.time()
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                # Delete files older than 1 hour
                if current_time - os.path.getmtime(file_path) > 3600:
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned temp file: {filename}")
                    except Exception as e:
                        logger.error(f"Error cleaning temp file {filename}: {e}")

class AuditLogger:
    """Audit logging for security events"""
    
    def __init__(self, log_file: str = "logs/security_audit.log"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_login_attempt(self, username: str, success: bool, ip_address: str = None):
        """Log login attempts"""
        event = {
            "event": "login_attempt",
            "username": username,
            "success": success,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._write_log(event)
    
    def log_face_recognition(self, user_id: int, confidence: float, image_path: str = None):
        """Log face recognition events"""
        event = {
            "event": "face_recognition",
            "user_id": user_id,
            "confidence": confidence,
            "image_path": image_path,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._write_log(event)
    
    def log_data_access(self, user_id: int, resource: str, action: str):
        """Log data access events"""
        event = {
            "event": "data_access",
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._write_log(event)
    
    def _write_log(self, event: Dict[str, Any]):
        """Write event to audit log"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{json.dumps(event)}\n")
        except Exception as e:
            logger.error(f"Error writing audit log: {e}")

# Rate limiting
class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        current_time = datetime.utcnow()
        
        # Clean old entries
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if (current_time - req_time).seconds < self.window_seconds
            ]
        else:
            self.requests[identifier] = []
        
        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(current_time)
            return True
        
        return False
