"""
FastAPI Web Application for Face Recognition System
"""
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import cv2
import numpy as np
import os
import logging
from datetime import datetime
import json

from database import get_db, User, FaceEncoding, RecognitionLog, create_tables
from face_engine import FaceRecognitionEngine
from config import config

# Create necessary directories
config.create_directories()

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition System",
    description="Modern face recognition system with web interface",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize face recognition engine
face_engine = FaceRecognitionEngine()

# Create database tables
create_tables()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{config.LOG_FILE}"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Main page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face Recognition System</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }
            .header p {
                margin: 10px 0 0 0;
                opacity: 0.9;
                font-size: 1.1em;
            }
            .content {
                padding: 40px;
            }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 30px;
                margin: 40px 0;
            }
            .feature {
                background: #f8f9fa;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                transition: transform 0.3s ease;
            }
            .feature:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            }
            .feature h3 {
                color: #667eea;
                margin-bottom: 15px;
            }
            .upload-area {
                border: 2px dashed #667eea;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 30px 0;
                background: #f8f9fa;
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                background: #e9ecef;
                border-color: #5a6fd8;
            }
            .upload-area.dragover {
                background: #e3f2fd;
                border-color: #2196f3;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
                margin: 10px;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .btn-secondary {
                background: #6c757d;
            }
            .btn-success {
                background: #28a745;
            }
            .btn-danger {
                background: #dc3545;
            }
            .results {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                display: none;
            }
            .face-result {
                background: white;
                padding: 20px;
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .confidence {
                font-weight: bold;
                color: #28a745;
            }
            .unknown {
                color: #dc3545;
            }
            .video-container {
                margin: 30px 0;
                text-align: center;
            }
            #video {
                border-radius: 10px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            }
            .controls {
                margin: 20px 0;
            }
            .status {
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                display: none;
            }
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .status.info {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎭 Face Recognition System</h1>
                <p>Advanced AI-powered face detection and recognition</p>
            </div>
            
            <div class="content">
                <div class="features">
                    <div class="feature">
                        <h3>📸 Real-time Detection</h3>
                        <p>Detect and recognize faces in real-time using your webcam</p>
                        <button class="btn" onclick="startWebcam()">Start Webcam</button>
                    </div>
                    
                    <div class="feature">
                        <h3>🖼️ Image Upload</h3>
                        <p>Upload images to detect and recognize faces</p>
                        <button class="btn" onclick="document.getElementById('fileInput').click()">Upload Image</button>
                    </div>
                    
                    <div class="feature">
                        <h3>👥 Face Registration</h3>
                        <p>Register new faces to expand the recognition database</p>
                        <button class="btn btn-success" onclick="showRegistration()">Register Face</button>
                    </div>
                </div>

                <div class="upload-area" id="uploadArea">
                    <h3>📁 Drop images here or click to upload</h3>
                    <p>Supports JPG, PNG, GIF formats</p>
                    <input type="file" id="fileInput" accept="image/*" style="display: none;" multiple>
                </div>

                <div class="video-container" id="videoContainer" style="display: none;">
                    <video id="video" width="640" height="480" autoplay></video>
                    <div class="controls">
                        <button class="btn btn-danger" onclick="stopWebcam()">Stop Webcam</button>
                        <button class="btn btn-success" onclick="captureFrame()">Capture Frame</button>
                    </div>
                </div>

                <div class="results" id="results">
                    <h3>🔍 Recognition Results</h3>
                    <div id="resultContent"></div>
                </div>

                <div class="status" id="status"></div>
            </div>
        </div>

        <script>
            let stream = null;
            let video = document.getElementById('video');
            let uploadArea = document.getElementById('uploadArea');
            let fileInput = document.getElementById('fileInput');
            let results = document.getElementById('results');
            let resultContent = document.getElementById('resultContent');
            let status = document.getElementById('status');

            // File upload handling
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                handleFiles(e.dataTransfer.files);
            });

            fileInput.addEventListener('change', (e) => {
                handleFiles(e.target.files);
            });

            function handleFiles(files) {
                Array.from(files).forEach(file => {
                    if (file.type.startsWith('image/')) {
                        uploadImage(file);
                    }
                });
            }

            function uploadImage(file) {
                const formData = new FormData();
                formData.append('file', file);
                
                showStatus('Processing image...', 'info');
                
                fetch('/api/recognize', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    displayResults(data);
                    showStatus('Image processed successfully!', 'success');
                })
                .catch(error => {
                    console.error('Error:', error);
                    showStatus('Error processing image: ' + error.message, 'error');
                });
            }

            function startWebcam() {
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(mediaStream => {
                        stream = mediaStream;
                        video.srcObject = stream;
                        document.getElementById('videoContainer').style.display = 'block';
                        showStatus('Webcam started', 'success');
                    })
                    .catch(error => {
                        console.error('Error accessing webcam:', error);
                        showStatus('Error accessing webcam: ' + error.message, 'error');
                    });
            }

            function stopWebcam() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                    video.srcObject = null;
                    document.getElementById('videoContainer').style.display = 'none';
                    showStatus('Webcam stopped', 'info');
                }
            }

            function captureFrame() {
                if (video.videoWidth === 0) return;
                
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                
                canvas.toBlob(blob => {
                    const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
                    uploadImage(file);
                }, 'image/jpeg');
            }

            function displayResults(data) {
                results.style.display = 'block';
                resultContent.innerHTML = '';
                
                if (data.faces && data.faces.length > 0) {
                    data.faces.forEach((face, index) => {
                        const faceDiv = document.createElement('div');
                        faceDiv.className = 'face-result';
                        faceDiv.innerHTML = `
                            <h4>Face ${index + 1}</h4>
                            <p><strong>Name:</strong> <span class="${face.name === 'Unknown' ? 'unknown' : 'confidence'}">${face.name}</span></p>
                            <p><strong>Confidence:</strong> <span class="confidence">${(face.confidence * 100).toFixed(1)}%</span></p>
                        `;
                        resultContent.appendChild(faceDiv);
                    });
                } else {
                    resultContent.innerHTML = '<p>No faces detected in the image.</p>';
                }
            }

            function showStatus(message, type) {
                status.textContent = message;
                status.className = 'status ' + type;
                status.style.display = 'block';
                setTimeout(() => {
                    status.style.display = 'none';
                }, 3000);
            }

            function showRegistration() {
                alert('Face registration feature coming soon!');
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/recognize")
async def recognize_faces(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Recognize faces in uploaded image"""
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Get known encodings from database
        known_encodings = []
        known_names = []
        
        face_encodings = db.query(FaceEncoding).all()
        for encoding_record in face_encodings:
            encoding = np.frombuffer(encoding_record.encoding, dtype=np.float64)
            known_encodings.append(encoding)
            user = db.query(User).filter(User.id == encoding_record.user_id).first()
            known_names.append(user.full_name if user else f"User_{encoding_record.user_id}")
        
        # Recognize faces
        results = face_engine.recognize_face(image, known_encodings, known_names)
        
        # Log recognition
        for result in results:
            user_id = None
            if result['name'] != 'Unknown':
                # Find user by name
                user = db.query(User).filter(User.full_name == result['name']).first()
                user_id = user.id if user else None
            
            log_entry = RecognitionLog(
                user_id=user_id,
                confidence_score=result['confidence'],
                recognition_time=datetime.utcnow()
            )
            db.add(log_entry)
        
        db.commit()
        
        return {
            "faces": results,
            "total_faces": len(results),
            "image_size": f"{image.shape[1]}x{image.shape[0]}"
        }
        
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/register")
async def register_face(
    file: UploadFile = File(...),
    name: str = Form(...),
    email: str = Form(...),
    db: Session = Depends(get_db)
):
    """Register a new face"""
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Detect faces
        face_locations = face_engine.detect_faces(image)
        if not face_locations:
            raise HTTPException(status_code=400, detail="No faces detected in image")
        
        # Use the first detected face
        face_location = face_locations[0]
        face_encoding = face_engine.encode_face(image, face_location)
        
        if face_encoding is None:
            raise HTTPException(status_code=400, detail="Could not encode face")
        
        # Create user
        user = User(
            username=email.split('@')[0],
            email=email,
            full_name=name
        )
        db.add(user)
        db.flush()  # Get the user ID
        
        # Save face encoding
        face_encoding_record = FaceEncoding(
            user_id=user.id,
            encoding=face_encoding.tobytes(),
            image_path=file.filename,
            confidence_score=1.0
        )
        db.add(face_encoding_record)
        db.commit()
        
        return {
            "message": "Face registered successfully",
            "user_id": user.id,
            "faces_detected": len(face_locations)
        }
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users")
async def get_users(db: Session = Depends(get_db)):
    """Get all registered users"""
    users = db.query(User).all()
    return [
        {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "created_at": user.created_at.isoformat()
        }
        for user in users
    ]

@app.get("/api/logs")
async def get_recognition_logs(db: Session = Depends(get_db)):
    """Get recognition logs"""
    logs = db.query(RecognitionLog).order_by(RecognitionLog.recognition_time.desc()).limit(100).all()
    return [
        {
            "id": log.id,
            "user_id": log.user_id,
            "confidence_score": log.confidence_score,
            "recognition_time": log.recognition_time.isoformat(),
            "is_verified": log.is_verified
        }
        for log in logs
    ]

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    total_users = db.query(User).count()
    total_encodings = db.query(FaceEncoding).count()
    total_logs = db.query(RecognitionLog).count()
    
    return {
        "total_users": total_users,
        "total_face_encodings": total_encodings,
        "total_recognition_logs": total_logs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
