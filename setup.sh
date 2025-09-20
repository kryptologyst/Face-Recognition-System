#!/bin/bash

# Face Recognition System Startup Script
# This script sets up and runs the face recognition system

set -e  # Exit on any error

echo "🎭 Face Recognition System Setup"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads known_faces logs temp

# Set up environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Setting up environment configuration..."
    cp config.env.example .env
    echo "📝 Please edit .env file with your configuration"
fi

# Initialize database
echo "🗄️ Initializing database..."
python3 -c "from database import create_tables; create_tables(); print('Database initialized')"

# Check if webcam is available
if command -v v4l2-ctl &> /dev/null; then
    echo "📹 Checking webcam availability..."
    if v4l2-ctl --list-devices &> /dev/null; then
        echo "✅ Webcam detected"
    else
        echo "⚠️ No webcam detected (optional for image processing)"
    fi
else
    echo "⚠️ Cannot check webcam (v4l2-utils not installed)"
fi

echo ""
echo "🚀 Setup complete! Choose how to run the system:"
echo ""
echo "1. Web Interface (recommended):"
echo "   python app.py"
echo "   Then open: http://localhost:8000"
echo ""
echo "2. Command Line Interface:"
echo "   python 0097.py --mode webcam"
echo "   python 0097.py --mode image --image path/to/image.jpg"
echo ""
echo "3. Docker (if Docker is installed):"
echo "   docker-compose up"
echo ""
echo "📖 For more information, see README.md"
echo ""

# Ask user what they want to do
read -p "Would you like to start the web interface now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🌐 Starting web interface..."
    python app.py
fi
