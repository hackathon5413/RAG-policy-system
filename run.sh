#!/bin/bash

# Quick Start Script for Policy RAG System
# This script sets up and runs the system with Python virtual environment

set -e  # Exit on any error

echo "🚀 Policy RAG System - Quick Python Setup"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.11+ first."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "✅ Using $PYTHON_CMD"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Set API key (use first one from the list)
export GEMINI_API_KEY_1="AIzaSyBuKHF-9oTwbCgWbY3B2-TmbJ6a1vd5iu4"
echo "🔑 API key configured"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 Starting the server..."
echo "Your Policy RAG System will be available at:"
echo "  📍 API: http://localhost:8080"
echo "  📚 Documentation: http://localhost:8080/docs"
echo "  🏥 Health Check: http://localhost:8080/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server
python server.py
