#!/usr/bin/env python3
"""
Startup script for the LLM-Powered Intelligent Query-Retrieval System
Run the FastAPI server for the HackRX challenge
"""

import uvicorn
from app.config import settings

def start_server():
    """Start the FastAPI server"""
    print(f"🚀 Starting {settings.app_name}")
    print(f"📍 Server: http://{settings.host}:{settings.port}")
    print(f"📚 API Docs: http://{settings.host}:{settings.port}/docs")
    print(f"🔑 Auth Token: {settings.bearer_token}")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
