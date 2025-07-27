

import uvicorn
from config import config 

def start_server():
    """Start the FastAPI server"""
    print(f"🚀 Starting {config.app_name}")
    print(f"📍 Server: http://{config.host}:{config.port}")
    print(f"📚 API Docs: http://{config.host}:{config.port}/docs")
    print(f"🔑 Auth Token: {config.bearer_token}")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
