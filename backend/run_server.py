"""
Server Runner Script
Multimodal Attendance & Engagement Agent
"""

import os
import uvicorn
from dotenv import load_dotenv

# Load environment configuration from .env file
load_dotenv()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

if __name__ == "__main__":
    print("=" * 60)
    print(" Starting Multimodal Attendance & Engagement Agent API...")
    print(f" Listening on: http://{HOST}:{PORT}")
    print(f" Swagger Docs: http://{HOST}:{PORT}/docs")
    print(f" Health Check: http://{HOST}:{PORT}/health")
    print("=" * 60)
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=DEBUG)
