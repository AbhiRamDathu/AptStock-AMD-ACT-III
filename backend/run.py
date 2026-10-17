"""
Entry point for AptStock Pro Backend
Run: python run.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 Starting AptStock Pro API Server")
    print("="*70)
    print("📍 Server: http://localhost:8001")
    print("📚 API Docs: http://localhost:8001/docs")
    print("🔐 Authentication: Enabled")
    print("📊 Database: MongoDB")
    print("="*70 + "\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
