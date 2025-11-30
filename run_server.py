#!/usr/bin/env python3
"""
Simple API Server Startup Script
"""

import sys
import uvicorn
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    # Start the API server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )