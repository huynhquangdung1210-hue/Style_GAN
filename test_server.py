#!/usr/bin/env python3
"""
Simple test server to verify connectivity
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Test API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Test server running"}

@app.get("/")
async def root():
    return {"message": "Test API", "status": "ok"}

if __name__ == "__main__":
    print("🚀 Starting test server on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")