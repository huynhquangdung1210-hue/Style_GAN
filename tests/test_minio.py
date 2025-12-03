#!/usr/bin/env python3
"""
Quick test script to verify MinIO storage integration
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config import get_settings
from utils.storage_simple import get_storage_client
import structlog

async def test_minio():
    """Test MinIO storage functionality."""
    print("🧪 Testing MinIO Storage Integration")
    
    # Get settings
    settings = get_settings()
    print(f"📋 Storage Type: {settings.storage.storage_type}")
    print(f"🏠 MinIO Endpoint: {settings.storage.minio_endpoint}")
    print(f"🪣 MinIO Bucket: {settings.storage.minio_bucket}")
    
    try:
        # Get storage client
        client = get_storage_client()
        print("✅ Storage client initialized")
        
        # Test upload
        test_data = b"Hello MinIO from Style Transfer API!"
        test_key = "test/hello.txt"
        
        print("📤 Testing upload...")
        url = await client.upload_bytes(test_data, test_key, "text/plain")
        print(f"✅ Upload successful: {url}")
        
        # Test download
        print("📥 Testing download...")
        downloaded = await client.download_file(test_key)
        if downloaded == test_data:
            print("✅ Download successful - data matches!")
        else:
            print("❌ Download failed - data mismatch")
            return False
        
        # Test file URL
        file_url = client.get_file_url(test_key)
        print(f"🔗 File URL: {file_url}")
        
        # Test deletion
        print("🗑️ Testing deletion...")
        deleted = await client.delete_file(test_key)
        if deleted:
            print("✅ Deletion successful")
        else:
            print("⚠️ Deletion may have failed")
        
        print("🎉 All MinIO tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ MinIO test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Configure basic logging
    import os
    os.environ.setdefault("STORAGE_TYPE", "minio")
    os.environ.setdefault("MINIO_ENDPOINT", "localhost:9000")
    
    success = asyncio.run(test_minio())
    sys.exit(0 if success else 1)