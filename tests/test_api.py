#!/usr/bin/env python3
"""
Basic API functionality test for the style transfer service.
Tests all major endpoints and validates responses.
"""

import asyncio
import aiohttp
import base64
import json
import time
from pathlib import Path
from typing import Optional

class StyleTransferAPITest:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    async def test_health_endpoint(self) -> bool:
        """Test the health check endpoint."""
        print("🔍 Testing health endpoint...")
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                data = await response.json()
                assert response.status == 200
                assert data["status"] == "healthy"
                print("✅ Health endpoint working")
                return True
        except Exception as e:
            print(f"❌ Health endpoint failed: {e}")
            return False
    
    async def test_generate_endpoint(self, subject_path: str, style_path: str) -> Optional[str]:
        """Test the style transfer generation endpoint."""
        print("🎨 Testing style transfer generation...")
        try:
            # Encode images
            subject_b64 = self.encode_image_to_base64(subject_path)
            style_b64 = self.encode_image_to_base64(style_path)
            
            payload = {
                "subject_image": subject_b64,
                "style_image": style_b64,
                "quality": "high",
                "style_strength": 0.8,
                "num_inference_steps": 20
            }
            
            async with self.session.post(
                f"{self.base_url}/v1/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 202:
                    data = await response.json()
                    job_id = data["job_id"]
                    print(f"✅ Job created: {job_id}")
                    return job_id
                else:
                    error_text = await response.text()
                    print(f"❌ Generate endpoint failed: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Generate endpoint failed: {e}")
            return None
    
    async def test_job_status(self, job_id: str) -> bool:
        """Test job status endpoint and wait for completion."""
        print(f"⏳ Checking job status for {job_id}...")
        
        max_wait = 300  # 5 minutes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                async with self.session.get(f"{self.base_url}/v1/jobs/{job_id}") as response:
                    data = await response.json()
                    
                    status = data["status"]
                    print(f"📊 Job status: {status}")
                    
                    if status == "completed":
                        print("✅ Job completed successfully")
                        if "result_url" in data:
                            print(f"🖼️  Result available at: {data['result_url']}")
                        return True
                    elif status == "failed":
                        print(f"❌ Job failed: {data.get('error', 'Unknown error')}")
                        return False
                    elif status in ["pending", "processing"]:
                        await asyncio.sleep(5)
                        continue
                    else:
                        print(f"❓ Unknown status: {status}")
                        return False
                        
            except Exception as e:
                print(f"❌ Status check failed: {e}")
                return False
        
        print("⏰ Job timed out")
        return False
    
    async def test_metrics_endpoint(self) -> bool:
        """Test the metrics endpoint."""
        print("📊 Testing metrics endpoint...")
        try:
            async with self.session.get(f"{self.base_url}/metrics") as response:
                assert response.status == 200
                metrics_text = await response.text()
                assert "http_requests_total" in metrics_text
                print("✅ Metrics endpoint working")
                return True
        except Exception as e:
            print(f"❌ Metrics endpoint failed: {e}")
            return False

async def run_tests():
    """Run all API tests."""
    print("🚀 Starting Style Transfer API Tests\n")
    
    # Check for test images
    test_dir = Path(__file__).parent
    subject_image = test_dir / "test_images" / "subject.jpg"
    style_image = test_dir / "test_images" / "style.jpg"
    
    if not subject_image.exists() or not style_image.exists():
        print("❌ Test images not found. Please add test images to tests/test_images/")
        print("   Required: subject.jpg, style.jpg")
        return False
    
    async with StyleTransferAPITest() as tester:
        # Test health endpoint
        if not await tester.test_health_endpoint():
            return False
        
        print()
        
        # Test metrics endpoint
        if not await tester.test_metrics_endpoint():
            print("⚠️  Metrics endpoint failed (non-critical)")
        
        print()
        
        # Test style transfer generation
        job_id = await tester.test_generate_endpoint(
            str(subject_image), 
            str(style_image)
        )
        
        if not job_id:
            return False
        
        print()
        
        # Test job status and wait for completion
        if not await tester.test_job_status(job_id):
            return False
    
    print("\n🎉 All tests completed successfully!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Style Transfer API")
    parser.add_argument(
        "--url", 
        default="http://localhost:8080",
        help="Base URL of the API (default: http://localhost:8080)"
    )
    
    args = parser.parse_args()
    
    # Update base URL
    StyleTransferAPITest.__init__ = lambda self, base_url=args.url: setattr(self, 'base_url', base_url) or setattr(self, 'session', None)
    
    # Run tests
    success = asyncio.run(run_tests())
    exit(0 if success else 1)