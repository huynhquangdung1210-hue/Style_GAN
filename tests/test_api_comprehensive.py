#!/usr/bin/env python3
"""
Comprehensive test suite for the real ML-powered style transfer API.
Tests the complete pipeline from job submission to result retrieval.
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
                print(f"✅ Health endpoint working - Storage: {data.get('storage_healthy')}, Queue: {data.get('queue_healthy')}")
                return True
        except Exception as e:
            print(f"❌ Health endpoint failed: {e}")
            return False
    
    async def test_queue_stats(self) -> bool:
        """Test queue statistics endpoint."""
        print("📊 Testing queue stats...")
        try:
            async with self.session.get(f"{self.base_url}/v1/queue/stats") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Queue stats: {data.get('queue_length', 0)} jobs queued, {data.get('total_jobs', 0)} total")
                    return True
                else:
                    print(f"⚠️  Queue stats unavailable (status: {response.status})")
                    return True  # Non-critical
        except Exception as e:
            print(f"❌ Queue stats failed: {e}")
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
                "quality": "medium",
                "style_strength": 0.8,
                "num_inference_steps": 10  # Reduced for faster testing
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
                    print(f"   📝 Quality: {data['quality']}, Estimated time: {data['estimated_time']}s")
                    print(f"   🖼️  Subject: {data['subject_image_info']['format']} {data['subject_image_info']['size']}")
                    print(f"   🎭 Style: {data['style_image_info']['format']} {data['style_image_info']['size']}")
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
        
        max_wait = 180  # 3 minutes max
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < max_wait:
            try:
                async with self.session.get(f"{self.base_url}/v1/jobs/{job_id}") as response:
                    if response.status != 200:
                        print(f"❌ Status check failed with status: {response.status}")
                        return False
                    
                    data = await response.json()
                    status = data["status"]
                    progress = data.get("progress", 0)
                    
                    if status != last_status:
                        print(f"📊 Job status: {status} ({progress:.1%})")
                        last_status = status
                    
                    if status == "completed":
                        print("✅ Job completed successfully")
                        if "result_url" in data:
                            print(f"🖼️  Result available at: {data['result_url']}")
                        return True
                    elif status == "failed":
                        error_msg = data.get('error', 'Unknown error')
                        print(f"❌ Job failed: {error_msg}")
                        return False
                    elif status in ["pending", "processing"]:
                        await asyncio.sleep(2)
                        continue
                    else:
                        print(f"❓ Unknown status: {status}")
                        return False
                        
            except Exception as e:
                print(f"❌ Status check failed: {e}")
                return False
        
        print("⏰ Job timed out")
        return False
    
    async def test_result_retrieval(self, job_id: str) -> bool:
        """Test result image retrieval."""
        print(f"🖼️  Testing result retrieval for {job_id}...")
        try:
            # Test result image
            async with self.session.get(f"{self.base_url}/v1/results/{job_id}") as response:
                if response.status == 200:
                    content = await response.read()
                    print(f"✅ Result image retrieved ({len(content)} bytes)")
                    
                    # Save result for inspection
                    result_path = Path(f"test_result_{job_id}.jpg")
                    with open(result_path, "wb") as f:
                        f.write(content)
                    print(f"💾 Result saved to: {result_path}")
                else:
                    print(f"❌ Result retrieval failed: {response.status}")
                    return False
            
            # Test thumbnail
            async with self.session.get(f"{self.base_url}/v1/thumbnails/{job_id}") as response:
                if response.status == 200:
                    content = await response.read()
                    print(f"✅ Thumbnail retrieved ({len(content)} bytes)")
                    
                    # Save thumbnail for inspection
                    thumb_path = Path(f"test_thumbnail_{job_id}.jpg")
                    with open(thumb_path, "wb") as f:
                        f.write(content)
                    print(f"💾 Thumbnail saved to: {thumb_path}")
                else:
                    print(f"⚠️  Thumbnail retrieval failed: {response.status}")
            
            return True
            
        except Exception as e:
            print(f"❌ Result retrieval failed: {e}")
            return False
    
    async def test_job_cancellation(self, subject_path: str, style_path: str) -> bool:
        """Test job cancellation."""
        print("🚫 Testing job cancellation...")
        try:
            # Create a job
            subject_b64 = self.encode_image_to_base64(subject_path)
            style_b64 = self.encode_image_to_base64(style_path)
            
            payload = {
                "subject_image": subject_b64,
                "style_image": style_b64,
                "quality": "high",  # Use high quality for longer processing
                "style_strength": 0.9,
                "num_inference_steps": 50
            }
            
            async with self.session.post(
                f"{self.base_url}/v1/generate",
                json=payload
            ) as response:
                if response.status != 202:
                    print("❌ Could not create job for cancellation test")
                    return False
                
                data = await response.json()
                job_id = data["job_id"]
            
            # Wait a moment then cancel
            await asyncio.sleep(1)
            
            async with self.session.delete(f"{self.base_url}/v1/jobs/{job_id}") as response:
                if response.status == 200:
                    print(f"✅ Job {job_id} cancelled successfully")
                    return True
                else:
                    error = await response.text()
                    print(f"❌ Cancellation failed: {error}")
                    return False
                    
        except Exception as e:
            print(f"❌ Cancellation test failed: {e}")
            return False

async def run_comprehensive_tests():
    """Run all API tests."""
    print("🧪 Starting Comprehensive Style Transfer API Tests\n")
    
    # Check for test images
    test_dir = Path(__file__).parent / "test_images"
    subject_image = test_dir / "subject.jpg"
    style_image = test_dir / "style.jpg"
    
    if not subject_image.exists() or not style_image.exists():
        print("❌ Test images not found. Please add test images to tests/test_images/")
        print("   Required: subject.jpg, style.jpg")
        return False
    
    async with StyleTransferAPITest() as tester:
        # Test health endpoint
        if not await tester.test_health_endpoint():
            return False
        print()
        
        # Test queue stats
        if not await tester.test_queue_stats():
            print("⚠️  Queue stats failed (non-critical)")
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
        print()
        
        # Test result retrieval
        if not await tester.test_result_retrieval(job_id):
            return False
        print()
        
        # Test job cancellation (optional)
        if not await tester.test_job_cancellation(str(subject_image), str(style_image)):
            print("⚠️  Cancellation test failed (non-critical)")
        print()
    
    print("🎉 All comprehensive tests completed successfully!")
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
    success = asyncio.run(run_comprehensive_tests())
    exit(0 if success else 1)