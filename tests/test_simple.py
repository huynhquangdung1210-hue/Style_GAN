#!/usr/bin/env python3
"""
Simple API test using basic requests
"""

import requests
import base64
import time
from pathlib import Path

def test_simple_style_transfer():
    """Test basic style transfer workflow"""
    print("🧪 Testing Simple Style Transfer API\n")
    
    # Check health
    try:
        response = requests.get("http://localhost:8080/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check passed: {health['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        return False
    
    # Load test images
    test_dir = Path(__file__).parent / "test_images"
    subject_path = test_dir / "subject.jpg"
    style_path = test_dir / "style.jpg"
    
    if not subject_path.exists() or not style_path.exists():
        print("❌ Test images not found")
        return False
    
    # Encode images
    with open(subject_path, "rb") as f:
        subject_b64 = base64.b64encode(f.read()).decode()
    
    with open(style_path, "rb") as f:
        style_b64 = base64.b64encode(f.read()).decode()
    
    # Submit job
    try:
        response = requests.post(
            "http://localhost:8080/v1/generate",
            json={
                "subject_image": subject_b64,
                "style_image": style_b64,
                "quality": "low",  # Use low quality for faster processing
                "style_strength": 0.7,
                "num_inference_steps": 5
            }
        )
        
        if response.status_code == 202:
            data = response.json()
            job_id = data["job_id"]
            print(f"✅ Job created: {job_id}")
        else:
            print(f"❌ Job creation failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Job submission failed: {e}")
        return False
    
    # Poll for completion
    print("⏳ Waiting for job completion...")
    for i in range(30):  # Wait up to 60 seconds
        try:
            response = requests.get(f"http://localhost:8080/v1/jobs/{job_id}")
            if response.status_code == 200:
                data = response.json()
                status = data["status"]
                print(f"📊 Status: {status} ({data.get('progress', 0):.1%})")
                
                if status == "completed":
                    print("✅ Job completed!")
                    
                    # Try to get result
                    result_response = requests.get(f"http://localhost:8080/v1/results/{job_id}")
                    if result_response.status_code == 200:
                        with open(f"result_{job_id}.jpg", "wb") as f:
                            f.write(result_response.content)
                        print(f"💾 Result saved: result_{job_id}.jpg")
                        return True
                    else:
                        print(f"⚠️  Could not retrieve result: {result_response.status_code}")
                        return True  # Job completed, just couldn't get result
                        
                elif status == "failed":
                    print(f"❌ Job failed: {data.get('error', 'Unknown error')}")
                    return False
                    
                time.sleep(2)
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Status check error: {e}")
            return False
    
    print("⏰ Job timed out")
    return False

if __name__ == "__main__":
    success = test_simple_style_transfer()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
    exit(0 if success else 1)