#!/usr/bin/env python3
"""
Test different image formats with the style transfer API
"""

import asyncio
import aiohttp
import base64
import json
from pathlib import Path

async def test_image_format(subject_path: str, style_path: str, format_name: str):
    """Test a specific image format combination"""
    print(f"🖼️  Testing {format_name}...")
    
    try:
        # Encode images
        with open(subject_path, "rb") as f:
            subject_b64 = base64.b64encode(f.read()).decode()
        with open(style_path, "rb") as f:
            style_b64 = base64.b64encode(f.read()).decode()
        
        payload = {
            "subject_image": subject_b64,
            "style_image": style_b64,
            "quality": "high",
            "style_strength": 0.8,
            "num_inference_steps": 20
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8080/v1/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 202:
                    data = await response.json()
                    print(f"✅ {format_name}: Job created {data['job_id']}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ {format_name}: Failed - {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ {format_name}: Exception - {e}")
        return False

async def main():
    """Test all available image format combinations"""
    print("🧪 Testing Different Image Formats\n")
    
    test_dir = Path("test_images")
    
    # Test different combinations
    test_cases = [
        ("subject.jpg", "style.jpg", "JPG + JPG"),
        ("subject.jpg", "style.jpeg", "JPG + JPEG"), 
        ("subject.avif", "style.jpg", "AVIF + JPG"),
        ("subject.avif", "style.jpeg", "AVIF + JPEG"),
    ]
    
    results = []
    for subject, style, name in test_cases:
        subject_path = test_dir / subject
        style_path = test_dir / style
        
        if subject_path.exists() and style_path.exists():
            success = await test_image_format(str(subject_path), str(style_path), name)
            results.append((name, success))
        else:
            print(f"⚠️  {name}: Files not found")
            results.append((name, False))
    
    print(f"\n📊 Results:")
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 {successful}/{total} format combinations working")

if __name__ == "__main__":
    asyncio.run(main())