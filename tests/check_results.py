#!/usr/bin/env python3
"""
Quick visual check of style transfer results
"""

from pathlib import Path
from PIL import Image
import sys

def check_recent_results():
    """Check the most recent test results"""
    current_dir = Path(".")
    test_dir = Path("tests")
    
    # Find recent result files
    result_files = list(current_dir.glob("result_*.jpg")) + list(test_dir.glob("result_*.jpg"))
    
    if not result_files:
        print("❌ No result files found")
        return False
    
    # Get the most recent result
    latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
    
    print(f"🔍 Checking: {latest_result}")
    
    # Load and analyze the image
    try:
        img = Image.open(latest_result)
        print(f"📐 Image size: {img.size}")
        print(f"🎨 Image mode: {img.mode}")
        
        # Get some basic stats
        if img.mode == 'RGB':
            # Convert to grayscale to check if it's not just a copy
            gray = img.convert('L')
            pixels = list(gray.getdata())
            
            # Check for variation
            min_val = min(pixels)
            max_val = max(pixels)
            avg_val = sum(pixels) / len(pixels)
            
            print(f"📊 Pixel range: {min_val} - {max_val}")
            print(f"📊 Average brightness: {avg_val:.1f}")
            
            # Check for artistic variation (good style transfer should have varied pixel values)
            if max_val - min_val > 100:
                print("✅ Good pixel variation - likely successful style transfer")
            else:
                print("⚠️  Low pixel variation - might be unchanged")
        
        print(f"✅ Result image is valid: {latest_result.name}")
        
        # Compare with source if available
        test_images_dir = Path("tests/test_images")
        if test_images_dir.exists():
            subject_path = test_images_dir / "subject.jpg"
            if subject_path.exists():
                original = Image.open(subject_path)
                if original.size == img.size:
                    print("📏 Same dimensions as original - good!")
                else:
                    print(f"📏 Resized: {original.size} → {img.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking image: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking Style Transfer Results\n")
    success = check_recent_results()
    if success:
        print("\n🎉 Image check completed!")
    else:
        print("\n💥 Image check failed!")