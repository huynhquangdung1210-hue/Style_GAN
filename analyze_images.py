#!/usr/bin/env python3
"""
Better visual comparison tool
"""

from pathlib import Path
from PIL import Image
import numpy as np

def analyze_image_differences():
    """Analyze differences between recent results and original"""
    print("🔍 Advanced Image Analysis\n")
    
    # Find most recent result
    current_dir = Path(".")
    result_files = list(current_dir.glob("result_*.jpg")) + list(current_dir.glob("direct_*.jpg"))
    
    if not result_files:
        print("❌ No result files found")
        return False
    
    # Get the most recent
    latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"🔍 Analyzing: {latest_result}")
    
    # Load original for comparison
    test_dir = Path("tests/test_images")
    original_path = test_dir / "subject.jpg"
    
    if not original_path.exists():
        print("❌ Original image not found")
        return False
    
    # Load images
    result_img = Image.open(latest_result)
    original_img = Image.open(original_path).resize(result_img.size)
    
    # Convert to arrays
    result_array = np.array(result_img)
    original_array = np.array(original_img)
    
    print(f"📐 Image size: {result_img.size}")
    print(f"🎨 Image mode: {result_img.mode}")
    
    # Calculate differences
    diff = np.abs(result_array.astype(float) - original_array.astype(float))
    
    print(f"\\n📊 Pixel Analysis:")
    print(f"📊 Mean difference: {diff.mean():.2f}")
    print(f"📊 Std difference: {diff.std():.2f}")
    print(f"📊 Max difference: {diff.max():.2f}")
    print(f"📊 Min difference: {diff.min():.2f}")
    
    # Threshold analysis
    thresholds = [5, 10, 20, 50]
    for threshold in thresholds:
        changed_pixels = (diff > threshold).sum()
        total_pixels = diff.size
        percentage = (changed_pixels / total_pixels) * 100
        print(f"📊 Pixels changed >{threshold}: {changed_pixels:,} / {total_pixels:,} ({percentage:.1f}%)")
    
    # Color channel analysis
    if len(result_array.shape) == 3:
        print(f"\\n🔴 Red channel mean diff: {diff[:,:,0].mean():.2f}")
        print(f"🟢 Green channel mean diff: {diff[:,:,1].mean():.2f}")
        print(f"🔵 Blue channel mean diff: {diff[:,:,2].mean():.2f}")
    
    # Overall assessment
    print(f"\\n🎯 Assessment:")
    if diff.mean() > 30:
        print("🔥 Excellent style transfer - dramatic changes!")
    elif diff.mean() > 15:
        print("✅ Good style transfer - noticeable changes!")
    elif diff.mean() > 5:
        print("⚠️  Moderate style transfer - subtle changes")
    else:
        print("❌ Poor style transfer - minimal changes")
    
    # Create difference visualization
    diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
    if len(diff_normalized.shape) == 3:
        diff_gray = np.mean(diff_normalized, axis=2).astype(np.uint8)
    else:
        diff_gray = diff_normalized
    
    diff_img = Image.fromarray(diff_gray, mode='L')
    diff_img.save("difference_map.jpg")
    print(f"🗺️  Difference map saved: difference_map.jpg")
    
    return True

if __name__ == "__main__":
    analyze_image_differences()