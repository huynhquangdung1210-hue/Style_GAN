#!/usr/bin/env python3
"""
Direct style transfer test
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from models.style_transfer import get_model
from PIL import Image
import asyncio

async def test_direct_style_transfer():
    """Test style transfer directly"""
    print("🎨 Direct Style Transfer Test\n")
    
    # Load test images
    test_dir = Path("tests/test_images")
    content_path = test_dir / "subject.jpg"
    style_path = test_dir / "style_artistic.jpg"  # Use the artistic style
    
    if not style_path.exists():
        style_path = test_dir / "style.jpg"  # Fallback
    
    print(f"📁 Content: {content_path}")
    print(f"📁 Style: {style_path}")
    
    # Load images
    content_img = Image.open(content_path)
    style_img = Image.open(style_path)
    
    # Get neural model
    model = get_model("neural")
    
    # Test with strong style
    print("\\n🔧 Testing with strong style transfer...")
    result = await model.transfer_style(
        content_image=content_img,
        style_image=style_img,
        style_strength=0.9,  # Very strong style
        num_inference_steps=10,  # Enough iterations
        quality="medium"
    )
    
    # Save result
    result.save("direct_test_result.jpg")
    print("💾 Saved: direct_test_result.jpg")
    
    # Analyze differences
    import numpy as np
    
    content_resized = content_img.resize((512, 512))
    content_array = np.array(content_resized)
    result_array = np.array(result)
    
    diff = np.abs(content_array.astype(float) - result_array.astype(float))
    print(f"\\n📊 Analysis:")
    print(f"📊 Mean pixel difference: {diff.mean():.2f}")
    print(f"📊 Max pixel difference: {diff.max():.2f}")
    print(f"📊 Pixels changed >10: {(diff > 10).sum():.0f} / {diff.size:.0f} ({(diff > 10).mean()*100:.1f}%)")
    
    if diff.mean() > 20:
        print("✅ Good style transfer - significant changes!")
    elif diff.mean() > 5:
        print("⚠️  Mild style transfer - some changes")
    else:
        print("❌ Poor style transfer - minimal changes")

if __name__ == "__main__":
    asyncio.run(test_direct_style_transfer())