#!/usr/bin/env python3
"""
Create synthetic test images for style transfer debugging
"""

from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

def create_test_images():
    """Create synthetic images with very distinct styles"""
    output_dir = Path("tests/test_images")
    
    # Content image: Simple geometric shapes (black and white)
    content = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(content)
    
    # Draw simple shapes
    draw.rectangle([100, 100, 200, 200], fill='black')
    draw.ellipse([300, 150, 400, 250], fill='gray') 
    draw.polygon([(250, 300), (300, 400), (200, 400)], fill='darkgray')
    
    content.save(output_dir / "synthetic_content.jpg", 'JPEG', quality=90)
    print("✅ Created synthetic_content.jpg")
    
    # Style image: Colorful abstract pattern
    style = Image.new('RGB', (512, 512), 'blue')
    draw = ImageDraw.Draw(style)
    
    # Create colorful pattern
    for i in range(0, 512, 25):
        for j in range(0, 512, 25):
            colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[(i + j) // 25 % len(colors)]
            draw.rectangle([i, j, i+20, j+20], fill=color)
    
    style.save(output_dir / "synthetic_style.jpg", 'JPEG', quality=90)
    print("✅ Created synthetic_style.jpg")
    
    # Analyze difference
    content_array = np.array(content)
    style_array = np.array(style)
    diff = np.abs(content_array.astype(float) - style_array.astype(float))
    print(f"📊 Synthetic image difference: mean={diff.mean():.2f}, max={diff.max():.2f}")
    
    return output_dir / "synthetic_content.jpg", output_dir / "synthetic_style.jpg"

if __name__ == "__main__":
    print("🎨 Creating synthetic test images...")
    create_test_images()