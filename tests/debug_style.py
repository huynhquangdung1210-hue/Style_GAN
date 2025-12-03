#!/usr/bin/env python3
"""
Debug style transfer by running it directly
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path

def debug_style_transfer():
    """Debug the style transfer process step by step"""
    print("🔧 Debug Style Transfer Process\n")
    
    # Load test images
    test_dir = Path("tests/test_images")
    content_path = test_dir / "subject.jpg"
    style_path = test_dir / "style.jpg"
    
    if not content_path.exists() or not style_path.exists():
        print("❌ Test images not found")
        return False
    
    print(f"📁 Loading: {content_path}")
    print(f"📁 Loading: {style_path}")
    
    # Load images
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')
    
    print(f"📐 Content size: {content_img.size}")
    print(f"📐 Style size: {style_img.size}")
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                           std=[1/0.229, 1/0.224, 1/0.225])
    ])
    
    # Process images
    content_tensor = transform(content_img).unsqueeze(0)
    style_tensor = transform(style_img).unsqueeze(0)
    
    print(f"📊 Content tensor range: {content_tensor.min():.3f} to {content_tensor.max():.3f}")
    print(f"📊 Style tensor range: {style_tensor.min():.3f} to {style_tensor.max():.3f}")
    
    # Initialize output
    output = content_tensor.clone().requires_grad_(True)
    
    # Test inverse transform
    print("\\n🔄 Testing inverse transform...")
    test_denorm = inverse_transform(content_tensor.squeeze(0))
    print(f"📊 Denormalized range: {test_denorm.min():.3f} to {test_denorm.max():.3f}")
    
    # Convert back to image
    test_denorm_clamped = torch.clamp(test_denorm, 0, 1)
    test_img = transforms.ToPILImage()(test_denorm_clamped)
    test_img.save("debug_denorm_test.jpg")
    print("💾 Saved debug_denorm_test.jpg")
    
    # Test with style transfer weights
    print("\\n⚖️  Testing with extreme style weights...")
    
    # Apply a simple style "transfer" (just blend the images)
    alpha = 0.8  # Heavy style weight
    blended = (1 - alpha) * content_tensor + alpha * style_tensor
    
    print(f"📊 Blended range: {blended.min():.3f} to {blended.max():.3f}")
    
    # Denormalize and save
    blended_denorm = inverse_transform(blended.squeeze(0))
    blended_clamped = torch.clamp(blended_denorm, 0, 1)
    blended_img = transforms.ToPILImage()(blended_clamped)
    blended_img.save("debug_blended.jpg")
    print("💾 Saved debug_blended.jpg")
    
    # Analyze pixel differences
    content_array = np.array(content_img.resize((512, 512)))
    blended_array = np.array(blended_img)
    
    diff = np.abs(content_array.astype(float) - blended_array.astype(float))
    print(f"📊 Pixel difference mean: {diff.mean():.2f}")
    print(f"📊 Pixel difference max: {diff.max():.2f}")
    
    return True

if __name__ == "__main__":
    debug_style_transfer()