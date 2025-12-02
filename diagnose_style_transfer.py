#!/usr/bin/env python3
"""
Diagnostic script to test neural style transfer step by step
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def diagnose_style_transfer():
    """Diagnose style transfer issues step by step"""
    print("🔧 Neural Style Transfer Diagnostics\n")
    
    # Test images - use synthetic images with high contrast
    test_dir = Path("tests/test_images")
    content_path = test_dir / "synthetic_content.jpg"
    style_path = test_dir / "synthetic_style.jpg"
    
    if not content_path.exists() or not style_path.exists():
        print("❌ Test images not found")
        return False
    
    print(f"📁 Content: {content_path}")
    print(f"📁 Style: {style_path}")
    
    # Load and analyze images
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')
    
    print(f"📐 Content size: {content_img.size}")
    print(f"📐 Style size: {style_img.size}")
    
    # Test if images are actually different
    content_array = np.array(content_img.resize((256, 256)))
    style_array = np.array(style_img.resize((256, 256)))
    
    diff = np.abs(content_array.astype(float) - style_array.astype(float))
    print(f"📊 Image difference: mean={diff.mean():.2f}, max={diff.max():.2f}")
    
    if diff.mean() < 10:
        print("⚠️  WARNING: Content and style images are very similar!")
        print("   This could cause minimal style transfer effects.")
    
    # Test VGG19 loading
    print("\\n🧠 Testing VGG19 model loading...")
    try:
        vgg = models.vgg19(weights='VGG19_Weights.DEFAULT').features
        print("✅ VGG19 loaded successfully")
        print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # Test feature extraction
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        content_tensor = transform(content_img).unsqueeze(0)
        print(f"📊 Content tensor range: {content_tensor.min():.3f} to {content_tensor.max():.3f}")
        
        # Extract features
        features = extract_features(vgg, content_tensor)
        print(f"✅ Extracted features from {len(features)} layers")
        
        for layer_name, feature in list(features.items())[:3]:
            print(f"   {layer_name}: {feature.shape}")
            
    except Exception as e:
        print(f"❌ VGG19 error: {e}")
        return False
    
    # Test style transfer with extreme settings
    print("\\n🎨 Testing neural style transfer...")
    try:
        from models.style_transfer import get_model
        model = get_model("neural")
        
        print("✅ Model loaded, testing with extreme settings:")
        print("   Style strength: 1.0 (maximum)")
        print("   Steps: 10 (for speed)")
        print("   Quality: low")
        
        import asyncio
        result = asyncio.run(model.transfer_style(
            content_image=content_img,
            style_image=style_img,
            style_strength=1.0,
            num_inference_steps=10,
            quality="low"
        ))
        
        # Analyze result
        content_resized = content_img.resize(result.size)
        content_array = np.array(content_resized)
        result_array = np.array(result)
        
        diff = np.abs(content_array.astype(float) - result_array.astype(float))
        print(f"\\n📊 Style transfer results:")
        print(f"   Mean difference: {diff.mean():.2f}")
        print(f"   Max difference: {diff.max():.2f}")
        print(f"   Pixels changed >10: {(diff > 10).sum()}/{diff.size} ({(diff > 10).mean()*100:.1f}%)")
        
        if diff.mean() > 20:
            print("✅ Excellent style transfer!")
        elif diff.mean() > 10:
            print("⚠️  Moderate style transfer")
        else:
            print("❌ Minimal style transfer - investigating...")
            
            # Additional diagnostics
            print("\\n🔍 Additional diagnostics:")
            
            # Check if result is just the input
            if np.allclose(content_array, result_array, atol=5):
                print("❌ Result is identical to input - no processing occurred")
            
            # Check loss calculation
            print("🔬 Testing loss calculation manually...")
            test_loss_calculation(vgg, content_tensor, transform(style_img).unsqueeze(0))
        
        # Save result for inspection
        result.save("diagnostic_result.jpg")
        print(f"💾 Result saved: diagnostic_result.jpg")
        
    except Exception as e:
        print(f"❌ Style transfer error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def extract_features(model, image):
    """Extract features from VGG model"""
    features = {}
    layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                  'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
    
    x = image
    conv_idx = 0
    for i, layer in enumerate(model):
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            if conv_idx < len(layer_names):
                features[layer_names[conv_idx]] = x
                conv_idx += 1
    
    return features

def test_loss_calculation(vgg, content_tensor, style_tensor):
    """Test loss calculation to ensure it's working"""
    try:
        content_features = extract_features(vgg, content_tensor)
        style_features = extract_features(vgg, style_tensor)
        
        # Calculate content loss
        content_loss = torch.mean((content_features['conv4_2'] - content_features['conv4_2']) ** 2)
        print(f"   Content loss (should be ~0): {content_loss.item():.6f}")
        
        # Calculate style loss
        def gram_matrix(tensor):
            batch_size, channels, height, width = tensor.size()
            features = tensor.view(batch_size * channels, height * width)
            gram = torch.mm(features, features.t())
            return gram.div(batch_size * channels * height * width)
        
        content_gram = gram_matrix(content_features['conv1_1'])
        style_gram = gram_matrix(style_features['conv1_1'])
        style_loss = torch.mean((content_gram - style_gram) ** 2)
        
        print(f"   Style loss: {style_loss.item():.6f}")
        
        if style_loss.item() < 1e-6:
            print("⚠️  WARNING: Style loss is very small - style image might be too similar to content")
        
    except Exception as e:
        print(f"❌ Loss calculation error: {e}")

if __name__ == "__main__":
    diagnose_style_transfer()