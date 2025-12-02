"""
Abstract base class and implementations for style transfer models.
"""

import asyncio
import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class StyleTransferModel(ABC):
    """Abstract base class for style transfer models."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Style transfer model initialized on device: {self.device}")
        
    @abstractmethod
    async def load_model(self) -> None:
        """Load the model weights and prepare for inference."""
        pass
    
    @abstractmethod
    async def transfer_style(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        style_strength: float = 0.8,
        num_inference_steps: int = 20,
        quality: str = "high"
    ) -> Image.Image:
        """
        Apply style transfer to content image.
        
        Args:
            content_image: PIL Image of content
            style_image: PIL Image of style reference
            style_strength: How strong the style should be applied (0.0-1.0)
            num_inference_steps: Number of inference steps
            quality: Quality setting ("low", "medium", "high")
            
        Returns:
            Stylized PIL Image
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        pass


class NeuralStyleTransfer(StyleTransferModel):
    """
    Neural Style Transfer implementation using VGG-based approach.
    This is a working implementation that can be used immediately.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None
        self.loaded = False
        
    async def load_model(self) -> None:
        """Load VGG19 model for neural style transfer."""
        try:
            # Run model loading in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._load_model_sync
            )
            self.loaded = True
            logger.info("Neural Style Transfer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_model_sync(self):
        """Synchronous model loading."""
        import torchvision.models as models
        
        # Load pre-trained VGG19
        vgg = models.vgg19(weights='VGG19_Weights.DEFAULT').features.to(self.device).eval()
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad_(False)
            
        self.model = vgg
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                               std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.ToPILImage()
        ])
    
    async def transfer_style(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        style_strength: float = 0.8,
        num_inference_steps: int = 20,
        quality: str = "high"
    ) -> Image.Image:
        """Apply neural style transfer."""
        if not self.loaded:
            await self.load_model()
            
        try:
            # Run style transfer in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._style_transfer_sync,
                content_image, 
                style_image, 
                style_strength, 
                num_inference_steps,
                quality
            )
            return result
        except Exception as e:
            logger.error(f"Style transfer failed: {e}")
            # Return original content image as fallback
            return content_image.copy()
    
    def _style_transfer_sync(
        self, 
        content_image: Image.Image, 
        style_image: Image.Image,
        style_strength: float,
        num_inference_steps: int,
        quality: str
    ) -> Image.Image:
        """Synchronous style transfer implementation."""
        
        # Adjust image size based on quality
        size_map = {"low": 256, "medium": 512, "high": 1024}
        target_size = size_map.get(quality, 512)
        
        # Prepare images
        content_tensor = self._preprocess_image(content_image, target_size)
        style_tensor = self._preprocess_image(style_image, target_size)
        
        # Initialize output as content image
        output = content_tensor.clone().requires_grad_(True)
        
        # Define optimizer with proper LBFGS settings
        optimizer = torch.optim.LBFGS([output], lr=0.1, max_iter=1, tolerance_grad=1e-7)
        
        # Get style and content features
        content_features = self._get_features(content_tensor)
        style_features = self._get_features(style_tensor)
        style_grams = {layer: self._gram_matrix(features) for layer, features in style_features.items()}
        
        # Style and content weights - much stronger style for dramatic effect
        style_weight = style_strength * 5e4  # Much stronger
        content_weight = (1 - style_strength) * 1  # Allow more style freedom
        tv_weight = 5  # Total variation weight for denoising
        
        def closure():
            optimizer.zero_grad()
            output_features = self._get_features(output)
            
            # Content loss
            content_loss = torch.mean((output_features['conv4_2'] - content_features['conv4_2']) ** 2)
            
            # Style loss
            style_loss = 0
            style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
            layer_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
            for layer, weight in zip(style_layers, layer_weights):
                output_gram = self._gram_matrix(output_features[layer])
                style_gram = style_grams[layer]
                layer_style_loss = torch.mean((output_gram - style_gram) ** 2)
                # style_loss += weight * layer_style_loss / len(style_layers)
                style_loss += weight * layer_style_loss
            
            # Total variation loss (denoising)
            tv_loss = torch.mean((output[:, :, :, 1:] - output[:, :, :, :-1])**2) + \
                      torch.mean((output[:, :, 1:, :] - output[:, :, :-1, :])**2)
            
            # Combine losses
            total_loss = (
                content_weight * content_loss +
                style_weight * style_loss +
                tv_weight * tv_loss
            )
            total_loss.backward()
            return total_loss
        
        # Optimize with more iterations for better results
        for i in range(min(num_inference_steps * 8, 400)):  # More iterations
            loss = optimizer.step(closure)
            with torch.no_grad():
                output.clamp_(-1.5, 1.5)   # Clamp pixels to prevent dark picels that would translate to noise

            
            # Log loss terms and diagnostics periodically
            if i % 50 == 0:  # Log every 50 iterations
                with torch.no_grad():
                    # Check pixel value ranges for normalization issues
                    output_min = output.min().item()
                    output_max = output.max().item()
                    output_mean = output.mean().item()
                    
                    # Check for extreme values that could cause artifacts
                    extreme_low = torch.sum(output < -3.0).item()  # Very negative values
                    extreme_high = torch.sum(output > 3.0).item()  # Very positive values
                    
                    output_features = self._get_features(output)
                    content_loss = torch.mean((output_features['conv4_2'] - content_features['conv4_2']) ** 2)
                    
                    # Check ReLU saturation in feature maps
                    relu_zeros = {}
                    total_activations = {}
                    for layer_name, features in output_features.items():
                        if layer_name in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
                            total_pixels = features.numel()
                            zero_pixels = torch.sum(features == 0).item()
                            relu_zeros[layer_name] = (zero_pixels / total_pixels) * 100
                            total_activations[layer_name] = total_pixels
                    
                    style_loss = 0
                    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
                    layer_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
                    for layer, weight in zip(style_layers, layer_weights):
                        output_gram = self._gram_matrix(output_features[layer])
                        style_gram = style_grams[layer]
                        layer_style_loss = torch.mean((output_gram - style_gram) ** 2)
                        style_loss += weight * layer_style_loss
                    
                    tv_loss = torch.mean((output[:, :, :, 1:] - output[:, :, :, :-1])**2) + \
                              torch.mean((output[:, :, 1:, :] - output[:, :, :-1, :])**2)

                    content_term = content_weight * content_loss
                    style_term = style_weight * style_loss
                    tv_term = tv_weight * tv_loss
                    
                    print(f"Step {i:3d}: content_term={content_term.item():.6f}, style_term={style_term.item():.6f}, tv_term={tv_term.item():.6f}, total={loss.item():.6f}")
                    print(f"         Pixel range: [{output_min:.3f}, {output_max:.3f}] mean={output_mean:.3f}")
                    print(f"         Extreme values: {extreme_low} low (<-3.0), {extreme_high} high (>3.0)")
                    
                    # Log ReLU saturation percentages
                    relu_info = ", ".join([f"{layer}={pct:.1f}%" for layer, pct in relu_zeros.items()])
                    print(f"         ReLU zeros: {relu_info}")
                    
                    # Warn about potential issues
                    if extreme_low > 1000 or extreme_high > 1000:
                        print(f"         WARNING: Many extreme pixel values detected - may cause artifacts")
                    
                    max_relu_saturation = max(relu_zeros.values()) if relu_zeros else 0
                    if max_relu_saturation > 80:
                        print(f"         WARNING: High ReLU saturation ({max_relu_saturation:.1f}%) - may cause dark artifacts")
                    
                    import time
                    time.sleep(0.001)  # Small delay to yield control
        
        # Log final pixel statistics before postprocessing
        with torch.no_grad():
            final_min = output.min().item()
            final_max = output.max().item()
            final_mean = output.mean().item()
            final_std = output.std().item()
            
            # Count extreme values
            extreme_low_final = torch.sum(output < -3.0).item()
            extreme_high_final = torch.sum(output > 3.0).item()
            
            print(f"\nFinal optimization stats:")
            print(f"  Pixel range: [{final_min:.3f}, {final_max:.3f}] mean={final_mean:.3f} std={final_std:.3f}")
            print(f"  Extreme values: {extreme_low_final} low, {extreme_high_final} high")
            
            # Apply gentle clamping to prevent artifacts while preserving dynamic range
            if extreme_low_final > 0 or extreme_high_final > 0:
                print(f"  Applying gentle clamping to prevent artifacts...")
                output.data = torch.clamp(output.data, -2.5, 2.5)  # Gentle clamp within ~6 std devs
                
                clamped_min = output.min().item()
                clamped_max = output.max().item()
                print(f"  After clamping: [{clamped_min:.3f}, {clamped_max:.3f}]")
        
        # Convert back to PIL Image
        output_image = self._postprocess_image(output)
        return output_image
    
    def _preprocess_image(self, image: Image.Image, size: int) -> torch.Tensor:
        """Preprocess PIL image to tensor."""
        # Resize maintaining aspect ratio
        image = image.convert('RGB')
        # image = image.resize((size, size), Image.Resampling.LANCZOS)
        npimage=np.array(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        return tensor
    
    def _postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL image."""
        tensor = tensor.cpu().squeeze(0)
        
        # Log tensor statistics before inverse transform
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        tensor_mean = tensor.mean().item()
        print(f"\nPostprocess input tensor: range=[{tensor_min:.3f}, {tensor_max:.3f}] mean={tensor_mean:.3f}")
        
        # Don't clamp here - let the inverse transform handle denormalization
        image = self.inverse_transform(tensor)
        
        # Convert to numpy to check final pixel values
        img_array = np.array(image)
        print(f"Final PIL image: range=[{img_array.min()}, {img_array.max()}] shape={img_array.shape}")
        
        # Check for any unusual pixel patterns
        if img_array.min() < 0 or img_array.max() > 255:
            print(f"WARNING: PIL image has out-of-range values!")
            
        return image
    
    def _get_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from VGG model."""
        features = {}
        layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                      'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        
        x = image
        for i, layer in enumerate(self.model):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                layer_name = layer_names[len(features)]
                features[layer_name] = x
                
        return features
    
    def _gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        """Calculate Gram matrix for style representation."""
        batch_size, channels, height, width = tensor.size()
        features = tensor.view(batch_size * channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * channels * height * width)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": "Neural Style Transfer",
            "type": "VGG19-based",
            "device": str(self.device),
            "loaded": self.loaded,
            "supported_formats": ["RGB", "RGBA"],
            "max_resolution": 1024,
            "capabilities": {
                "style_strength_control": True,
                "quality_settings": True,
                "batch_processing": False,
                "real_time": False
            }
        }


class MockStyleTransfer(StyleTransferModel):
    """Mock implementation for testing purposes."""
    
    async def load_model(self) -> None:
        """Mock model loading."""
        await asyncio.sleep(0.1)  # Simulate loading time
        logger.info("Mock style transfer model loaded")
    
    async def transfer_style(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        style_strength: float = 0.8,
        num_inference_steps: int = 20,
        quality: str = "high"
    ) -> Image.Image:
        """Mock style transfer - returns slightly modified content image."""
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Create a simple "stylized" version by adjusting colors
        img_array = np.array(content_image.convert('RGB'))
        
        # Apply some color adjustment to simulate style transfer
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + style_strength * 0.3), 0, 255)  # Increase saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + style_strength * 0.1), 0, 255)  # Increase brightness
        
        stylized = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        result_image = Image.fromarray(stylized.astype(np.uint8))
        
        logger.info(f"Mock style transfer completed with strength {style_strength}")
        return result_image
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "name": "Mock Style Transfer",
            "type": "Testing Mock",
            "device": "cpu",
            "loaded": True,
            "supported_formats": ["RGB", "RGBA"],
            "max_resolution": 4096,
            "capabilities": {
                "style_strength_control": True,
                "quality_settings": True,
                "batch_processing": True,
                "real_time": True
            }
        }


def get_model(model_type: str = "neural") -> StyleTransferModel:
    """Factory function to get style transfer model."""
    models = {
        "neural": NeuralStyleTransfer,
        "mock": MockStyleTransfer,
    }
    
    model_class = models.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return model_class()