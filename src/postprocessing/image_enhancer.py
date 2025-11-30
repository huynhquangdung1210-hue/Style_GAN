"""
Image Enhancement and Post-processing

Production-ready post-processing pipeline with denoising, upsampling,
color correction, and quality enhancement.
"""

import asyncio
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import skimage
from skimage import restoration, exposure, filters
from typing import Dict, List, Tuple, Optional, Union
import structlog
import concurrent.futures
from pathlib import Path

logger = structlog.get_logger()


class ImageEnhancer:
    """
    Production-ready image enhancement and post-processing pipeline.
    
    Features:
    - Noise reduction and denoising
    - Super-resolution upscaling
    - Color correction and enhancement
    - Artifact removal
    - Quality optimization
    - Batch processing support
    """
    
    def __init__(
        self,
        enable_denoising: bool = True,
        enable_upsampling: bool = True,
        enable_color_correction: bool = True,
        max_workers: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.enable_denoising = enable_denoising
        self.enable_upsampling = enable_upsampling
        self.enable_color_correction = enable_color_correction
        self.max_workers = max_workers
        self.device = device
        
        # Initialize thread pool for CPU operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Enhancement parameters
        self.enhancement_params = {
            'denoise': {
                'method': 'bilateral',  # bilateral, nlmeans, wavelet
                'sigma_color': 75,
                'sigma_space': 75,
                'kernel_size': 9
            },
            'upscale': {
                'method': 'lanczos',  # lanczos, bicubic, edsr
                'scale_factor': 2.0,
                'max_resolution': 2048
            },
            'color': {
                'gamma': 1.0,
                'brightness': 1.0,
                'contrast': 1.1,
                'saturation': 1.05,
                'auto_adjust': True
            }
        }
    
    async def enhance_image(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        enhancement_level: str = "balanced",
        custom_params: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Main image enhancement function.
        
        Args:
            image: Input image in various formats
            enhancement_level: "light", "balanced", "aggressive"
            custom_params: Override default parameters
            
        Returns:
            Enhanced image as numpy array [0, 1]
        """
        
        logger.info("Starting image enhancement", level=enhancement_level)
        
        try:
            # Convert to standardized format
            np_image = self._convert_to_numpy(image)
            
            # Apply enhancement preset
            params = self._get_enhancement_preset(enhancement_level)
            if custom_params:
                params = self._merge_params(params, custom_params)
            
            enhanced_image = np_image.copy()
            
            # Step 1: Denoising
            if self.enable_denoising:
                enhanced_image = await self._denoise_image(enhanced_image, params['denoise'])
            
            # Step 2: Color correction
            if self.enable_color_correction:
                enhanced_image = await self._correct_colors(enhanced_image, params['color'])
            
            # Step 3: Upsampling (if requested)
            if self.enable_upsampling and params.get('upscale', {}).get('enabled', False):
                enhanced_image = await self._upsample_image(enhanced_image, params['upscale'])
            
            # Step 4: Final quality adjustments
            enhanced_image = await self._final_adjustments(enhanced_image)
            
            # Ensure output is in valid range
            enhanced_image = np.clip(enhanced_image, 0, 1)
            
            logger.info("Image enhancement completed", 
                       input_shape=np_image.shape, 
                       output_shape=enhanced_image.shape)
            
            return enhanced_image
            
        except Exception as e:
            logger.error("Image enhancement failed", error=str(e))
            # Return original image if enhancement fails
            return self._convert_to_numpy(image)
    
    async def batch_enhance(
        self,
        images: List[Union[np.ndarray, torch.Tensor, Image.Image]],
        enhancement_level: str = "balanced"
    ) -> List[np.ndarray]:
        """Enhance multiple images in parallel."""
        
        logger.info("Starting batch enhancement", batch_size=len(images))
        
        # Process in parallel
        tasks = [
            self.enhance_image(img, enhancement_level) 
            for img in images
        ]
        
        enhanced_images = await asyncio.gather(*tasks)
        
        logger.info("Batch enhancement completed")
        return enhanced_images
    
    def _convert_to_numpy(self, image: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
        """Convert various image formats to numpy array [0, 1]."""
        
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy (C, H, W) -> (H, W, C)
            np_image = image.detach().cpu().permute(1, 2, 0).numpy()
            # Denormalize if needed
            if np_image.min() < 0:  # Assume [-1, 1] range
                np_image = (np_image + 1.0) / 2.0
            
        elif isinstance(image, Image.Image):
            np_image = np.array(image)
            if np_image.dtype == np.uint8:
                np_image = np_image.astype(np.float32) / 255.0
            
        elif isinstance(image, np.ndarray):
            np_image = image.copy()
            if np_image.dtype == np.uint8:
                np_image = np_image.astype(np.float32) / 255.0
            
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Ensure 3 channels
        if len(np_image.shape) == 2:
            np_image = np.stack([np_image] * 3, axis=-1)
        elif np_image.shape[-1] == 4:  # RGBA
            np_image = np_image[:, :, :3]  # Drop alpha channel
        
        return np_image
    
    def _get_enhancement_preset(self, level: str) -> Dict:
        """Get enhancement parameters for different quality levels."""
        
        if level == "light":
            return {
                'denoise': {
                    'method': 'bilateral',
                    'sigma_color': 50,
                    'sigma_space': 50,
                    'kernel_size': 5
                },
                'color': {
                    'gamma': 1.0,
                    'brightness': 1.0,
                    'contrast': 1.02,
                    'saturation': 1.01,
                    'auto_adjust': False
                },
                'upscale': {
                    'enabled': False
                }
            }
        
        elif level == "balanced":
            return {
                'denoise': {
                    'method': 'bilateral',
                    'sigma_color': 75,
                    'sigma_space': 75,
                    'kernel_size': 9
                },
                'color': {
                    'gamma': 1.0,
                    'brightness': 1.0,
                    'contrast': 1.1,
                    'saturation': 1.05,
                    'auto_adjust': True
                },
                'upscale': {
                    'enabled': False
                }
            }
        
        elif level == "aggressive":
            return {
                'denoise': {
                    'method': 'nlmeans',
                    'h': 10,
                    'template_window_size': 7,
                    'search_window_size': 21
                },
                'color': {
                    'gamma': 0.9,
                    'brightness': 1.05,
                    'contrast': 1.2,
                    'saturation': 1.1,
                    'auto_adjust': True
                },
                'upscale': {
                    'enabled': True,
                    'method': 'lanczos',
                    'scale_factor': 1.5
                }
            }
        
        else:
            return self._get_enhancement_preset("balanced")
    
    def _merge_params(self, base_params: Dict, custom_params: Dict) -> Dict:
        """Merge custom parameters with base parameters."""
        
        merged = base_params.copy()
        
        for key, value in custom_params.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
        
        return merged
    
    async def _denoise_image(self, image: np.ndarray, denoise_params: Dict) -> np.ndarray:
        """Apply denoising to the image."""
        
        method = denoise_params.get('method', 'bilateral')
        
        try:
            if method == 'bilateral':
                # Bilateral filter for edge-preserving denoising
                denoised = await self._run_in_thread(
                    self._bilateral_denoise, image, denoise_params
                )
            
            elif method == 'nlmeans':
                # Non-local means denoising
                denoised = await self._run_in_thread(
                    self._nlmeans_denoise, image, denoise_params
                )
            
            elif method == 'wavelet':
                # Wavelet denoising
                denoised = await self._run_in_thread(
                    self._wavelet_denoise, image, denoise_params
                )
            
            else:
                logger.warning("Unknown denoising method, using bilateral", method=method)
                denoised = await self._run_in_thread(
                    self._bilateral_denoise, image, denoise_params
                )
            
            return denoised
            
        except Exception as e:
            logger.error("Denoising failed", method=method, error=str(e))
            return image
    
    def _bilateral_denoise(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply bilateral filtering."""
        
        # Convert to uint8 for cv2.bilateralFilter
        img_uint8 = (image * 255).astype(np.uint8)
        
        denoised = cv2.bilateralFilter(
            img_uint8,
            params.get('kernel_size', 9),
            params.get('sigma_color', 75),
            params.get('sigma_space', 75)
        )
        
        return denoised.astype(np.float32) / 255.0
    
    def _nlmeans_denoise(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply non-local means denoising."""
        
        # Use scikit-image implementation
        denoised = restoration.denoise_nl_means(
            image,
            h=params.get('h', 0.1),
            fast_mode=True,
            patch_size=params.get('template_window_size', 7),
            patch_distance=params.get('search_window_size', 11),
            multichannel=True
        )
        
        return denoised
    
    def _wavelet_denoise(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply wavelet denoising."""
        
        # Use scikit-image wavelet denoising
        denoised = restoration.denoise_wavelet(
            image,
            multichannel=True,
            convert2ycbcr=True,
            method='BayesShrink',
            mode='soft',
            rescale_sigma=True
        )
        
        return denoised
    
    async def _correct_colors(self, image: np.ndarray, color_params: Dict) -> np.ndarray:
        """Apply color correction and enhancement."""
        
        try:
            enhanced = image.copy()
            
            # Auto-adjust levels if enabled
            if color_params.get('auto_adjust', False):
                enhanced = await self._run_in_thread(self._auto_adjust_levels, enhanced)
            
            # Apply manual adjustments
            enhanced = await self._run_in_thread(
                self._manual_color_adjustments, enhanced, color_params
            )
            
            return enhanced
            
        except Exception as e:
            logger.error("Color correction failed", error=str(e))
            return image
    
    def _auto_adjust_levels(self, image: np.ndarray) -> np.ndarray:
        """Automatically adjust image levels for better contrast."""
        
        # Adjust each channel separately
        enhanced = np.zeros_like(image)
        
        for i in range(image.shape[2]):
            channel = image[:, :, i]
            
            # Compute percentiles for contrast stretching
            p2, p98 = np.percentile(channel, (2, 98))
            
            # Stretch contrast
            enhanced[:, :, i] = exposure.rescale_intensity(
                channel, in_range=(p2, p98), out_range=(0, 1)
            )
        
        return enhanced
    
    def _manual_color_adjustments(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply manual color adjustments."""
        
        enhanced = image.copy()
        
        # Gamma correction
        gamma = params.get('gamma', 1.0)
        if gamma != 1.0:
            enhanced = np.power(enhanced, gamma)
        
        # Brightness adjustment
        brightness = params.get('brightness', 1.0)
        if brightness != 1.0:
            enhanced = enhanced * brightness
        
        # Contrast adjustment
        contrast = params.get('contrast', 1.0)
        if contrast != 1.0:
            enhanced = (enhanced - 0.5) * contrast + 0.5
        
        # Saturation adjustment (convert to HSV)
        saturation = params.get('saturation', 1.0)
        if saturation != 1.0:
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return np.clip(enhanced, 0, 1)
    
    async def _upsample_image(self, image: np.ndarray, upscale_params: Dict) -> np.ndarray:
        """Upsample image using various methods."""
        
        method = upscale_params.get('method', 'lanczos')
        scale_factor = upscale_params.get('scale_factor', 2.0)
        max_resolution = upscale_params.get('max_resolution', 2048)
        
        try:
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # Limit maximum resolution
            if new_w > max_resolution or new_h > max_resolution:
                scale = min(max_resolution / new_w, max_resolution / new_h)
                new_w = int(new_w * scale)
                new_h = int(new_h * scale)
            
            if method == 'lanczos':
                upsampled = await self._run_in_thread(
                    self._lanczos_upsample, image, (new_w, new_h)
                )
            
            elif method == 'bicubic':
                upsampled = await self._run_in_thread(
                    self._bicubic_upsample, image, (new_w, new_h)
                )
            
            else:
                logger.warning("Unknown upsampling method, using lanczos", method=method)
                upsampled = await self._run_in_thread(
                    self._lanczos_upsample, image, (new_w, new_h)
                )
            
            return upsampled
            
        except Exception as e:
            logger.error("Upsampling failed", method=method, error=str(e))
            return image
    
    def _lanczos_upsample(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Upsample using Lanczos interpolation."""
        
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        upsampled_pil = pil_image.resize(target_size, Image.LANCZOS)
        return np.array(upsampled_pil).astype(np.float32) / 255.0
    
    def _bicubic_upsample(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Upsample using bicubic interpolation."""
        
        upsampled = cv2.resize(
            image, target_size, interpolation=cv2.INTER_CUBIC
        )
        return upsampled
    
    async def _final_adjustments(self, image: np.ndarray) -> np.ndarray:
        """Apply final quality adjustments."""
        
        try:
            # Subtle sharpening
            sharpened = await self._run_in_thread(self._sharpen_image, image)
            
            # Color balance
            balanced = await self._run_in_thread(self._balance_colors, sharpened)
            
            return balanced
            
        except Exception as e:
            logger.error("Final adjustments failed", error=str(e))
            return image
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle sharpening."""
        
        # Create sharpening kernel
        kernel = np.array([[-0.5, -1, -0.5],
                          [-1,   6, -1],
                          [-0.5, -1, -0.5]]) * 0.1
        
        # Apply to each channel
        sharpened = np.zeros_like(image)
        for i in range(image.shape[2]):
            sharpened[:, :, i] = cv2.filter2D(image[:, :, i], -1, kernel)
        
        # Blend with original (subtle effect)
        return 0.8 * image + 0.2 * sharpened
    
    def _balance_colors(self, image: np.ndarray) -> np.ndarray:
        """Apply automatic color balance."""
        
        # Simple gray world assumption
        avg_r = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_b = np.mean(image[:, :, 2])
        
        avg_gray = (avg_r + avg_g + avg_b) / 3
        
        # Compute correction factors
        scale_r = avg_gray / avg_r if avg_r > 0 else 1.0
        scale_g = avg_gray / avg_g if avg_g > 0 else 1.0
        scale_b = avg_gray / avg_b if avg_b > 0 else 1.0
        
        # Limit adjustment to avoid overcorrection
        scale_r = np.clip(scale_r, 0.8, 1.2)
        scale_g = np.clip(scale_g, 0.8, 1.2)
        scale_b = np.clip(scale_b, 0.8, 1.2)
        
        # Apply correction
        balanced = image.copy()
        balanced[:, :, 0] *= scale_r
        balanced[:, :, 1] *= scale_g
        balanced[:, :, 2] *= scale_b
        
        return np.clip(balanced, 0, 1)
    
    async def _run_in_thread(self, func, *args) -> any:
        """Run CPU-intensive function in thread pool."""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)
    
    async def shutdown(self):
        """Clean up resources."""
        
        self.executor.shutdown(wait=True)
        logger.info("Image enhancer shutdown complete")


# Factory function
def create_image_enhancer(config: dict = None) -> ImageEnhancer:
    """Create ImageEnhancer with optional configuration."""
    
    config = config or {}
    return ImageEnhancer(**config)


# Example usage
async def main():
    """Example usage of ImageEnhancer."""
    
    enhancer = create_image_enhancer()
    
    # Test with a sample image
    test_image = np.random.rand(512, 512, 3).astype(np.float32)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, test_image.shape).astype(np.float32)
    noisy_image = np.clip(test_image + noise, 0, 1)
    
    # Enhance the image
    enhanced = await enhancer.enhance_image(noisy_image, enhancement_level="balanced")
    
    print(f"Original shape: {test_image.shape}")
    print(f"Enhanced shape: {enhanced.shape}")
    print(f"Original range: [{noisy_image.min():.3f}, {noisy_image.max():.3f}]")
    print(f"Enhanced range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    
    await enhancer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())