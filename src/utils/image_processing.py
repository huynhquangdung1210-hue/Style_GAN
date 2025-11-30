"""
Image processing utilities for style transfer pipeline.
Handles validation, format conversion, preprocessing, and post-processing.
"""

import asyncio
import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import base64
import hashlib
from PIL import Image, ImageOps, ExifTags
import numpy as np
import cv2
from enum import Enum

logger = logging.getLogger(__name__)

class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    JPG = "jpg" 
    PNG = "png"
    WEBP = "webp"
    AVIF = "avif"
    TIFF = "tiff"

class QualitySetting(Enum):
    """Quality presets for processing."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    ULTRA = "ultra"

class ImageProcessor:
    """Handles all image processing operations for style transfer."""
    
    def __init__(self):
        self.max_dimensions = {
            QualitySetting.LOW: (512, 512),
            QualitySetting.MEDIUM: (768, 768), 
            QualitySetting.HIGH: (1024, 1024),
            QualitySetting.ULTRA: (1536, 1536)
        }
        
        self.jpeg_quality = {
            QualitySetting.LOW: 80,
            QualitySetting.MEDIUM: 90,
            QualitySetting.HIGH: 95,
            QualitySetting.ULTRA: 98
        }
        
        # Supported input formats
        self.supported_formats = {
            'JPEG', 'JPG', 'PNG', 'WEBP', 'AVIF', 'TIFF', 'BMP'
        }
        
        logger.info("Image processor initialized")
    
    async def validate_image_data(self, image_data: bytes) -> Dict[str, Any]:
        """
        Validate image data and return metadata.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary with validation results and metadata
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._validate_image_sync, image_data
            )
            return result
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return {
                "valid": False,
                "error": str(e),
                "format": None,
                "size": None,
                "file_size": len(image_data)
            }
    
    def _validate_image_sync(self, image_data: bytes) -> Dict[str, Any]:
        """Synchronous image validation."""
        try:
            # Try to open with PIL
            with Image.open(io.BytesIO(image_data)) as img:
                # Check format
                format_name = img.format
                if format_name not in self.supported_formats:
                    return {
                        "valid": False,
                        "error": f"Unsupported format: {format_name}",
                        "format": format_name,
                        "size": img.size,
                        "file_size": len(image_data)
                    }
                
                # Check dimensions
                width, height = img.size
                if width < 64 or height < 64:
                    return {
                        "valid": False,
                        "error": "Image too small (minimum 64x64)",
                        "format": format_name,
                        "size": (width, height),
                        "file_size": len(image_data)
                    }
                
                if width > 4096 or height > 4096:
                    return {
                        "valid": False,
                        "error": "Image too large (maximum 4096x4096)",
                        "format": format_name,
                        "size": (width, height),
                        "file_size": len(image_data)
                    }
                
                # Check file size (max 20MB)
                if len(image_data) > 20 * 1024 * 1024:
                    return {
                        "valid": False,
                        "error": "File too large (maximum 20MB)",
                        "format": format_name,
                        "size": (width, height),
                        "file_size": len(image_data)
                    }
                
                return {
                    "valid": True,
                    "format": format_name,
                    "size": (width, height),
                    "mode": img.mode,
                    "file_size": len(image_data),
                    "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                }
                
        except Exception as e:
            return {
                "valid": False,
                "error": f"Invalid image data: {str(e)}",
                "format": None,
                "size": None,
                "file_size": len(image_data)
            }
    
    async def load_image_from_base64(self, base64_data: str) -> Image.Image:
        """
        Load PIL Image from base64 string.
        
        Args:
            base64_data: Base64 encoded image string
            
        Returns:
            PIL Image object
        """
        try:
            # Decode base64
            image_data = base64.b64decode(base64_data)
            
            # Validate first
            validation = await self.validate_image_data(image_data)
            if not validation["valid"]:
                raise ValueError(f"Invalid image: {validation['error']}")
            
            # Load image
            image = await asyncio.get_event_loop().run_in_executor(
                None, lambda: Image.open(io.BytesIO(image_data))
            )
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image from base64: {e}")
            raise ValueError(f"Invalid image data: {str(e)}")
    
    async def preprocess_image(
        self, 
        image: Image.Image, 
        quality: QualitySetting = QualitySetting.MEDIUM,
        maintain_aspect_ratio: bool = True
    ) -> Image.Image:
        """
        Preprocess image for style transfer.
        
        Args:
            image: Input PIL Image
            quality: Quality setting determining output size
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._preprocess_image_sync, image, quality, maintain_aspect_ratio
            )
            return result
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def _preprocess_image_sync(
        self, 
        image: Image.Image, 
        quality: QualitySetting,
        maintain_aspect_ratio: bool
    ) -> Image.Image:
        """Synchronous image preprocessing."""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # Handle transparency by compositing on white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            else:
                image = image.convert('RGB')
        
        # Fix orientation based on EXIF
        image = self._fix_image_orientation(image)
        
        # Resize based on quality setting
        max_width, max_height = self.max_dimensions[quality]
        
        if maintain_aspect_ratio:
            # Calculate new dimensions maintaining aspect ratio
            width, height = image.size
            aspect_ratio = width / height
            
            if aspect_ratio > 1:  # Landscape
                new_width = min(width, max_width)
                new_height = int(new_width / aspect_ratio)
            else:  # Portrait or square
                new_height = min(height, max_height)
                new_width = int(new_height * aspect_ratio)
            
            # Ensure neither dimension exceeds maximums
            if new_width > max_width:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            if new_height > max_height:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
        else:
            new_width, new_height = max_width, max_height
        
        # Resize image
        if image.size != (new_width, new_height):
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        logger.debug(f"Preprocessed image: {image.size}, mode: {image.mode}")
        return image
    
    def _fix_image_orientation(self, image: Image.Image) -> Image.Image:
        """Fix image orientation based on EXIF data."""
        try:
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif is not None:
                    orientation_key = None
                    for key, tag in ExifTags.TAGS.items():
                        if tag == 'Orientation':
                            orientation_key = key
                            break
                    
                    if orientation_key and orientation_key in exif:
                        orientation = exif[orientation_key]
                        
                        if orientation == 3:
                            image = image.rotate(180, expand=True)
                        elif orientation == 6:
                            image = image.rotate(270, expand=True)
                        elif orientation == 8:
                            image = image.rotate(90, expand=True)
        except Exception as e:
            logger.warning(f"Could not fix orientation: {e}")
        
        return image
    
    async def postprocess_image(
        self, 
        image: Image.Image,
        output_format: ImageFormat = ImageFormat.JPEG,
        quality: QualitySetting = QualitySetting.MEDIUM
    ) -> bytes:
        """
        Postprocess and encode image for output.
        
        Args:
            image: Processed PIL Image
            output_format: Desired output format
            quality: Quality setting for encoding
            
        Returns:
            Encoded image bytes
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._postprocess_image_sync, image, output_format, quality
            )
            return result
        except Exception as e:
            logger.error(f"Image postprocessing failed: {e}")
            raise
    
    def _postprocess_image_sync(
        self, 
        image: Image.Image,
        output_format: ImageFormat,
        quality: QualitySetting
    ) -> bytes:
        """Synchronous image postprocessing."""
        
        # Ensure RGB mode for JPEG
        if output_format in [ImageFormat.JPEG, ImageFormat.JPG] and image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create output buffer
        output_buffer = io.BytesIO()
        
        # Set encoding parameters
        save_kwargs = {}
        
        if output_format in [ImageFormat.JPEG, ImageFormat.JPG]:
            save_kwargs.update({
                'format': 'JPEG',
                'quality': self.jpeg_quality[quality],
                'optimize': True,
                'progressive': True
            })
        elif output_format == ImageFormat.PNG:
            save_kwargs.update({
                'format': 'PNG',
                'optimize': True
            })
        elif output_format == ImageFormat.WEBP:
            save_kwargs.update({
                'format': 'WEBP',
                'quality': self.jpeg_quality[quality],
                'method': 6  # Best compression
            })
        
        # Save image
        image.save(output_buffer, **save_kwargs)
        
        # Get bytes
        image_bytes = output_buffer.getvalue()
        output_buffer.close()
        
        logger.debug(f"Postprocessed image: {len(image_bytes)} bytes, format: {output_format.value}")
        return image_bytes
    
    def create_image_hash(self, image_data: bytes) -> str:
        """Create SHA-256 hash of image data for caching."""
        return hashlib.sha256(image_data).hexdigest()
    
    async def create_thumbnail(
        self, 
        image: Image.Image, 
        size: Tuple[int, int] = (256, 256)
    ) -> Image.Image:
        """Create thumbnail of image."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._create_thumbnail_sync, image, size
            )
            return result
        except Exception as e:
            logger.error(f"Thumbnail creation failed: {e}")
            raise
    
    def _create_thumbnail_sync(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Synchronous thumbnail creation."""
        # Make a copy to avoid modifying original
        thumb = image.copy()
        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        return thumb
    
    def get_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """Get comprehensive image information."""
        return {
            "size": image.size,
            "mode": image.mode,
            "format": image.format,
            "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in getattr(image, 'info', {}),
            "aspect_ratio": image.size[0] / image.size[1] if image.size[1] > 0 else 1.0
        }

# Global processor instance
_processor = None

def get_image_processor() -> ImageProcessor:
    """Get global image processor instance."""
    global _processor
    if _processor is None:
        _processor = ImageProcessor()
    return _processor