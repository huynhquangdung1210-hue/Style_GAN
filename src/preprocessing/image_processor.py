"""
Image Processing Pipeline

Robust preprocessing pipeline with validation, format conversion,
resizing, normalization, and cloud storage integration.
"""

import asyncio
import aiohttp
import aiofiles
import numpy as np
from PIL import Image, ImageOps, ExifTags
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import io
import hashlib
from typing import Tuple, Optional, Union, List
import structlog
from pathlib import Path
import tempfile
import mimetypes
from urllib.parse import urlparse

from ..utils.storage import get_storage_client, StorageError

logger = structlog.get_logger()


class ImageValidationError(Exception):
    """Custom exception for image validation errors."""
    pass


class ImageProcessor:
    """
    Production-ready image processing pipeline.
    
    Features:
    - Multi-format support (JPEG, PNG, WebP, TIFF)
    - Automatic EXIF orientation correction
    - Smart resizing with aspect ratio preservation
    - Robust validation and safety checks
    - Cloud storage integration (S3)
    - Memory-efficient processing
    - Async operations
    """
    
    def __init__(
        self,
        max_file_size: int = 50 * 1024 * 1024,  # 50MB
        min_resolution: Tuple[int, int] = (256, 256),
        max_resolution: Tuple[int, int] = (4096, 4096),
        supported_formats: List[str] = None,
        quality_threshold: float = 0.5
    ):
        self.max_file_size = max_file_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.quality_threshold = quality_threshold
        
        self.supported_formats = supported_formats or [
            'jpeg', 'jpg', 'png', 'webp', 'tiff', 'bmp'
        ]
        
        # Initialize transforms
        self.transforms = {
            'preprocess': A.Compose([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ]),
            'augment': A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            ])
        }
        
        # Storage client (lazy initialization)
        self._storage_client = None
    
    @property
    def storage_client(self):
        """Get storage client with lazy initialization."""
        if self._storage_client is None:
            self._storage_client = get_storage_client()
        return self._storage_client
    
    async def download_and_preprocess(
        self,
        image_uri: str,
        target_size: Optional[Tuple[int, int]] = None,
        apply_augmentation: bool = False,
        return_format: str = 'numpy'
    ) -> Union[np.ndarray, torch.Tensor, Image.Image]:
        """
        Download image from URI and preprocess it.
        
        Args:
            image_uri: Storage URI (minio://bucket/key, s3://bucket/key), HTTP URL, or local file path
            target_size: Target resolution (width, height)
            apply_augmentation: Whether to apply data augmentation
            return_format: 'numpy', 'tensor', or 'pil'
            
        Returns:
            Processed image in requested format
        """
        
        logger.info("Downloading and preprocessing image", uri=image_uri)
        
        try:
            # Download image
            image_data = await self._download_image(image_uri)
            
            # Validate and load image
            image = await self._validate_and_load_image(image_data)
            
            # Preprocess image
            processed_image = await self._preprocess_image(
                image,
                target_size=target_size,
                apply_augmentation=apply_augmentation
            )
            
            # Convert to requested format
            return await self._convert_format(processed_image, return_format)
            
        except Exception as e:
            logger.error("Image preprocessing failed", uri=image_uri, error=str(e))
            raise ImageValidationError(f"Failed to process image: {str(e)}")
    
    async def _download_image(self, uri: str) -> bytes:
        """Download image from various sources."""
        
        parsed = urlparse(uri)
        
        if parsed.scheme in ['s3', 'minio']:
            # Storage URI: s3://bucket/key or minio://bucket/key
            key = parsed.path.lstrip('/')
            if parsed.netloc:  # bucket in netloc
                key = f"{parsed.netloc}/{key}"
            return await self._download_from_storage(key)
            
        elif parsed.scheme in ['http', 'https']:
            # HTTP URL
            return await self._download_from_url(uri)
            
        elif parsed.scheme in ['file', ''] and Path(uri).exists():
            # Local file
            return await self._read_local_file(uri)
            
        else:
            raise ImageValidationError(f"Unsupported URI scheme: {uri}")
    
    async def _download_from_storage(self, object_name: str) -> bytes:
        """Download image from configured storage backend."""
        
        try:
            # Use storage client to download
            data = await self.storage_client.download_file(object_name)
            
            if isinstance(data, str):
                # If storage returned a local path, read the file
                async with aiofiles.open(data, 'rb') as f:
                    return await f.read()
            else:
                # Storage returned bytes directly
                return data
                
        except StorageError as e:
            logger.error("Storage download failed", object_name=object_name, error=str(e))
            raise ImageValidationError(f"Failed to download from storage: {str(e)}")
        except Exception as e:
            logger.error("Unexpected download error", object_name=object_name, error=str(e))
            raise ImageValidationError(f"Download error: {str(e)}")

    
    async def _download_from_url(self, url: str) -> bytes:
        """Download image from HTTP URL."""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        raise ImageValidationError(f"HTTP {response.status}: {url}")
                    
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_file_size:
                        raise ImageValidationError(f"File too large: {content_length} bytes")
                    
                    data = await response.read()
                    
                    if len(data) > self.max_file_size:
                        raise ImageValidationError(f"Downloaded file too large: {len(data)} bytes")
                    
                    return data
                    
        except aiohttp.ClientError as e:
            logger.error("URL download failed", url=url, error=str(e))
            raise ImageValidationError(f"Failed to download from URL: {str(e)}")
    
    async def _read_local_file(self, path: str) -> bytes:
        """Read local image file."""
        
        try:
            file_path = Path(path)
            
            if file_path.stat().st_size > self.max_file_size:
                raise ImageValidationError(f"File too large: {file_path.stat().st_size} bytes")
            
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
                
        except Exception as e:
            logger.error("Local file read failed", path=path, error=str(e))
            raise ImageValidationError(f"Failed to read local file: {str(e)}")
    
    async def _validate_and_load_image(self, image_data: bytes) -> Image.Image:
        """Validate and load image data."""
        
        # Basic validation
        if len(image_data) == 0:
            raise ImageValidationError("Empty image data")
        
        if len(image_data) > self.max_file_size:
            raise ImageValidationError(f"Image too large: {len(image_data)} bytes")
        
        try:
            # Load image with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Verify image format
            if image.format.lower() not in [fmt.upper() for fmt in self.supported_formats]:
                raise ImageValidationError(f"Unsupported format: {image.format}")
            
            # Check resolution
            width, height = image.size
            
            if width < self.min_resolution[0] or height < self.min_resolution[1]:
                raise ImageValidationError(
                    f"Resolution too small: {width}x{height}, minimum: {self.min_resolution}"
                )
            
            if width > self.max_resolution[0] or height > self.max_resolution[1]:
                raise ImageValidationError(
                    f"Resolution too large: {width}x{height}, maximum: {self.max_resolution}"
                )
            
            # Fix EXIF orientation
            image = self._fix_exif_orientation(image)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Quality check (detect very low quality images)
            quality_score = self._estimate_image_quality(image)
            if quality_score < self.quality_threshold:
                logger.warning("Low quality image detected", quality_score=quality_score)
            
            logger.info("Image validated", size=image.size, mode=image.mode, format=image.format)
            return image
            
        except Exception as e:
            logger.error("Image validation failed", error=str(e))
            raise ImageValidationError(f"Invalid image: {str(e)}")
    
    def _fix_exif_orientation(self, image: Image.Image) -> Image.Image:
        """Fix image orientation based on EXIF data."""
        
        try:
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif:
                    orientation_key = None
                    for key, value in ExifTags.TAGS.items():
                        if value == 'Orientation':
                            orientation_key = key
                            break
                    
                    if orientation_key and orientation_key in exif:
                        orientation = exif[orientation_key]
                        
                        if orientation == 2:
                            image = image.transpose(Image.FLIP_LEFT_RIGHT)
                        elif orientation == 3:
                            image = image.rotate(180)
                        elif orientation == 4:
                            image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
                        elif orientation == 5:
                            image = image.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
                        elif orientation == 6:
                            image = image.rotate(-90)
                        elif orientation == 7:
                            image = image.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
                        elif orientation == 8:
                            image = image.rotate(90)
        except Exception:
            # If EXIF processing fails, just return original image
            pass
        
        return image
    
    def _estimate_image_quality(self, image: Image.Image) -> float:
        """Estimate image quality using Laplacian variance (blur detection)."""
        
        try:
            # Convert to grayscale
            gray = np.array(image.convert('L'))
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range (empirically determined thresholds)
            quality_score = min(laplacian_var / 1000.0, 1.0)
            
            return quality_score
            
        except Exception:
            return 1.0  # Assume good quality if quality check fails
    
    async def _preprocess_image(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
        apply_augmentation: bool = False
    ) -> np.ndarray:
        """Preprocess image with resizing, normalization, and optional augmentation."""
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Resize if target size specified
        if target_size:
            image_array = await self._smart_resize(image_array, target_size)
        
        # Apply augmentation if requested
        if apply_augmentation:
            augmented = self.transforms['augment'](image=image_array)
            image_array = augmented['image']
        
        print("First image pixel range:", image_array.min(), image_array.max())
        # Apply preprocessing transforms
        preprocessed = self.transforms['preprocess'](image=image_array)
        print("Second image pixel range after preprocessing:", preprocessed['image'].min(), preprocessed['image'].max())

        return preprocessed['image']
    
    async def _smart_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Smart resizing that preserves aspect ratio and quality."""
        
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate aspect ratios
        current_aspect = w / h
        target_aspect = target_w / target_h
        
        if current_aspect > target_aspect:
            # Image is wider, fit by width
            new_w = target_w
            new_h = int(target_w / current_aspect)
        else:
            # Image is taller, fit by height
            new_h = target_h
            new_w = int(target_h * current_aspect)
        
        # Resize using high-quality interpolation
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Pad to exact target size if needed
        if (new_w, new_h) != target_size:
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            
            # Center padding
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            
            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        
        return resized
    
    async def _convert_format(self, image: torch.Tensor, format_type: str) -> Union[np.ndarray, torch.Tensor, Image.Image]:
        """Convert processed tensor to requested format."""
        
        if format_type == 'tensor':
            return image
        
        elif format_type == 'numpy':
            # Convert tensor to numpy (C, H, W) -> (H, W, C)
            if isinstance(image, torch.Tensor):
                numpy_image = image.permute(1, 2, 0).numpy()
                # Denormalize from [-1, 1] to [0, 1]
                numpy_image = (numpy_image + 1.0) / 2.0
                return np.clip(numpy_image, 0, 1)
            return image
        
        elif format_type == 'pil':
            if isinstance(image, torch.Tensor):
                # Convert tensor to PIL
                numpy_image = image.permute(1, 2, 0).numpy()
                numpy_image = (numpy_image + 1.0) / 2.0 * 255
                numpy_image = np.clip(numpy_image, 0, 255).astype(np.uint8)
                return Image.fromarray(numpy_image)
            elif isinstance(image, np.ndarray):
                if image.dtype == np.float32 or image.dtype == np.float64:
                    image = (image * 255).astype(np.uint8)
                return Image.fromarray(image)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    async def postprocess(self, result_tensor: torch.Tensor) -> np.ndarray:
        """Postprocess model output tensor to displayable image."""
        
        # Convert tensor to numpy
        if isinstance(result_tensor, torch.Tensor):
            # (C, H, W) -> (H, W, C)
            image = result_tensor.detach().cpu().permute(1, 2, 0).numpy()
        else:
            image = result_tensor
        
        # Denormalize from [-1, 1] to [0, 1]
        image = (image + 1.0) / 2.0
        image = np.clip(image, 0, 1)
        
        return image
    
    def compute_image_hash(self, image: Union[np.ndarray, torch.Tensor, Image.Image]) -> str:
        """Compute hash of image for caching."""
        
        if isinstance(image, torch.Tensor):
            data = image.cpu().numpy().tobytes()
        elif isinstance(image, Image.Image):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            data = img_byte_arr.getvalue()
        else:
            data = image.tobytes()
        
        return hashlib.sha256(data).hexdigest()
    
    async def upload_to_s3(
        self,
        image: Union[np.ndarray, Image.Image],
        bucket: str,
        key: str,
        quality: int = 95
    ) -> str:
        """Upload processed image to S3."""
        
        try:
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Save to bytes buffer
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG', quality=quality)
            img_bytes = img_buffer.getvalue()
            
            # Upload to S3
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=img_bytes,
                    ContentType='image/png'
                )
            )
            
            s3_uri = f"s3://{bucket}/{key}"
            logger.info("Uploaded image to S3", uri=s3_uri, size=len(img_bytes))
            
            return s3_uri
            
        except Exception as e:
            logger.error("S3 upload failed", bucket=bucket, key=key, error=str(e))
            raise


# Factory function
def create_image_processor(config: dict = None) -> ImageProcessor:
    """Create ImageProcessor with optional configuration."""
    
    config = config or {}
    return ImageProcessor(**config)


# Example usage and testing
async def main():
    """Example usage of ImageProcessor."""
    
    processor = create_image_processor()
    
    # Test with a sample image URL
    test_url = "https://example.com/test.jpg"
    
    try:
        processed = await processor.download_and_preprocess(
            test_url,
            target_size=(512, 512),
            return_format='numpy'
        )
        
        print(f"Processed image shape: {processed.shape}")
        print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
        
    except Exception as e:
        print(f"Processing failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())