"""
Model utilities and helper functions for style transfer.
"""

import torch
import hashlib
import io
from PIL import Image
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def compute_image_hash(image: Union[torch.Tensor, Image.Image]) -> str:
    """
    Compute a hash of an image for caching purposes.
    
    Args:
        image: PIL Image or torch tensor
        
    Returns:
        SHA256 hash string
    """
    if isinstance(image, torch.Tensor):
        # Convert tensor to bytes
        image_bytes = image.cpu().numpy().tobytes()
    else:
        # Convert PIL to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
    
    return hashlib.sha256(image_bytes).hexdigest()


def validate_model_inputs(
    subject_image: torch.Tensor,
    style_image: torch.Tensor,
    preset: str,
    available_presets: list
) -> bool:
    """
    Validate model inputs.
    
    Args:
        subject_image: Subject image tensor
        style_image: Style image tensor  
        preset: Quality preset name
        available_presets: List of valid presets
        
    Returns:
        True if valid, raises ValueError if not
    """
    
    # Check tensor shapes
    if subject_image.dim() != 4 or subject_image.shape[1] != 3:
        raise ValueError(f"Subject image must be [B, 3, H, W], got {subject_image.shape}")
    
    if style_image.dim() != 4 or style_image.shape[1] != 3:
        raise ValueError(f"Style image must be [B, 3, H, W], got {style_image.shape}")
    
    # Check batch size compatibility
    if style_image.shape[0] not in [1, subject_image.shape[0]]:
        raise ValueError(
            f"Style batch size must be 1 or match subject batch size. "
            f"Got style: {style_image.shape[0]}, subject: {subject_image.shape[0]}"
        )
    
    # Check preset
    if preset not in available_presets:
        raise ValueError(f"Invalid preset '{preset}'. Available: {available_presets}")
    
    # Check value ranges (expecting [-1, 1] normalized tensors)
    if subject_image.min() < -1.1 or subject_image.max() > 1.1:
        logger.warning(f"Subject image values outside expected range [-1, 1]: "
                      f"min={subject_image.min():.3f}, max={subject_image.max():.3f}")
    
    if style_image.min() < -1.1 or style_image.max() > 1.1:
        logger.warning(f"Style image values outside expected range [-1, 1]: "
                      f"min={style_image.min():.3f}, max={style_image.max():.3f}")
    
    return True


def estimate_inference_time(
    batch_size: int,
    resolution: Tuple[int, int],
    preset: str,
    device: str = "cuda"
) -> float:
    """
    Estimate inference time based on input parameters.
    
    Args:
        batch_size: Number of images in batch
        resolution: Image resolution (H, W)
        preset: Quality preset
        device: Device type
        
    Returns:
        Estimated time in seconds
    """
    
    # Base times per image (GPU estimates for A100)
    base_times = {
        "fast": 2.0,
        "balanced": 5.0, 
        "high-quality": 12.0
    }
    
    if preset not in base_times:
        preset = "balanced"
    
    base_time = base_times[preset]
    
    # Resolution scaling (square relationship)
    resolution_factor = (resolution[0] * resolution[1]) / (512 * 512)
    
    # Batch efficiency (diminishing returns)
    batch_factor = batch_size ** 0.8
    
    # Device scaling
    device_factor = 3.0 if device == "cpu" else 1.0
    
    estimated_time = base_time * resolution_factor * batch_factor * device_factor
    
    return estimated_time


def optimize_batch_size(
    target_latency: float,
    resolution: Tuple[int, int],
    preset: str,
    max_batch_size: int = 8,
    device: str = "cuda"
) -> int:
    """
    Find optimal batch size for target latency.
    
    Args:
        target_latency: Target latency in seconds
        resolution: Image resolution
        preset: Quality preset
        max_batch_size: Maximum allowed batch size
        device: Device type
        
    Returns:
        Optimal batch size
    """
    
    best_batch_size = 1
    
    for batch_size in range(1, max_batch_size + 1):
        estimated_time = estimate_inference_time(batch_size, resolution, preset, device)
        per_image_time = estimated_time / batch_size
        
        if per_image_time <= target_latency:
            best_batch_size = batch_size
        else:
            break
    
    return best_batch_size


class ModelProfiler:
    """Profile model performance and resource usage."""
    
    def __init__(self):
        self.metrics = {
            "total_inferences": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "memory_peak": 0.0,
            "preset_counts": {}
        }
    
    def start_inference(self):
        """Start timing an inference."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = torch.cuda.Event(enable_timing=True)
        self.end_time = torch.cuda.Event(enable_timing=True) 
        self.start_time.record()
    
    def end_inference(self, preset: str):
        """End timing an inference and update metrics."""
        self.end_time.record()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        inference_time = self.start_time.elapsed_time(self.end_time) / 1000.0  # Convert to seconds
        
        self.metrics["total_inferences"] += 1
        self.metrics["total_time"] += inference_time
        self.metrics["avg_time"] = self.metrics["total_time"] / self.metrics["total_inferences"]
        
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            self.metrics["memory_peak"] = max(self.metrics["memory_peak"], current_memory)
        
        self.metrics["preset_counts"][preset] = self.metrics["preset_counts"].get(preset, 0) + 1
    
    def get_stats(self) -> dict:
        """Get profiling statistics."""
        return self.metrics.copy()
    
    def reset(self):
        """Reset profiling metrics."""
        self.metrics = {
            "total_inferences": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "memory_peak": 0.0,
            "preset_counts": {}
        }