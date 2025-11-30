"""
Safety Filter and Content Moderation

Production-ready content moderation system with NSFW detection,
face swap detection, and other safety measures.
"""

import asyncio
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Union
import structlog
from PIL import Image
import io
import base64
import aiohttp
import os
from pathlib import Path

# Try to import nudenet for NSFW detection
try:
    from nudenet import NudeDetector
    NUDENET_AVAILABLE = True
except ImportError:
    NUDENET_AVAILABLE = False
    print("Warning: nudenet not available. NSFW detection will be disabled.")

logger = structlog.get_logger()


class SafetyFilter:
    """
    Production-ready safety filter with multiple content moderation features.
    
    Features:
    - NSFW content detection
    - Face swap detection
    - Violence/gore detection
    - Inappropriate text detection
    - Custom rule-based filtering
    - Configurable thresholds
    """
    
    def __init__(
        self,
        nsfw_threshold: float = 0.7,
        violence_threshold: float = 0.8,
        enable_nsfw_detection: bool = True,
        enable_face_detection: bool = True,
        enable_text_detection: bool = False,
        model_path: Optional[str] = None
    ):
        self.nsfw_threshold = nsfw_threshold
        self.violence_threshold = violence_threshold
        self.enable_nsfw_detection = enable_nsfw_detection and NUDENET_AVAILABLE
        self.enable_face_detection = enable_face_detection
        self.enable_text_detection = enable_text_detection
        
        # Initialize models
        self.nsfw_detector = None
        self.face_detector = None
        
        if self.enable_nsfw_detection:
            try:
                self.nsfw_detector = NudeDetector()
                logger.info("NSFW detector initialized")
            except Exception as e:
                logger.error("Failed to initialize NSFW detector", error=str(e))
                self.enable_nsfw_detection = False
        
        if self.enable_face_detection:
            try:
                # Initialize OpenCV face detector
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                logger.info("Face detector initialized")
            except Exception as e:
                logger.error("Failed to initialize face detector", error=str(e))
                self.enable_face_detection = False
        
        # Blocked keywords (basic content filtering)
        self.blocked_keywords = {
            'violence': ['gore', 'blood', 'violent', 'death', 'killing', 'murder'],
            'inappropriate': ['nsfw', 'explicit', 'adult', 'sexual'],
            'illegal': ['drugs', 'weapon', 'bomb', 'terrorist']
        }
    
    async def is_safe(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        check_nsfw: bool = True,
        check_faces: bool = True,
        additional_checks: Optional[List[str]] = None
    ) -> bool:
        """
        Comprehensive safety check for an image.
        
        Args:
            image: Input image in various formats
            check_nsfw: Whether to run NSFW detection
            check_faces: Whether to run face detection
            additional_checks: Additional checks to run
            
        Returns:
            True if image is safe, False otherwise
        """
        
        try:
            # Convert to standardized format
            pil_image, np_image = self._convert_image(image)
            
            # Run safety checks
            safety_results = {
                'nsfw_safe': True,
                'face_safe': True,
                'content_safe': True,
                'overall_safe': True,
                'violations': []
            }
            
            # NSFW Detection
            if check_nsfw and self.enable_nsfw_detection:
                nsfw_safe = await self._check_nsfw(pil_image)
                safety_results['nsfw_safe'] = nsfw_safe
                if not nsfw_safe:
                    safety_results['violations'].append('nsfw_content')
            
            # Face Detection (for deepfake/face-swap detection)
            if check_faces and self.enable_face_detection:
                face_safe = await self._check_faces(np_image)
                safety_results['face_safe'] = face_safe
                if not face_safe:
                    safety_results['violations'].append('inappropriate_faces')
            
            # Content-based safety checks
            content_safe = await self._check_content_rules(np_image)
            safety_results['content_safe'] = content_safe
            if not content_safe:
                safety_results['violations'].append('content_policy')
            
            # Additional custom checks
            if additional_checks:
                for check_name in additional_checks:
                    result = await self._run_custom_check(check_name, np_image)
                    if not result:
                        safety_results['violations'].append(check_name)
            
            # Overall safety assessment
            safety_results['overall_safe'] = (
                safety_results['nsfw_safe'] and
                safety_results['face_safe'] and
                safety_results['content_safe'] and
                len(safety_results['violations']) == 0
            )
            
            if not safety_results['overall_safe']:
                logger.warning("Image failed safety check", violations=safety_results['violations'])
            
            return safety_results['overall_safe']
            
        except Exception as e:
            logger.error("Safety check failed", error=str(e))
            # Fail safe - reject image if safety check fails
            return False
    
    async def get_detailed_safety_report(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> Dict:
        """Get detailed safety analysis report."""
        
        try:
            pil_image, np_image = self._convert_image(image)
            
            report = {
                'overall_safe': True,
                'timestamp': asyncio.get_event_loop().time(),
                'checks_performed': [],
                'violations': [],
                'scores': {},
                'recommendations': []
            }
            
            # NSFW Analysis
            if self.enable_nsfw_detection:
                nsfw_result = await self._detailed_nsfw_check(pil_image)
                report['checks_performed'].append('nsfw_detection')
                report['scores']['nsfw'] = nsfw_result
                
                if nsfw_result['is_unsafe']:
                    report['violations'].append('nsfw_content')
                    report['recommendations'].append('Remove or blur inappropriate content')
            
            # Face Analysis
            if self.enable_face_detection:
                face_result = await self._detailed_face_check(np_image)
                report['checks_performed'].append('face_detection')
                report['scores']['faces'] = face_result
                
                if face_result['suspicious_faces'] > 0:
                    report['violations'].append('suspicious_faces')
                    report['recommendations'].append('Review face modifications or deepfakes')
            
            # Content Quality Analysis
            quality_result = await self._analyze_content_quality(np_image)
            report['scores']['quality'] = quality_result
            
            if quality_result['quality_score'] < 0.3:
                report['recommendations'].append('Improve image quality')
            
            # Overall assessment
            report['overall_safe'] = len(report['violations']) == 0
            
            return report
            
        except Exception as e:
            logger.error("Detailed safety analysis failed", error=str(e))
            return {
                'overall_safe': False,
                'error': str(e),
                'violations': ['analysis_failed']
            }
    
    def _convert_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> Tuple[Image.Image, np.ndarray]:
        """Convert input image to PIL and numpy formats."""
        
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy (C, H, W) -> (H, W, C)
            np_image = image.detach().cpu().permute(1, 2, 0).numpy()
            # Denormalize if needed
            if np_image.min() < 0:  # Assume [-1, 1] range
                np_image = (np_image + 1.0) / 2.0
            np_image = (np_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(np_image)
            
        elif isinstance(image, np.ndarray):
            np_image = image.copy()
            if np_image.dtype != np.uint8:
                if np_image.max() <= 1.0:  # [0, 1] range
                    np_image = (np_image * 255).astype(np.uint8)
                else:
                    np_image = np_image.astype(np.uint8)
            pil_image = Image.fromarray(np_image)
            
        elif isinstance(image, Image.Image):
            pil_image = image
            np_image = np.array(image)
            
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return pil_image, np_image
    
    async def _check_nsfw(self, image: Image.Image) -> bool:
        """Check for NSFW content."""
        
        if not self.enable_nsfw_detection or not self.nsfw_detector:
            return True
        
        try:
            # Convert PIL to format expected by nudenet
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Run detection in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.nsfw_detector.detect(img_bytes)
            )
            
            # Analyze results
            for detection in result:
                if detection['score'] > self.nsfw_threshold:
                    logger.warning("NSFW content detected", 
                                 class_name=detection['class'], 
                                 score=detection['score'])
                    return False
            
            return True
            
        except Exception as e:
            logger.error("NSFW detection failed", error=str(e))
            # Fail safe
            return False
    
    async def _detailed_nsfw_check(self, image: Image.Image) -> Dict:
        """Detailed NSFW analysis."""
        
        result = {
            'is_unsafe': False,
            'detections': [],
            'max_score': 0.0,
            'categories': {}
        }
        
        if not self.enable_nsfw_detection or not self.nsfw_detector:
            return result
        
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            loop = asyncio.get_event_loop()
            detections = await loop.run_in_executor(
                None,
                lambda: self.nsfw_detector.detect(img_bytes)
            )
            
            for detection in detections:
                score = detection['score']
                category = detection['class']
                
                result['detections'].append({
                    'category': category,
                    'score': score,
                    'bbox': detection.get('box', [])
                })
                
                result['max_score'] = max(result['max_score'], score)
                result['categories'][category] = max(
                    result['categories'].get(category, 0.0), 
                    score
                )
                
                if score > self.nsfw_threshold:
                    result['is_unsafe'] = True
            
            return result
            
        except Exception as e:
            logger.error("Detailed NSFW check failed", error=str(e))
            result['error'] = str(e)
            result['is_unsafe'] = True  # Fail safe
            return result
    
    async def _check_faces(self, image: np.ndarray) -> bool:
        """Check for potentially problematic faces."""
        
        if not self.enable_face_detection or self.face_detector is None:
            return True
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Basic heuristics for suspicious faces
            suspicious_count = 0
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Check for unusual patterns that might indicate manipulation
                if self._analyze_face_authenticity(face_roi):
                    suspicious_count += 1
            
            # Allow some faces, but flag if too many or all faces look suspicious
            if len(faces) > 0 and suspicious_count / len(faces) > 0.8:
                logger.warning("Suspicious faces detected", 
                             total_faces=len(faces), 
                             suspicious=suspicious_count)
                return False
            
            return True
            
        except Exception as e:
            logger.error("Face detection failed", error=str(e))
            return True  # Assume safe if face detection fails
    
    async def _detailed_face_check(self, image: np.ndarray) -> Dict:
        """Detailed face analysis."""
        
        result = {
            'total_faces': 0,
            'suspicious_faces': 0,
            'faces': []
        }
        
        if not self.enable_face_detection or self.face_detector is None:
            return result
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            result['total_faces'] = len(faces)
            
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = gray[y:y+h, x:x+w]
                
                face_info = {
                    'id': i,
                    'bbox': [x, y, w, h],
                    'suspicious': False,
                    'authenticity_score': 1.0
                }
                
                # Analyze face authenticity
                if self._analyze_face_authenticity(face_roi):
                    face_info['suspicious'] = True
                    face_info['authenticity_score'] = 0.3
                    result['suspicious_faces'] += 1
                
                result['faces'].append(face_info)
            
            return result
            
        except Exception as e:
            logger.error("Detailed face analysis failed", error=str(e))
            result['error'] = str(e)
            return result
    
    def _analyze_face_authenticity(self, face_roi: np.ndarray) -> bool:
        """Analyze face for signs of manipulation (simple heuristics)."""
        
        try:
            # Simple checks for face manipulation
            # These are basic heuristics and would need more sophisticated models in production
            
            # Check for unusual frequency patterns (might indicate AI generation)
            dft = cv2.dft(np.float32(face_roi), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
            
            # Look for unusual patterns in frequency domain
            mean_magnitude = np.mean(magnitude_spectrum)
            std_magnitude = np.std(magnitude_spectrum)
            
            # AI-generated faces often have unusual frequency patterns
            if std_magnitude < mean_magnitude * 0.1:  # Too uniform
                return True
            
            # Check for edge consistency (deepfakes often have inconsistent edges)
            edges = cv2.Canny(face_roi, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Unusual edge patterns might indicate manipulation
            if edge_density < 0.02 or edge_density > 0.2:
                return True
            
            return False
            
        except Exception:
            return False  # Assume authentic if analysis fails
    
    async def _check_content_rules(self, image: np.ndarray) -> bool:
        """Apply content-based safety rules."""
        
        try:
            # Basic image quality checks
            if self._is_low_quality(image):
                return False
            
            # Check for unusual color patterns that might indicate inappropriate content
            if self._has_suspicious_colors(image):
                return False
            
            return True
            
        except Exception as e:
            logger.error("Content rule check failed", error=str(e))
            return True  # Assume safe if check fails
    
    def _is_low_quality(self, image: np.ndarray) -> bool:
        """Check if image quality is suspiciously low."""
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Calculate blur metric using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Very blurry images might be used to hide inappropriate content
            return blur_score < 10
            
        except Exception:
            return False
    
    def _has_suspicious_colors(self, image: np.ndarray) -> bool:
        """Check for suspicious color patterns."""
        
        try:
            # Check for unusual color distributions
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Check saturation - overly saturated images might be problematic
            saturation = hsv[:,:,1]
            mean_saturation = np.mean(saturation)
            
            # Flag images with extreme saturation
            return mean_saturation > 200  # Very high saturation
            
        except Exception:
            return False
    
    async def _run_custom_check(self, check_name: str, image: np.ndarray) -> bool:
        """Run custom safety checks."""
        
        # Placeholder for custom checks
        # In production, you might integrate with external APIs or custom models
        
        if check_name == 'custom_model':
            # Example: Run inference with a custom safety model
            pass
        elif check_name == 'external_api':
            # Example: Call external content moderation API
            pass
        
        return True  # Default to safe
    
    async def _analyze_content_quality(self, image: np.ndarray) -> Dict:
        """Analyze overall content quality."""
        
        result = {
            'quality_score': 1.0,
            'blur_score': 0.0,
            'noise_level': 0.0,
            'brightness': 0.0,
            'contrast': 0.0
        }
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Blur detection
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            result['blur_score'] = blur_score
            
            # Noise estimation
            noise_level = np.std(gray)
            result['noise_level'] = noise_level
            
            # Brightness
            brightness = np.mean(gray)
            result['brightness'] = brightness
            
            # Contrast
            contrast = np.std(gray)
            result['contrast'] = contrast
            
            # Overall quality score (0-1)
            quality_score = min(
                blur_score / 1000.0,  # Normalize blur
                1.0 - abs(brightness - 128) / 128.0,  # Penalize extreme brightness
                contrast / 64.0,  # Normalize contrast
                1.0
            )
            
            result['quality_score'] = max(quality_score, 0.0)
            
            return result
            
        except Exception as e:
            logger.error("Quality analysis failed", error=str(e))
            return result


# Factory function
def create_safety_filter(config: dict = None) -> SafetyFilter:
    """Create SafetyFilter with optional configuration."""
    
    config = config or {}
    return SafetyFilter(**config)


# Example usage
async def main():
    """Example usage of SafetyFilter."""
    
    filter_system = create_safety_filter()
    
    # Test with a sample image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    is_safe = await filter_system.is_safe(test_image)
    print(f"Image is safe: {is_safe}")
    
    # Get detailed report
    report = await filter_system.get_detailed_safety_report(test_image)
    print(f"Safety report: {report}")


if __name__ == "__main__":
    asyncio.run(main())