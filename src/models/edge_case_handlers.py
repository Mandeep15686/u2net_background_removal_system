
"""
Edge Case Handling Modules for U²-Net Background Removal
Team 1: The Isolationists - Specialized Subject & Background Separation

This module provides advanced processing capabilities for challenging scenarios:
- Transparency: Glass, plastic, and translucent materials
- Reflectivity: Jewelry, metallic surfaces, and shiny objects  
- Fine Details: Hair, fur, and intricate patterns
- Texture Complexity: Fabrics, woven materials, and complex surfaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional, Union
import logging

class EdgeCaseHandler:
    """Specialized handlers for challenging segmentation scenarios"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)

    def process_challenging_image(self, image: Union[str, Image.Image], 
                                mask_pred: np.ndarray) -> np.ndarray:
        """
        Main entry point for processing challenging scenarios

        Args:
            image: Input image (path or PIL Image)
            mask_pred: Initial mask prediction from U²-Net

        Returns:
            Refined mask with edge case handling applied
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        image_np = np.array(image)

        # Apply all edge case handlers
        refined_mask = mask_pred.copy()

        # Detect and handle transparency
        if self._has_transparency(image_np):
            refined_mask = self.handle_transparency(image, refined_mask)

        # Detect and handle reflectivity  
        if self._has_reflectivity(image_np):
            refined_mask = self.handle_reflectivity(image, refined_mask)

        # Always apply fine detail preservation
        refined_mask = self.handle_fine_details(image, refined_mask)

        # Handle texture complexity
        if self._has_complex_texture(image_np):
            refined_mask = self.handle_texture_complexity(image, refined_mask)

        return refined_mask

    def handle_transparency(self, image: Image.Image, mask_pred: np.ndarray) -> np.ndarray:
        """Enhanced processing for transparent and translucent objects"""
        # Convert to different color spaces for transparency analysis
        image_np = np.array(image)
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

        # Detect potential transparent regions using multiple cues
        transparency_mask = self._detect_transparency_regions(hsv, lab)

        # Refine mask with transparency awareness
        refined_mask = self._refine_transparent_mask(mask_pred, transparency_mask)

        self.logger.info("Applied transparency handling")
        return refined_mask

    def handle_reflectivity(self, image: Image.Image, mask_pred: np.ndarray) -> np.ndarray:
        """Enhanced processing for reflective surfaces (jewelry, metallic objects)"""
        image_np = np.array(image)

        # Detect reflective regions using gradient and intensity analysis
        reflective_regions = self._detect_reflective_regions(image_np)

        # Apply reflectivity-aware refinement
        refined_mask = self._refine_reflective_mask(mask_pred, reflective_regions)

        self.logger.info("Applied reflectivity handling")
        return refined_mask

    def handle_fine_details(self, image: Image.Image, mask_pred: np.ndarray) -> np.ndarray:
        """Enhanced processing for hair, fur, and intricate patterns"""
        image_np = np.array(image)

        # Multi-scale edge detection for fine details
        fine_details = self._detect_fine_details(image_np)

        # Apply detail preservation filter
        refined_mask = self._preserve_fine_details(mask_pred, fine_details)

        self.logger.info("Applied fine detail preservation")
        return refined_mask

    def handle_texture_complexity(self, image: Image.Image, mask_pred: np.ndarray) -> np.ndarray:
        """Processing for complex textured surfaces"""
        image_np = np.array(image)

        # Analyze texture complexity using multiple methods
        texture_map = self._analyze_texture_complexity(image_np)

        # Apply texture-aware refinement
        refined_mask = self._refine_textured_mask(mask_pred, texture_map)

        self.logger.info("Applied texture complexity handling")
        return refined_mask

    def _has_transparency(self, image_np: np.ndarray) -> bool:
        """Detect if image likely contains transparent elements"""
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

        # Check for low saturation + high value (common in transparency)
        low_sat_high_val = np.sum((hsv[:,:,1] < 50) & (hsv[:,:,2] > 200))
        total_pixels = image_np.shape[0] * image_np.shape[1]

        return (low_sat_high_val / total_pixels) > 0.1

    def _has_reflectivity(self, image_np: np.ndarray) -> bool:
        """Detect if image likely contains reflective elements"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Compute gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Check for high gradient regions (potential reflections)
        high_gradient_threshold = np.percentile(gradient_magnitude, 95)
        high_gradient_pixels = np.sum(gradient_magnitude > high_gradient_threshold)
        total_pixels = image_np.shape[0] * image_np.shape[1]

        return (high_gradient_pixels / total_pixels) > 0.05

    def _has_complex_texture(self, image_np: np.ndarray) -> bool:
        """Detect if image has complex texture patterns"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Compute local standard deviation
        kernel = np.ones((9,9), np.float32) / 81
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)

        # High variance indicates complex texture
        mean_variance = np.mean(local_variance)
        return mean_variance > 400  # Threshold for complex texture

    def _detect_transparency_regions(self, hsv_image: np.ndarray, 
                                   lab_image: np.ndarray) -> np.ndarray:
        """Detect potentially transparent regions using multiple color spaces"""
        # HSV-based detection
        low_saturation = hsv_image[:, :, 1] < 60
        high_value = hsv_image[:, :, 2] > 180
        hsv_transparency = low_saturation & high_value

        # LAB-based detection (low A and B values often indicate transparency)
        lab_a_centered = np.abs(lab_image[:, :, 1] - 128) < 20
        lab_b_centered = np.abs(lab_image[:, :, 2] - 128) < 20
        lab_transparency = lab_a_centered & lab_b_centered

        # Combine both methods
        combined_transparency = hsv_transparency | lab_transparency

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        transparency_mask = cv2.morphologyEx(
            combined_transparency.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        )

        return transparency_mask.astype(np.float32)

    def _detect_reflective_regions(self, image_np: np.ndarray) -> np.ndarray:
        """Detect reflective regions using gradient and intensity analysis"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Multi-scale gradient detection
        grad_scales = []
        for kernel_size in [3, 5, 7]:
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            grad_scales.append(grad_magnitude)

        # Combine multi-scale gradients
        combined_gradient = np.mean(grad_scales, axis=0)

        # Threshold for reflection detection
        reflection_threshold = np.percentile(combined_gradient, 85)
        reflective_regions = (combined_gradient > reflection_threshold).astype(np.float32)

        # Also detect very bright regions (specular highlights)
        bright_threshold = np.percentile(gray, 95)
        bright_regions = (gray > bright_threshold).astype(np.float32)

        # Combine gradient-based and brightness-based detection
        combined_reflective = np.maximum(reflective_regions, bright_regions * 0.5)

        return combined_reflective

    def _detect_fine_details(self, image_np: np.ndarray) -> np.ndarray:
        """Detect fine details using multi-scale edge detection"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Multi-scale Canny edge detection
        edges_scales = []
        for low_threshold in [50, 100, 150]:
            edges = cv2.Canny(gray, low_threshold, low_threshold * 2)
            edges_scales.append(edges)

        # Combine multi-scale edges
        combined_edges = np.maximum.reduce(edges_scales)

        # Enhance hair-like structures using morphological operations
        # Hair typically appears as thin, elongated structures
        hair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        hair_enhanced = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, hair_kernel)

        # Also try horizontal hair detection
        hair_kernel_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 1))
        hair_enhanced_h = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, hair_kernel_h)

        # Combine both orientations
        fine_details = np.maximum(hair_enhanced, hair_enhanced_h)

        return fine_details.astype(np.float32) / 255.0

    def _analyze_texture_complexity(self, image_np: np.ndarray) -> np.ndarray:
        """Analyze texture complexity using multiple methods"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Local Binary Pattern for texture analysis
        def local_binary_pattern(image, radius=1, n_points=8):
            """Simplified LBP implementation"""
            rows, cols = image.shape
            lbp = np.zeros_like(image)

            for i in range(radius, rows-radius):
                for j in range(radius, cols-radius):
                    center = image[i, j]
                    binary_string = ''

                    # Sample points around the center
                    for angle in np.linspace(0, 2*np.pi, n_points, endpoint=False):
                        x = int(j + radius * np.cos(angle))
                        y = int(i + radius * np.sin(angle))

                        if 0 <= x < cols and 0 <= y < rows:
                            binary_string += '1' if image[y, x] > center else '0'

                    lbp[i, j] = int(binary_string, 2)

            return lbp

        # Compute texture features
        lbp = local_binary_pattern(gray)

        # Local standard deviation
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        local_std = np.sqrt(local_variance)

        # Combine LBP variance and local standard deviation
        lbp_variance = cv2.filter2D(lbp.astype(np.float32), -1, kernel)

        # Normalize and combine features
        texture_complexity = (local_std / np.max(local_std) + 
                            lbp_variance / np.max(lbp_variance)) / 2

        return texture_complexity

    def _preserve_fine_details(self, mask_pred: np.ndarray, 
                             fine_details: np.ndarray) -> np.ndarray:
        """Preserve fine details using edge information"""
        # Dilate fine details to create preservation zones
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_details = cv2.dilate((fine_details * 255).astype(np.uint8), kernel, iterations=1)
        dilated_details = dilated_details.astype(np.float32) / 255.0

        # Enhance mask in fine detail regions
        detail_boost = 0.3  # Boost factor for fine details
        refined_mask = mask_pred + (dilated_details * detail_boost * mask_pred)

        return np.clip(refined_mask, 0, 1)

    def _refine_transparent_mask(self, mask_pred: np.ndarray, 
                               transparency_mask: np.ndarray) -> np.ndarray:
        """Refine mask for transparent regions"""
        # Reduce confidence in transparent regions that might be background
        transparency_penalty = 0.2
        transparency_weight = 1.0 - (transparency_mask * transparency_penalty)

        refined_mask = mask_pred * transparency_weight

        return np.clip(refined_mask, 0, 1)

    def _refine_reflective_mask(self, mask_pred: np.ndarray, 
                              reflective_regions: np.ndarray) -> np.ndarray:
        """Refine mask for reflective regions"""
        # Boost confidence in reflective regions that are part of the object
        reflection_boost = 0.15
        refined_mask = mask_pred + (reflective_regions * reflection_boost * mask_pred)

        return np.clip(refined_mask, 0, 1)

    def _refine_textured_mask(self, mask_pred: np.ndarray, 
                            texture_map: np.ndarray) -> np.ndarray:
        """Refine mask based on texture complexity"""
        # High texture regions need more careful boundary handling
        texture_weight = 1.0 + (texture_map * 0.1)
        refined_mask = mask_pred * texture_weight

        return np.clip(refined_mask, 0, 1)

class AdvancedDataAugmentation:
    """Advanced data augmentation for challenging scenarios"""

    def __init__(self, image_size: int = 320):
        self.image_size = image_size
        self.transform = self._create_augmentation_pipeline()

    def _create_augmentation_pipeline(self):
        """Create comprehensive augmentation pipeline"""
        return A.Compose([
            # Geometric transformations
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
            ], p=0.5),

            A.OneOf([
                A.Rotate(limit=15, p=1.0),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=1.0),
                A.Perspective(scale=(0.05, 0.1), p=1.0),
            ], p=0.3),

            # Color and lighting augmentations for challenging scenarios
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            ], p=0.6),

            # Challenging lighting conditions
            A.OneOf([
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=1.0),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, 
                               num_flare_circles_lower=1, num_flare_circles_upper=2, p=1.0),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, alpha_coef=0.1, p=1.0),
            ], p=0.2),

            # Noise and quality degradation
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=1.0),
            ], p=0.2),

            # Blur effects (simulating motion, focus issues)
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.5), p=1.0),
            ], p=0.15),

            # Resize and normalize
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation to image and mask pair"""
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']

class EnsemblePredictor:
    """Ensemble of multiple models for improved accuracy"""

    def __init__(self, model_paths: List[str], device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = []

        # Load multiple model variants
        for path in model_paths:
            from .u2net_model import U2NET
            model = U2NET(3, 1)
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models.append(model)

    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Ensemble prediction with confidence weighting"""
        predictions = []
        confidences = []

        with torch.no_grad():
            for model in self.models:
                outputs = model(image_tensor)
                main_output = outputs[0]
                predictions.append(main_output)

                # Use entropy as confidence measure
                entropy = -torch.sum(main_output * torch.log(main_output + 1e-8), dim=1, keepdim=True)
                confidence = torch.exp(-entropy)  # Higher confidence for lower entropy
                confidences.append(confidence)

        # Weighted ensemble based on confidence
        total_confidence = torch.sum(torch.stack(confidences), dim=0)
        weighted_pred = torch.zeros_like(predictions[0])

        for pred, conf in zip(predictions, confidences):
            weight = conf / (total_confidence + 1e-8)
            weighted_pred += pred * weight

        return weighted_pred

# Example usage and utility functions
def create_edge_case_test_suite():
    """Create test cases for different edge case scenarios"""
    test_cases = {
        'transparency': [
            'glass_objects.jpg',
            'plastic_containers.jpg', 
            'translucent_materials.jpg'
        ],
        'reflectivity': [
            'jewelry_rings.jpg',
            'metallic_surfaces.jpg',
            'shiny_objects.jpg'
        ],
        'fine_details': [
            'human_hair.jpg',
            'animal_fur.jpg',
            'intricate_patterns.jpg'
        ],
        'texture_complexity': [
            'fabric_textures.jpg',
            'woven_materials.jpg',
            'complex_surfaces.jpg'
        ]
    }
    return test_cases

if __name__ == "__main__":
    print("Edge Case Handling Modules for U²-Net")
    print("=" * 50)
    print("✓ Transparency detection and handling")
    print("✓ Reflectivity processing for metallic surfaces")
    print("✓ Fine detail preservation (hair, fur, patterns)")
    print("✓ Texture complexity analysis and refinement") 
    print("✓ Advanced data augmentation pipeline")
    print("✓ Ensemble prediction with confidence weighting")
    print("✓ Comprehensive test case framework")
