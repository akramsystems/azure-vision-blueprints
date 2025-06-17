#!/usr/bin/env python3
"""
Advanced Blueprint Icon Detection with Enhanced Algorithm

This enhanced version includes multiple improvements for better detection:
- Multi-threshold detection with adaptive thresholds
- Enhanced template preprocessing 
- Multiple template variations
- Better false positive filtering
- Improved edge case handling

Author: AI Assistant
Date: 2024
"""

import os
import cv2
import numpy as np
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from azure_object_detection import AzureObjectDetector
import matplotlib.pyplot as plt
from math import sqrt

load_dotenv()

class AdvancedBlueprintDetector:
    """
    Enhanced blueprint detector with multiple detection improvements
    """
    
    def __init__(self):
        """Initialize the advanced detector"""
        self.azure_detector = AzureObjectDetector()
        
        # Detection parameters - 15 degree increments for maximum coverage
        self.rotation_angles = [i * 15 for i in range(24)]  # 24 rotation angles: 0°, 15°, 30°, 45°... 345°
        self.scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2]  # Size variations
        
        # Multi-threshold approach
        self.thresholds = [0.85, 0.8, 0.75, 0.7]  # Try multiple thresholds
        
    def create_enhanced_templates(self, template: np.ndarray) -> List[np.ndarray]:
        """
        Create multiple enhanced template variations
        
        Args:
            template: Original template image (grayscale)
            
        Returns:
            List of template variations (all grayscale)
        """
        templates = []
        
        # Original template
        templates.append(template)
        
        # Enhanced contrast version
        enhanced = cv2.convertScaleAbs(template, alpha=1.2, beta=10)
        templates.append(enhanced)
        
        # Edge-enhanced version
        edges = cv2.Canny(template, 50, 150)
        # Combine edges with original template (both are grayscale)
        edge_template = cv2.bitwise_or(template, edges)
        templates.append(edge_template)
        
        # Slightly blurred version (for noisy blueprints)
        blurred = cv2.GaussianBlur(template, (3, 3), 0)
        templates.append(blurred)
        
        return templates
        
    def create_rotated_templates(self, template: np.ndarray) -> Dict[int, List[np.ndarray]]:
        """
        Create rotated versions of the template with variations
        
        Args:
            template: Original template image
            
        Returns:
            Dict mapping angles to list of template variations
        """
        # First create template variations
        template_variations = self.create_enhanced_templates(template)
        
        rotated_templates = {}
        
        for angle in self.rotation_angles:
            rotated_templates[angle] = []
            
            for template_var in template_variations:
                if angle == 0:
                    rotated_templates[angle].append(template_var)
                else:
                    # Get image center
                    (h, w) = template_var.shape[:2]
                    center = (w // 2, h // 2)
                    
                    # Create rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    
                    # Calculate new bounding dimensions
                    cos = np.abs(rotation_matrix[0, 0])
                    sin = np.abs(rotation_matrix[0, 1])
                    new_w = int((h * sin) + (w * cos))
                    new_h = int((h * cos) + (w * sin))
                    
                    # Adjust rotation matrix for new center
                    rotation_matrix[0, 2] += (new_w / 2) - center[0]
                    rotation_matrix[1, 2] += (new_h / 2) - center[1]
                    
                    # Perform rotation
                    rotated = cv2.warpAffine(template_var, rotation_matrix, (new_w, new_h), 
                                           borderValue=255)  # White background
                    
                    rotated_templates[angle].append(rotated)
        
        return rotated_templates

    def template_match_multi_threshold(self, blueprint_path: str, reference_icon_path: str, 
                                     primary_threshold: float = 0.9) -> List[Dict]:
        """
        Enhanced template matching with multiple thresholds and template variations
        
        Args:
            blueprint_path: Path to blueprint image
            reference_icon_path: Path to reference icon
            primary_threshold: Primary threshold for high-confidence detections
            
        Returns:
            List of detections with confidence and method info
        """
        # Load images
        blueprint = cv2.imread(blueprint_path, cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(reference_icon_path, cv2.IMREAD_GRAYSCALE)
        
        if blueprint is None or reference is None:
            print(f"Error loading images: {blueprint_path}, {reference_icon_path}")
            return []

        # Preprocess blueprint for better matching
        blueprint_enhanced = cv2.convertScaleAbs(blueprint, alpha=1.1, beta=5)
        
        all_detections = []
        
        # Create rotated template variations
        rotated_templates = self.create_rotated_templates(reference)
        
        total_combinations = len(self.rotation_angles) * len(self.scale_factors) * len(self.thresholds) * len(rotated_templates[0])
        print(f"Testing {len(self.rotation_angles)} rotations × {len(self.scale_factors)} scales × {len(self.thresholds)} thresholds × {len(rotated_templates[0])} template variations = {total_combinations} combinations")
        
        for threshold in self.thresholds:
            print(f"  Testing threshold {threshold}...")
            
            for angle, template_variations in rotated_templates.items():
                for template_idx, template in enumerate(template_variations):
                    for scale in self.scale_factors:
                        # Resize template
                        if scale != 1.0:
                            new_width = max(1, int(template.shape[1] * scale))
                            new_height = max(1, int(template.shape[0] * scale))
                            scaled_template = cv2.resize(template, (new_width, new_height))
                        else:
                            scaled_template = template
                        
                        # Skip if template is larger than blueprint
                        if (scaled_template.shape[0] >= blueprint.shape[0] or 
                            scaled_template.shape[1] >= blueprint.shape[1]):
                            continue
                        
                        # Try both original and enhanced blueprint
                        for bp_version, bp_name in [(blueprint, "original"), (blueprint_enhanced, "enhanced")]:
                            try:
                                result = cv2.matchTemplate(bp_version, scaled_template, cv2.TM_CCOEFF_NORMED)
                                locations = np.where(result >= threshold)
                                
                                h, w = scaled_template.shape
                                
                                for pt in zip(*locations[::-1]):
                                    detection = {
                                        'x': int(pt[0]),
                                        'y': int(pt[1]),
                                        'width': w,
                                        'height': h,
                                        'confidence': float(result[pt[1], pt[0]]),
                                        'rotation': angle,
                                        'scale': scale,
                                        'threshold': threshold,
                                        'template_variation': template_idx,
                                        'blueprint_version': bp_name,
                                        'method': 'enhanced_multi_threshold'
                                    }
                                    all_detections.append(detection)
                            
                            except cv2.error as e:
                                continue
        
        # Apply enhanced non-maximum suppression across all detections
        if all_detections:
            print(f"Raw detections before NMS: {len(all_detections)}")
            all_detections = self._apply_enhanced_nms(all_detections, overlap_threshold=0.25)
        
        print(f"Found {len(all_detections)} detections after enhanced NMS")
        return all_detections

    def _apply_enhanced_nms(self, detections: List[Dict], overlap_threshold: float = 0.25) -> List[Dict]:
        """
        Enhanced Non-Maximum Suppression with better duplicate handling
        """
        if not detections:
            return []
        
        # First, group detections by approximate location (within 20 pixels)
        location_groups = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
                
            group = [det]
            used.add(i)
            
            for j, other_det in enumerate(detections):
                if j in used or j == i:
                    continue
                    
                # Check if detections are close in location
                distance = sqrt((det['x'] - other_det['x'])**2 + (det['y'] - other_det['y'])**2)
                if distance < 20:  # Within 20 pixels
                    group.append(other_det)
                    used.add(j)
            
            location_groups.append(group)
        
        # For each group, keep the best detection
        final_detections = []
        for group in location_groups:
            # Sort by confidence
            group.sort(key=lambda x: x['confidence'], reverse=True)
            best_detection = group[0]
            
            # If we have multiple high-confidence detections, prefer certain characteristics
            high_conf_group = [d for d in group if d['confidence'] > 0.9]
            if len(high_conf_group) > 1:
                # Prefer original template (template_variation = 0)
                original_templates = [d for d in high_conf_group if d.get('template_variation', 0) == 0]
                if original_templates:
                    best_detection = max(original_templates, key=lambda x: x['confidence'])
            
            final_detections.append(best_detection)
        
        return final_detections

    def filter_false_positives(self, detections: List[Dict], min_confidence: float = 0.8,
                             aspect_ratio_range: Tuple[float, float] = (0.5, 2.0)) -> List[Dict]:
        """
        Filter out likely false positives based on confidence and geometric constraints
        
        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold
            aspect_ratio_range: Valid aspect ratio range (width/height)
            
        Returns:
            Filtered detections
        """
        filtered = []
        
        for det in detections:
            # Confidence filter
            if det['confidence'] < min_confidence:
                continue
            
            # Aspect ratio filter (for door-like objects)
            aspect_ratio = det['width'] / det['height']
            if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
                continue
            
            # Size filter (avoid very small matches)
            min_size = 20  # Minimum size in pixels
            if det['width'] < min_size or det['height'] < min_size:
                continue
            
            filtered.append(det)
        
        return filtered
    
    def region_adaptive_detection(self, blueprint_path: str, reference_icon_path: str, 
                                 threshold: float = 0.7) -> List[Dict]:
        """
        Adaptive detection that processes different regions with optimized parameters
        
        Args:
            blueprint_path: Path to blueprint image
            reference_icon_path: Path to reference icon
            threshold: Base threshold for detection
            
        Returns:
            List of detections from all regions
        """
        # Load images
        blueprint = cv2.imread(blueprint_path, cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(reference_icon_path, cv2.IMREAD_GRAYSCALE)
        
        if blueprint is None or reference is None:
            print(f"Error loading images: {blueprint_path}, {reference_icon_path}")
            return []

        h, w = blueprint.shape
        all_detections = []
        
        # Define overlapping regions for comprehensive coverage
        regions = [
            # Main regions
            {'name': 'top-left', 'bounds': (0, 0, w//2 + 50, h//2 + 50), 'threshold_adj': 0.0},
            {'name': 'top-right', 'bounds': (w//2 - 50, 0, w, h//2 + 50), 'threshold_adj': 0.0},
            {'name': 'bottom-left', 'bounds': (0, h//2 - 50, w//2 + 50, h), 'threshold_adj': 0.0},
            {'name': 'bottom-right', 'bounds': (w//2 - 50, h//2 - 50, w, h), 'threshold_adj': 0.0},
            
            # Edge regions (where doors might be harder to detect)
            {'name': 'top-edge', 'bounds': (0, 0, w, h//4), 'threshold_adj': -0.05},
            {'name': 'bottom-edge', 'bounds': (0, 3*h//4, w, h), 'threshold_adj': -0.05},
            {'name': 'left-edge', 'bounds': (0, 0, w//4, h), 'threshold_adj': -0.05},
            {'name': 'right-edge', 'bounds': (3*w//4, 0, w, h), 'threshold_adj': -0.05},
            
            # Center region (often has good contrast)
            {'name': 'center', 'bounds': (w//4, h//4, 3*w//4, 3*h//4), 'threshold_adj': 0.02},
        ]
        
        print(f"Processing {len(regions)} adaptive regions...")
        
        # Create rotated templates once
        rotated_templates = self.create_rotated_templates(reference)
        
        for region in regions:
            x1, y1, x2, y2 = region['bounds']
            region_threshold = max(0.6, threshold + region['threshold_adj'])
            
            # Extract region
            region_img = blueprint[y1:y2, x1:x2]
            if region_img.size == 0:
                continue
                
            # Process this region with current parameters
            for angle, template_variations in rotated_templates.items():
                for template_idx, template in enumerate(template_variations):
                    for scale in [0.9, 1.0, 1.1]:  # Focus on most likely scales
                        
                        # Resize template
                        if scale != 1.0:
                            new_width = max(1, int(template.shape[1] * scale))
                            new_height = max(1, int(template.shape[0] * scale))
                            scaled_template = cv2.resize(template, (new_width, new_height))
                        else:
                            scaled_template = template
                        
                        # Skip if template is larger than region
                        if (scaled_template.shape[0] >= region_img.shape[0] or 
                            scaled_template.shape[1] >= region_img.shape[1]):
                            continue
                        
                        try:
                            result = cv2.matchTemplate(region_img, scaled_template, cv2.TM_CCOEFF_NORMED)
                            locations = np.where(result >= region_threshold)
                            
                            temp_h, temp_w = scaled_template.shape
                            
                            for pt in zip(*locations[::-1]):
                                detection = {
                                    'x': int(pt[0] + x1),  # Adjust to global coordinates
                                    'y': int(pt[1] + y1),  # Adjust to global coordinates
                                    'width': temp_w,
                                    'height': temp_h,
                                    'confidence': float(result[pt[1], pt[0]]),
                                    'rotation': angle,
                                    'scale': scale,
                                    'threshold': region_threshold,
                                    'template_variation': template_idx,
                                    'region': region['name'],
                                    'method': 'region_adaptive'
                                }
                                all_detections.append(detection)
                        
                        except cv2.error:
                            continue
        
        # Apply global NMS across all regions
        if all_detections:
            print(f"Raw detections from all regions: {len(all_detections)}")
            all_detections = self._apply_enhanced_nms(all_detections, overlap_threshold=0.2)
        
        return all_detections

    def ultimate_detection(self, blueprint_path: str, reference_icon_path: str,
                          confidence_threshold: float = 0.7) -> Dict:
        """
        Ultimate detection combining multiple methods for maximum door discovery
        
        Args:
            blueprint_path: Path to blueprint
            reference_icon_path: Path to reference icon  
            confidence_threshold: Base detection threshold
            
        Returns:
            Combined results from all detection methods
        """
        print(f"=== Ultimate Detection: {os.path.basename(blueprint_path)} ===")
        
        # Method 1: Multi-threshold detection
        print("1. Running multi-threshold detection...")
        multi_detections = self.template_match_multi_threshold(
            blueprint_path, reference_icon_path, confidence_threshold
        )
        
        # Method 2: Region-adaptive detection
        print("2. Running region-adaptive detection...")
        region_detections = self.region_adaptive_detection(
            blueprint_path, reference_icon_path, confidence_threshold
        )
        
        # Combine and deduplicate
        all_detections = multi_detections + region_detections
        print(f"Combined raw detections: {len(all_detections)}")
        
        # Final NMS across all methods
        final_detections = self._apply_enhanced_nms(all_detections, overlap_threshold=0.15)
        
        print(f"Final detections after ultimate NMS: {len(final_detections)}")
        
        # Filter false positives
        original_count = len(final_detections)
        final_detections = self.filter_false_positives(
            final_detections, 
            min_confidence=confidence_threshold
        )
        print(f"After false positive filtering: {original_count} → {len(final_detections)} detections")
        
        # Group by rotation for analysis
        rotation_groups = {}
        for det in final_detections:
            angle = det['rotation']
            if angle not in rotation_groups:
                rotation_groups[angle] = []
            rotation_groups[angle].append(det)
        
        # Azure Vision for context
        azure_results = self.azure_detector.analyze_image_file(blueprint_path)
        
        results = {
            'template_matches': final_detections,
            'rotation_analysis': rotation_groups,
            'azure_detections': azure_results or {},
            'summary': {
                'total_matches': len(final_detections),
                'by_rotation': {str(angle): len(dets) for angle, dets in rotation_groups.items()},
                'by_method': {
                    'multi_threshold': len([d for d in final_detections if d.get('method') == 'enhanced_multi_threshold']),
                    'region_adaptive': len([d for d in final_detections if d.get('method') == 'region_adaptive'])
                },
                'confidence_stats': {
                    'min': min([d['confidence'] for d in final_detections]) if final_detections else 0,
                    'max': max([d['confidence'] for d in final_detections]) if final_detections else 0,
                    'avg': sum([d['confidence'] for d in final_detections]) / len(final_detections) if final_detections else 0
                }
            }
        }
        
        return results
    
    def visualize_enhanced_results(self, blueprint_path: str, results: Dict, 
                                 output_path: str = None) -> str:
        """
        Visualize enhanced detection results with rotation information
        """
        image = cv2.imread(blueprint_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Single color for all detections
        detection_color = (0, 255, 0)  # Green for all detections
        
        # Draw detections with same color
        for match in results.get('template_matches', []):
            x, y, w, h = match['x'], match['y'], match['width'], match['height']
            rotation = match['rotation']
            color = detection_color
            
            # Draw rectangle
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color, 2)
            
            # Add detailed label
            label = f"{rotation}°: {match['confidence']:.2f}"
            cv2.putText(image_rgb, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add simple legend
        cv2.putText(image_rgb, "Door Detections", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)
        
        # Save result
        if output_path is None:
            output_path = f"enhanced_detection_{os.path.basename(blueprint_path)}"
        
        cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        print(f"Enhanced visualization saved to: {output_path}")
        
        return output_path

def test_enhanced_detection():
    """Test the enhanced detection on our demo blueprint"""
    detector = AdvancedBlueprintDetector()
    
    blueprint_path = "demo_images/blueprint_with_doors.png"
    reference_path = "demo_images/door_icon.png"
    
    if not os.path.exists(blueprint_path):
        print("Demo images not found. Run blueprint_icon_detector.py first to create them.")
        return
    
    # Test with lower threshold to catch more doors
    results = detector.ultimate_detection(
        blueprint_path=blueprint_path,
        reference_icon_path=reference_path,
        confidence_threshold=0.7
    )
    
    print(f"\n=== Ultimate Results ===")
    print(f"Total detections: {results['summary']['total_matches']}")
    print(f"Detections by rotation: {results['summary']['by_rotation']}")
    print(f"Detections by method: {results['summary']['by_method']}")
    
    confidence_stats = results['summary']['confidence_stats']
    print(f"Confidence range: {confidence_stats['min']:.3f} - {confidence_stats['max']:.3f}")
    
    print(f"\nDetailed detections:")
    for i, match in enumerate(results['template_matches'], 1):
        method = match.get('method', 'unknown')
        region = match.get('region', '')
        region_info = f" ({region})" if region else ""
        
        print(f"{i}. Position: ({match['x']}, {match['y']}) "
              f"Rotation: {match['rotation']}° "
              f"Scale: {match['scale']} "
              f"Method: {method}{region_info} "
              f"Confidence: {match['confidence']:.3f}")
    
    # Create visualization
    viz_path = detector.visualize_enhanced_results(blueprint_path, results)
    
    # Save detailed results
    with open("ultimate_detection_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: ultimate_detection_results.json")

if __name__ == "__main__":
    print("=== Enhanced Blueprint Detection Test ===\n")
    test_enhanced_detection() 