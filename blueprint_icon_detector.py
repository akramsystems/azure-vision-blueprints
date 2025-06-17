#!/usr/bin/env python3
"""
Blueprint Icon Detection and Counting

This script provides specialized functionality for detecting and counting 
architectural icons in blueprint images using Azure Vision API.

Features:
- Template matching for reference icons
- Azure Vision API integration
- Icon counting with confidence thresholds
- Visualization of detected icons

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
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

class BlueprintIconDetector:
    """
    Specialized detector for blueprint architectural icons
    """
    
    def __init__(self):
        """Initialize the blueprint icon detector"""
        self.azure_detector = AzureObjectDetector()
        
        # Common architectural elements that Azure Vision might detect
        self.architectural_keywords = [
            'door', 'window', 'stairs', 'table', 'chair', 'sink', 'toilet',
            'cabinet', 'counter', 'appliance', 'light', 'fixture', 'outlet',
            'switch', 'vent', 'radiator', 'fireplace'
        ]
    
    def template_match_icons(self, blueprint_path: str, reference_icon_path: str, 
                           threshold: float = 0.7) -> List[Dict]:
        """
        Use OpenCV template matching to find reference icons in blueprint
        
        Args:
            blueprint_path (str): Path to the blueprint image
            reference_icon_path (str): Path to the reference icon
            threshold (float): Matching threshold (0-1)
            
        Returns:
            List[Dict]: List of detected icon locations and scores
        """
        # Load images in grayscale
        blueprint = cv2.imread(blueprint_path, cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(reference_icon_path, cv2.IMREAD_GRAYSCALE)
        
        if blueprint is None or reference is None:
            print(f"Error loading images: {blueprint_path}, {reference_icon_path}")
            return []
        
        # Perform template matching
        result = cv2.matchTemplate(blueprint, reference, cv2.TM_CCOEFF_NORMED)
        
        # Find locations where matching exceeds threshold
        locations = np.where(result >= threshold)
        
        detections = []
        h, w = reference.shape
        
        for pt in zip(*locations[::-1]):  # Switch x and y coordinates
            detection = {
                'x': int(pt[0]),
                'y': int(pt[1]),
                'width': w,
                'height': h,
                'confidence': float(result[pt[1], pt[0]]),
                'method': 'template_matching'
            }
            detections.append(detection)
        
        # Apply non-maximum suppression to remove overlapping detections
        if detections:
            detections = self._apply_nms(detections, overlap_threshold=0.3)
        
        return detections
    
    def _apply_nms(self, detections: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections (List[Dict]): List of detections
            overlap_threshold (float): IoU threshold for suppression
            
        Returns:
            List[Dict]: Filtered detections
        """
        if not detections:
            return []
        
        # Sort by confidence score
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # Keep the highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            remaining = []
            for det in detections:
                if self._calculate_iou(current, det) < overlap_threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_min, y1_min = box1['x'], box1['y']
        x1_max, y1_max = x1_min + box1['width'], y1_min + box1['height']
        
        x2_min, y2_min = box2['x'], box2['y']
        x2_max, y2_max = x2_min + box2['width'], y2_min + box2['height']
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def detect_with_azure(self, blueprint_path: str, target_objects: List[str] = None) -> Dict:
        """
        Use Azure Vision API to detect objects in blueprint
        
        Args:
            blueprint_path (str): Path to the blueprint image
            target_objects (List[str]): Specific objects to look for
            
        Returns:
            Dict: Azure detection results
        """
        results = self.azure_detector.analyze_image_file(blueprint_path)
        
        if not results:
            return {'objects': [], 'architectural_objects': []}
        
        # Filter for architectural objects
        architectural_objects = []
        for obj in results['objects']:
            obj_name = obj['name'].lower()
            if any(keyword in obj_name for keyword in self.architectural_keywords):
                architectural_objects.append(obj)
        
        results['architectural_objects'] = architectural_objects
        return results
    
    def combined_detection(self, blueprint_path: str, reference_icon_path: str = None,
                          template_threshold: float = 0.7) -> Dict:
        """
        Combine template matching and Azure Vision for comprehensive detection
        
        Args:
            blueprint_path (str): Path to the blueprint image
            reference_icon_path (str): Path to reference icon (optional)
            template_threshold (float): Template matching threshold
            
        Returns:
            Dict: Combined detection results
        """
        results = {
            'template_matches': [],
            'azure_detections': {},
            'summary': {}
        }
        
        # Template matching if reference icon provided
        if reference_icon_path and os.path.exists(reference_icon_path):
            print(f"Performing template matching with reference: {reference_icon_path}")
            template_matches = self.template_match_icons(
                blueprint_path, reference_icon_path, template_threshold
            )
            results['template_matches'] = template_matches
        
        # Azure Vision detection
        print("Performing Azure Vision object detection...")
        azure_results = self.detect_with_azure(blueprint_path)
        results['azure_detections'] = azure_results
        
        # Generate summary
        results['summary'] = {
            'template_matches_count': len(results['template_matches']),
            'azure_objects_count': len(azure_results.get('objects', [])),
            'azure_architectural_count': len(azure_results.get('architectural_objects', []))
        }
        
        return results
    
    def visualize_combined_results(self, blueprint_path: str, results: Dict, 
                                 output_path: str = None) -> str:
        """
        Visualize both template matching and Azure detection results
        
        Args:
            blueprint_path (str): Path to the original blueprint
            results (Dict): Combined detection results
            output_path (str): Output path for visualization
            
        Returns:
            str: Path to saved visualization
        """
        # Load image
        image = cv2.imread(blueprint_path)
        if image is None:
            print(f"Error loading image: {blueprint_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw template matches in red
        for match in results.get('template_matches', []):
            x, y, w, h = match['x'], match['y'], match['width'], match['height']
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label = f"Template: {match['confidence']:.2f}"
            cv2.putText(image_rgb, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw Azure detections in blue
        azure_objects = results.get('azure_detections', {}).get('objects', [])
        for obj in azure_objects:
            bbox = obj['bounding_box']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"Azure: {obj['name']} ({obj['confidence']:.2f})"
            cv2.putText(image_rgb, label, (x, y - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save visualization
        if output_path is None:
            output_path = f"blueprint_detection_{os.path.basename(blueprint_path)}"
        
        cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        print(f"Visualization saved to: {output_path}")
        
        return output_path

def create_demo_images():
    """
    Create demo images for testing - a simple door icon and a comprehensive blueprint with 28 doors
    """
    os.makedirs("demo_images", exist_ok=True)
    
    # Create a more realistic door icon (architectural style)
    door_icon = np.ones((60, 40, 3), dtype=np.uint8) * 255  # White background
    
    # Draw door opening (gap in wall) - thinner lines
    cv2.line(door_icon, (5, 5), (35, 5), (0, 0, 0), 1)    # Top of opening
    cv2.line(door_icon, (5, 55), (35, 55), (0, 0, 0), 1)  # Bottom of opening
    cv2.line(door_icon, (5, 5), (5, 55), (0, 0, 0), 1)    # Left side
    cv2.line(door_icon, (35, 5), (35, 55), (0, 0, 0), 1)  # Right side
    
    # Draw door leaf (the actual door) - lighter gray
    cv2.rectangle(door_icon, (8, 8), (32, 52), (100, 100, 100), 1)
    
    # Draw door handle - small circle
    cv2.circle(door_icon, (28, 30), 1, (0, 0, 0), -1)
    
    # Draw door swing arc (architectural convention) - very light
    cv2.ellipse(door_icon, (35, 55), (25, 25), 0, 0, 90, (180, 180, 180), 1)
    
    cv2.imwrite("demo_images/door_icon.png", door_icon)
    
    # Create a larger, more complex blueprint with 28 doors
    blueprint = np.ones((800, 1200, 3), dtype=np.uint8) * 255  # Larger white background
    
    # Draw building outline
    cv2.rectangle(blueprint, (50, 50), (1150, 750), (0, 0, 0), 3)
    
    # Draw complex floor plan with multiple rooms
    # Horizontal walls
    cv2.line(blueprint, (50, 200), (400, 200), (0, 0, 0), 2)
    cv2.line(blueprint, (450, 200), (800, 200), (0, 0, 0), 2)
    cv2.line(blueprint, (850, 200), (1150, 200), (0, 0, 0), 2)
    
    cv2.line(blueprint, (50, 350), (300, 350), (0, 0, 0), 2)
    cv2.line(blueprint, (350, 350), (600, 350), (0, 0, 0), 2)
    cv2.line(blueprint, (650, 350), (1150, 350), (0, 0, 0), 2)
    
    cv2.line(blueprint, (50, 500), (500, 500), (0, 0, 0), 2)
    cv2.line(blueprint, (550, 500), (900, 500), (0, 0, 0), 2)
    cv2.line(blueprint, (950, 500), (1150, 500), (0, 0, 0), 2)
    
    cv2.line(blueprint, (200, 600), (1150, 600), (0, 0, 0), 2)
    
    # Vertical walls  
    cv2.line(blueprint, (200, 50), (200, 350), (0, 0, 0), 2)
    cv2.line(blueprint, (400, 50), (400, 500), (0, 0, 0), 2)
    cv2.line(blueprint, (600, 50), (600, 600), (0, 0, 0), 2)
    cv2.line(blueprint, (800, 50), (800, 350), (0, 0, 0), 2)
    cv2.line(blueprint, (1000, 50), (1000, 750), (0, 0, 0), 2)
    
    cv2.line(blueprint, (300, 200), (300, 500), (0, 0, 0), 2)
    cv2.line(blueprint, (500, 350), (500, 600), (0, 0, 0), 2)
    cv2.line(blueprint, (700, 200), (700, 500), (0, 0, 0), 2)
    cv2.line(blueprint, (900, 350), (900, 600), (0, 0, 0), 2)
    
    # Define doors with 15-degree rotation increments for comprehensive testing!
    # Door icon is 60h×40w, margins adjusted for all rotations
    door_positions = [
        # Perimeter doors with major rotations
        (120, 80, 0), (320, 80, 0), (520, 80, 0), (720, 80, 0), (920, 80, 0),  # Top wall
        (1070, 120, 90), (1070, 280, 90), (1070, 420, 90), (1070, 680, 90),      # Right wall  
        (150, 670, 180), (450, 670, 180), (750, 670, 180), (1050, 670, 180),     # Bottom wall
        (100, 150, 270), (100, 420, 270), (100, 650, 270),                       # Left wall
        
        # Internal doors with 15-degree increments for comprehensive testing
        (195, 120, 15),   # 15° rotation
        (395, 320, 30),   # 30° rotation  
        (595, 420, 45),   # 45° rotation
        (795, 120, 60),   # 60° rotation
        (995, 320, 75),   # 75° rotation
        
        (120, 195, 105),  # 105° rotation
        (520, 195, 120),  # 120° rotation
        (920, 195, 135),  # 135° rotation
        
        (220, 345, 150),  # 150° rotation
        (720, 345, 165),  # 165° rotation
        
        (420, 495, 195),  # 195° rotation
        (820, 495, 210),  # 210° rotation
        
        (520, 595, 225),  # 225° rotation
        (300, 400, 240),  # 240° rotation
        (500, 250, 255),  # 255° rotation
        (700, 550, 285),  # 285° rotation
        (900, 400, 300),  # 300° rotation
        (200, 550, 315),  # 315° rotation
        (800, 250, 330),  # 330° rotation
        (600, 300, 345),  # 345° rotation
        
        # Additional doors to reach higher count
        (250, 250, 90),   # Extra vertical
        (350, 150, 180),  # Extra horizontal inverted
        (850, 450, 270), # Extra vertical left
        (450, 350, 0),   # Extra horizontal
    ]
    
    print(f"Creating blueprint with {len(door_positions)} doors at various rotations...")
    
    for i, (x, y, rotation) in enumerate(door_positions):
        # Create rotated door icon
        door_copy = door_icon.copy()
        
        # Apply rotation - handle any angle with rotation matrix
        if rotation == 0:
            # No rotation needed
            pass
        elif rotation == 90:
            # Use optimized OpenCV rotation for common angles
            door_copy = cv2.rotate(door_copy, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            door_copy = cv2.rotate(door_copy, cv2.ROTATE_180)
        elif rotation == 270:
            door_copy = cv2.rotate(door_copy, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # Use rotation matrix for all other angles (15°, 30°, 45°, etc.)
            h, w = door_copy.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -rotation, 1.0)
            
            # Calculate new bounding dimensions to avoid clipping
            cos_val = abs(rotation_matrix[0, 0])
            sin_val = abs(rotation_matrix[0, 1])
            new_w = int((h * sin_val) + (w * cos_val))
            new_h = int((h * cos_val) + (w * sin_val))
            
            # Adjust translation to center the rotated image
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            # Apply rotation with white background
            door_copy = cv2.warpAffine(door_copy, rotation_matrix, (new_w, new_h), 
                                     borderValue=(255, 255, 255))
        
        h, w = door_copy.shape[:2]
        
        # Ensure coordinates are within bounds
        x_end = min(x + w, blueprint.shape[1])
        y_end = min(y + h, blueprint.shape[0])
        w_actual = x_end - x
        h_actual = y_end - y
        
        if w_actual > 0 and h_actual > 0 and x >= 0 and y >= 0:
            # Use alpha blending so door doesn't completely cover wall lines
            door_region = door_copy[:h_actual, :w_actual]
            background_region = blueprint[y:y_end, x:x_end]
            
            # Create a mask where door pixels are not white
            door_gray = cv2.cvtColor(door_region, cv2.COLOR_BGR2GRAY)
            door_mask = door_gray < 240  # Non-white pixels
            
            # Only overlay the door lines, not the white background
            for c in range(3):  # For each color channel
                background_region[:, :, c] = np.where(
                    door_mask,
                    door_region[:, :, c] * 0.8 + background_region[:, :, c] * 0.2,  # Blend door lines
                    background_region[:, :, c]  # Keep original background (walls)
                )
            
            blueprint[y:y_end, x:x_end] = background_region
    
    cv2.imwrite("demo_images/blueprint_with_doors.png", blueprint)
    
    print("Demo images created:")
    print("- demo_images/door_icon.png (reference icon)")
    print(f"- demo_images/blueprint_with_doors.png (blueprint with {len(door_positions)} doors)")
    print("  Door rotations: 15-degree increments (0°, 15°, 30°, 45°... 345°) for comprehensive testing")

def main():
    """
    Main demo function
    """
    print("=== Blueprint Icon Detection Demo ===\n")
    
    # Create demo images
    create_demo_images()
    
    # Initialize detector
    detector = BlueprintIconDetector()
    
    # Test detection
    blueprint_path = "demo_images/blueprint_with_doors.png"
    reference_path = "demo_images/door_icon.png"
    
    print(f"\nAnalyzing blueprint: {blueprint_path}")
    print(f"Using reference icon: {reference_path}")
    
    # Run combined detection
    results = detector.combined_detection(
        blueprint_path=blueprint_path,
        reference_icon_path=reference_path,
        template_threshold=0.6  # Lower threshold for demo
    )
    
    # Print results
    print("\n=== Detection Results ===")
    print(f"Template matches found: {results['summary']['template_matches_count']}")
    print(f"Azure objects detected: {results['summary']['azure_objects_count']}")
    print(f"Azure architectural objects: {results['summary']['azure_architectural_count']}")
    
    if results['template_matches']:
        print("\nTemplate matching results:")
        for i, match in enumerate(results['template_matches'], 1):
            print(f"{i}. Position: ({match['x']}, {match['y']}) "
                  f"Confidence: {match['confidence']:.3f}")
    
    if results['azure_detections'].get('architectural_objects'):
        print("\nAzure architectural objects:")
        for i, obj in enumerate(results['azure_detections']['architectural_objects'], 1):
            print(f"{i}. {obj['name']} (confidence: {obj['confidence']:.3f})")
    
    # Create visualization
    viz_path = detector.visualize_combined_results(blueprint_path, results)
    print(f"\nVisualization saved to: {viz_path}")
    
    # Save results to JSON
    results_path = "detection_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_path}")

if __name__ == "__main__":
    main() 