#!/usr/bin/env python3
"""
Azure Vision Embedding-Based Blueprint Scanner

This approach combines Azure Vision's semantic understanding with systematic scanning:
- Extracts semantic features from reference image using Azure Vision
- Scans blueprint with sliding windows at multiple rotations and scales
- Uses Azure Vision to analyze each window region
- Compares semantic similarity to find matches
- Provides comprehensive detection with visualization

Author: AI Assistant  
Date: 2024
"""

import os
import cv2
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import tempfile
import time
from math import sqrt, cos, sin, radians
from .azure_object_detection import AzureObjectDetector

class AzureEmbeddingScanner:
    """
    Advanced Azure Vision embedding-based blueprint scanner with systematic approach
    """
    
    def __init__(self):
        """Initialize the Azure embedding scanner"""
        self.azure_detector = AzureObjectDetector()
        
        # Scanning parameters - similar to template matching but for semantic analysis
        self.rotation_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
        self.scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        self.similarity_thresholds = [0.85, 0.80, 0.75, 0.70, 0.65]
        
        # Window scanning parameters
        self.window_overlap = 0.5  # 50% overlap between windows
        self.min_window_size = 60  # Minimum window size in pixels
        
        # Cache for reference analysis
        self.reference_analysis = None
        self.reference_tags = None
        
    def analyze_reference_image(self, reference_path: str) -> Dict:
        """
        Analyze reference image with Azure Vision to extract semantic features
        
        Args:
            reference_path: Path to reference image
            
        Returns:
            Dict containing Azure Vision analysis of reference
        """
        print(f"🔍 Analyzing reference image with Azure Vision: {os.path.basename(reference_path)}")
        
        # Ensure reference image meets minimum size requirements
        img = cv2.imread(reference_path)
        if img is None:
            raise ValueError(f"Could not load reference image: {reference_path}")
            
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            print(f"⚠️  Reference image too small ({w}x{h}), resizing to meet Azure requirements")
            # Resize to minimum 50x50 while maintaining aspect ratio
            scale = max(50/w, 50/h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h))
            
            # Save temporary resized image
            temp_path = tempfile.mktemp(suffix='.png')
            cv2.imwrite(temp_path, img_resized)
            
            try:
                analysis = self.azure_detector.analyze_image_file(temp_path)
            finally:
                os.unlink(temp_path)
        else:
            analysis = self.azure_detector.analyze_image_file(reference_path)
        
        if not analysis:
            raise ValueError("Failed to analyze reference image with Azure Vision")
        
        # Cache the analysis
        self.reference_analysis = analysis
        self.reference_tags = analysis.get('tags', [])
        
        print(f"✅ Reference analysis complete: {len(self.reference_tags)} tags found")
        print(f"🏷️  Top reference tags: {', '.join([t['name'] for t in self.reference_tags[:5]])}")
        
        return analysis
    
    def create_rotated_windows(self, image: np.ndarray, window_size: Tuple[int, int]) -> List[Dict]:
        """
        Create rotated scanning windows across the image
        
        Args:
            image: Input image array
            window_size: (width, height) of scanning window
            
        Returns:
            List of window dictionaries with position and rotation info
        """
        windows = []
        img_h, img_w = image.shape[:2]
        win_w, win_h = window_size
        
        # Calculate step size based on overlap
        step_w = int(win_w * (1 - self.window_overlap))
        step_h = int(win_h * (1 - self.window_overlap))
        
        # Generate window positions
        for y in range(0, img_h - win_h + 1, step_h):
            for x in range(0, img_w - win_w + 1, step_w):
                for angle in self.rotation_angles:
                    for scale in self.scale_factors:
                        # Calculate scaled window size
                        scaled_w = int(win_w * scale)
                        scaled_h = int(win_h * scale)
                        
                        # Skip if scaled window is too large
                        if scaled_w > img_w or scaled_h > img_h:
                            continue
                        
                        # Adjust position for scaled window
                        center_x = x + win_w // 2
                        center_y = y + win_h // 2
                        
                        scaled_x = max(0, min(center_x - scaled_w // 2, img_w - scaled_w))
                        scaled_y = max(0, min(center_y - scaled_h // 2, img_h - scaled_h))
                        
                        window = {
                            'x': scaled_x,
                            'y': scaled_y,
                            'width': scaled_w,
                            'height': scaled_h,
                            'rotation': angle,
                            'scale': scale,
                            'center_x': center_x,
                            'center_y': center_y
                        }
                        windows.append(window)
        
        return windows
    
    def extract_and_analyze_window(self, image: np.ndarray, window: Dict) -> Optional[Dict]:
        """
        Extract window region and analyze with Azure Vision
        
        Args:
            image: Source image
            window: Window definition dict
            
        Returns:
            Azure Vision analysis of window region or None if failed
        """
        try:
            # Extract window region
            x, y, w, h = window['x'], window['y'], window['width'], window['height']
            window_region = image[y:y+h, x:x+w]
            
            # Apply rotation if needed
            if window['rotation'] != 0:
                window_region = self._rotate_image(window_region, window['rotation'])
            
            # Ensure window meets minimum size requirements
            region_h, region_w = window_region.shape[:2]
            if region_h < 50 or region_w < 50:
                return None
            
            # Save window to temporary file for Azure analysis
            temp_path = tempfile.mktemp(suffix='.png')
            cv2.imwrite(temp_path, window_region)
            
            try:
                analysis = self.azure_detector.analyze_image_file(temp_path)
                if analysis:
                    analysis['window_info'] = window
                return analysis
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"⚠️  Failed to analyze window at ({window['x']}, {window['y']}): {e}")
            return None
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle"""
        if angle == 0:
            return image
            
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                               borderValue=(255, 255, 255))
        return rotated
    
    def calculate_semantic_similarity(self, window_analysis: Dict, threshold: float = 0.7) -> float:
        """
        Calculate semantic similarity between window and reference
        
        Args:
            window_analysis: Azure analysis of window region
            threshold: Minimum similarity threshold
            
        Returns:
            Similarity score (0-1)
        """
        if not self.reference_tags or not window_analysis.get('tags'):
            return 0.0
        
        window_tags = window_analysis['tags']
        
        # Create tag dictionaries for easier lookup
        ref_tag_dict = {tag['name'].lower(): tag['confidence'] for tag in self.reference_tags}
        win_tag_dict = {tag['name'].lower(): tag['confidence'] for tag in window_tags}
        
        # Calculate weighted similarity based on common tags
        common_tags = set(ref_tag_dict.keys()) & set(win_tag_dict.keys())
        
        if not common_tags:
            return 0.0
        
        similarity_score = 0.0
        total_weight = 0.0
        
        for tag in common_tags:
            ref_conf = ref_tag_dict[tag]
            win_conf = win_tag_dict[tag]
            
            # Weight by reference confidence and calculate similarity
            weight = ref_conf
            tag_similarity = min(ref_conf, win_conf) / max(ref_conf, win_conf)
            
            similarity_score += tag_similarity * weight
            total_weight += weight
        
        final_score = similarity_score / total_weight if total_weight > 0 else 0.0
        
        # Boost score for highly specific architectural terms
        architectural_boost = 0.0
        architectural_terms = ['door', 'window', 'opening', 'entrance', 'gate', 'barrier']
        for term in architectural_terms:
            if term in common_tags:
                architectural_boost += 0.1
        
        return min(1.0, final_score + architectural_boost)
    
    def scan_blueprint_with_embeddings(self, blueprint_path: str, reference_path: str, 
                                     confidence_threshold: float = 0.7) -> Dict:
        """
        Comprehensive blueprint scanning using Azure Vision embeddings
        
        Args:
            blueprint_path: Path to blueprint image
            reference_path: Path to reference template
            confidence_threshold: Minimum similarity threshold for detections
            
        Returns:
            Dictionary containing all detections and analysis
        """
        print(f"🔍 Starting Azure Vision Embedding Scan")
        print(f"📋 Blueprint: {os.path.basename(blueprint_path)}")
        print(f"🎯 Reference: {os.path.basename(reference_path)}")
        print(f"🎚️  Confidence threshold: {confidence_threshold}")
        
        # Analyze reference image first
        self.analyze_reference_image(reference_path)
        
        # Load blueprint image
        blueprint = cv2.imread(blueprint_path)
        if blueprint is None:
            raise ValueError(f"Could not load blueprint: {blueprint_path}")
        
        blueprint_h, blueprint_w = blueprint.shape[:2]
        print(f"📐 Blueprint dimensions: {blueprint_w}x{blueprint_h}")
        
        # Load reference to get window size
        reference = cv2.imread(reference_path)
        ref_h, ref_w = reference.shape[:2]
        base_window_size = (ref_w, ref_h)
        
        print(f"🪟 Base window size: {base_window_size[0]}x{base_window_size[1]}")
        
        # Generate scanning windows
        windows = self.create_rotated_windows(blueprint, base_window_size)
        total_windows = len(windows)
        print(f"🔄 Generated {total_windows} scanning windows")
        print(f"   • {len(self.rotation_angles)} rotations × {len(self.scale_factors)} scales")
        print(f"   • Window overlap: {self.window_overlap * 100}%")
        
        detections = []
        processed_count = 0
        
        # Process windows in batches to avoid API rate limits
        batch_size = 10
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1}/{(len(windows) + batch_size - 1)//batch_size} ({len(batch)} windows)")
            
            for window in batch:
                processed_count += 1
                
                # Analyze window with Azure Vision
                window_analysis = self.extract_and_analyze_window(blueprint, window)
                
                if window_analysis:
                    # Calculate similarity with reference
                    for threshold in self.similarity_thresholds:
                        similarity = self.calculate_semantic_similarity(window_analysis, threshold)
                        
                        if similarity >= threshold:
                            detection = {
                                'x': window['x'],
                                'y': window['y'],
                                'width': window['width'],
                                'height': window['height'],
                                'confidence': similarity,
                                'rotation': window['rotation'],
                                'scale': window['scale'],
                                'threshold': threshold,
                                'method': 'azure_embedding_scan',
                                'azure_tags': window_analysis.get('tags', [])[:3],  # Top 3 tags
                                'azure_objects': window_analysis.get('objects', [])
                            }
                            detections.append(detection)
                            break  # Only use highest threshold that passes
                
                # Progress update
                if processed_count % 50 == 0:
                    print(f"   ⏳ Processed {processed_count}/{total_windows} windows ({processed_count/total_windows*100:.1f}%)")
            
            # Small delay between batches to respect API limits
            time.sleep(0.5)
        
        print(f"✅ Completed scanning: {processed_count} windows processed")
        print(f"🎯 Raw detections found: {len(detections)}")
        
        # Apply Non-Maximum Suppression to remove overlapping detections
        if detections:
            detections = self._apply_nms(detections, overlap_threshold=0.3)
            print(f"🔧 After NMS filtering: {len(detections)} detections")
        
        # Analyze reference image for comparison
        reference_analysis = self.reference_analysis
        
        results = {
            'detections': detections,
            'reference_analysis': reference_analysis,
            'scan_parameters': {
                'total_windows_scanned': processed_count,
                'rotation_angles': self.rotation_angles,
                'scale_factors': self.scale_factors,
                'similarity_thresholds': self.similarity_thresholds,
                'confidence_threshold': confidence_threshold,
                'window_overlap': self.window_overlap
            },
            'summary': {
                'total_detections': len(detections),
                'by_rotation': self._group_by_rotation(detections),
                'by_scale': self._group_by_scale(detections),
                'confidence_stats': self._calculate_confidence_stats(detections)
            }
        }
        
        return results
    
    def _apply_nms(self, detections: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
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
        """Calculate Intersection over Union between two bounding boxes"""
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
    
    def _group_by_rotation(self, detections: List[Dict]) -> Dict:
        """Group detections by rotation angle"""
        groups = {}
        for det in detections:
            angle = det['rotation']
            if angle not in groups:
                groups[angle] = []
            groups[angle].append(det)
        return {str(k): len(v) for k, v in groups.items()}
    
    def _group_by_scale(self, detections: List[Dict]) -> Dict:
        """Group detections by scale factor"""
        groups = {}
        for det in detections:
            scale = det['scale']
            if scale not in groups:
                groups[scale] = []
            groups[scale].append(det)
        return {str(k): len(v) for k, v in groups.items()}
    
    def _calculate_confidence_stats(self, detections: List[Dict]) -> Dict:
        """Calculate confidence statistics"""
        if not detections:
            return {'min': 0, 'max': 0, 'avg': 0, 'count': 0}
        
        confidences = [det['confidence'] for det in detections]
        return {
            'min': min(confidences),
            'max': max(confidences),
            'avg': sum(confidences) / len(confidences),
            'count': len(confidences)
        }
    
    def visualize_results(self, blueprint_path: str, results: Dict, output_path: str = None) -> str:
        """
        Create visualization of detection results
        
        Args:
            blueprint_path: Path to original blueprint
            results: Detection results dictionary
            output_path: Output path for visualization image
            
        Returns:
            Path to saved visualization
        """
        # Load blueprint image
        image = cv2.imread(blueprint_path)
        if image is None:
            raise ValueError(f"Could not load blueprint: {blueprint_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = results['detections']
        
        print(f"🎨 Creating visualization with {len(detections)} detections")
        
        # Color scheme for different confidence levels
        colors = {
            'high': (0, 255, 0),      # Green for high confidence (>0.8)
            'medium': (255, 165, 0),   # Orange for medium confidence (0.6-0.8)
            'low': (255, 0, 0)         # Red for low confidence (<0.6)
        }
        
        # Draw detections
        for i, detection in enumerate(detections):
            x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
            confidence = detection['confidence']
            rotation = detection['rotation']
            scale = detection['scale']
            
            # Choose color based on confidence
            if confidence >= 0.8:
                color = colors['high']
                color_name = 'High'
            elif confidence >= 0.6:
                color = colors['medium']
                color_name = 'Med'
            else:
                color = colors['low']
                color_name = 'Low'
            
            # Draw bounding box
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color, 2)
            
            # Create detailed label
            label = f"{i+1}: {confidence:.2f} ({rotation}°, {scale:.1f}x)"
            
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            cv2.rectangle(image_rgb, (x, y - text_h - 5), (x + text_w, y), color, -1)
            cv2.putText(image_rgb, label, (x, y - 2), font, font_scale, (255, 255, 255), font_thickness)
            
            # Add confidence indicator
            conf_label = f"{color_name}"
            cv2.putText(image_rgb, conf_label, (x + w - 40, y + 15), font, 0.3, color, 1)
        
        # Add legend
        legend_y = 30
        cv2.putText(image_rgb, "Azure Vision Embedding Scan Results", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        legend_y += 25
        cv2.putText(image_rgb, f"Total detections: {len(detections)}", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add confidence legend
        legend_y += 20
        for conf_level, color in [('High (>0.8)', colors['high']), 
                                 ('Medium (0.6-0.8)', colors['medium']), 
                                 ('Low (<0.6)', colors['low'])]:
            cv2.rectangle(image_rgb, (10, legend_y), (25, legend_y + 10), color, -1)
            cv2.putText(image_rgb, conf_level, (30, legend_y + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            legend_y += 15
        
        # Save visualization
        if output_path is None:
            output_path = f"results/images/azure_embedding_scan_{os.path.basename(blueprint_path)}"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        print(f"💾 Visualization saved to: {output_path}")
        
        return output_path

def test_azure_embedding_scanner():
    """Test function for the Azure embedding scanner"""
    print("=== Azure Vision Embedding Scanner Test ===\n")
    
    scanner = AzureEmbeddingScanner()
    
    # Test with organized image paths
    blueprint_path = "images/blueprints/blueprint_with_doors.png"
    reference_path = "images/templates/door_icon.png"
    
    # Check if files exist
    if not os.path.exists(blueprint_path):
        print(f"❌ Blueprint not found: {blueprint_path}")
        print("Available blueprints:")
        if os.path.exists("images/blueprints"):
            for f in os.listdir("images/blueprints"):
                if f.endswith('.png'):
                    print(f"  - images/blueprints/{f}")
        return
    
    if not os.path.exists(reference_path):
        print(f"❌ Reference template not found: {reference_path}")
        print("Available templates:")
        if os.path.exists("images/templates"):
            for f in os.listdir("images/templates"):
                if f.endswith('.png'):
                    print(f"  - images/templates/{f}")
        return
    
    try:
        # Run the embedding scan
        results = scanner.scan_blueprint_with_embeddings(
            blueprint_path=blueprint_path,
            reference_path=reference_path,
            confidence_threshold=0.65
        )
        
        # Print results summary
        print(f"\n=== Scan Results Summary ===")
        print(f"🎯 Total detections: {results['summary']['total_detections']}")
        print(f"📊 Confidence stats: {results['summary']['confidence_stats']}")
        print(f"🔄 By rotation: {results['summary']['by_rotation']}")
        print(f"📏 By scale: {results['summary']['by_scale']}")
        
        # Print detailed detections
        if results['detections']:
            print(f"\n📋 Detailed Detections:")
            for i, det in enumerate(results['detections'][:10], 1):  # Show first 10
                azure_tags = [tag['name'] for tag in det.get('azure_tags', [])]
                print(f"  {i}. Position: ({det['x']}, {det['y']}) "
                      f"Size: {det['width']}x{det['height']} "
                      f"Confidence: {det['confidence']:.3f} "
                      f"Rotation: {det['rotation']}° "
                      f"Scale: {det['scale']:.1f}x "
                      f"Tags: {', '.join(azure_tags[:2])}")
            
            if len(results['detections']) > 10:
                print(f"  ... and {len(results['detections']) - 10} more detections")
        
        # Create visualization
        viz_path = scanner.visualize_results(blueprint_path, results)
        print(f"\n🎨 Visualization created: {viz_path}")
        
        # Save detailed results
        results_path = "results/analysis/azure_embedding_scan_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Detailed results saved to: {results_path}")
        
    except Exception as e:
        print(f"❌ Error during scanning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_azure_embedding_scanner() 