#!/usr/bin/env python3
"""
Azure DinoV2 Embedding-based Blueprint Detection

This approach uses Azure ML DinoV2 endpoint to generate embeddings for images
and uses cosine similarity for object detection in blueprints.
"""

import os
import cv2
import numpy as np
import json
import base64
import requests
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity

class AzureEmbeddingDetector:
    """Azure DinoV2 embedding-based object detector"""
    
    def __init__(self):
        """Initialize detector"""
        self.endpoint = os.environ.get('VISION_DINO_V2_ENDPOINT', 'https://quoto-measurement-detecti-wsmaj.eastus.inference.ml.azure.com/score')
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('VISION_DINO_V2_KEY', '')}"
        }
        
        if not os.environ.get('VISION_DINO_V2_KEY'):
            print("⚠️ VISION_DINO_V2_KEY environment variable not set")
        
        print("✅ Azure DinoV2 Embedding Detector initialized")
    
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Get DinoV2 embedding for an image"""
        try:
            # Encode image to base64
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            
            payload = {
                "input_data": {
                    "columns": ["image"],
                    "index": [0],
                    "data": [[b64]]
                }
            }
            
            resp = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=30)
            resp.raise_for_status()
            
            embedding = np.asarray(resp.json()[0]["image_features"])
            return embedding
            
        except Exception as e:
            print(f"❌ Error getting embedding for {image_path}: {e}")
            return None
    
    def analyze_reference_image(self, reference_path: str) -> dict:
        """Analyze reference image to get its embedding"""
        print(f"🔍 Analyzing reference image: {reference_path}")
        
        # Check if image exists
        if not os.path.exists(reference_path):
            print(f"❌ Reference image not found: {reference_path}")
            return None
            
        # Check image dimensions and resize if needed
        img = cv2.imread(reference_path)
        if img is not None:
            height, width = img.shape[:2]
            print(f"📐 Reference image dimensions: {width}x{height}")
            
            # If image is too small, resize it
            if width < 50 or height < 50:
                print(f"⚠️ Image too small. Resizing...")
                
                # Calculate new dimensions maintaining aspect ratio
                scale_factor = max(50 / width, 50 / height)
                new_width = int(width * scale_factor * 1.2)
                new_height = int(height * scale_factor * 1.2)
                
                # Resize image
                resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                # Create a larger canvas with white background
                canvas_size = max(new_width, new_height, 100)
                canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
                
                # Center the resized image on the canvas
                y_offset = (canvas_size - new_height) // 2
                x_offset = (canvas_size - new_width) // 2
                canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img
                
                # Save the resized image temporarily
                temp_reference_path = "temp_resized_reference.png"
                cv2.imwrite(temp_reference_path, canvas)
                
                print(f"✅ Resized to {canvas_size}x{canvas_size}")
                
                # Get embedding for resized image
                embedding = self.get_image_embedding(temp_reference_path)
                
                # Clean up temporary file
                if os.path.exists(temp_reference_path):
                    os.remove(temp_reference_path)
            else:
                embedding = self.get_image_embedding(reference_path)
        else:
            embedding = self.get_image_embedding(reference_path)
        
        if embedding is not None:
            print(f"✅ Reference embedding generated: shape {embedding.shape}")
            return {
                'embedding': embedding,
                'shape': embedding.shape,
                'path': reference_path
            }
        else:
            print(f"❌ Failed to generate embedding for reference image")
            return None
    
    def sliding_window_search(self, blueprint_path: str, reference_embedding: np.ndarray, 
                            window_size: tuple = (80, 80), stride: int = 40, 
                            similarity_threshold: float = 0.85) -> list:
        """Search for reference object using sliding window with embeddings"""
        print(f"🔍 Sliding window search on: {blueprint_path}")
        
        # Load blueprint
        blueprint = cv2.imread(blueprint_path)
        if blueprint is None:
            print(f"❌ Could not load blueprint: {blueprint_path}")
            return []
        
        height, width = blueprint.shape[:2]
        win_w, win_h = window_size
        detections = []
        
        print(f"📐 Blueprint: {width}x{height}, Window: {win_w}x{win_h}, Stride: {stride}")
        print(f"🎯 Similarity threshold: {similarity_threshold}")
        
        # Calculate total windows for progress
        total_windows = ((width - win_w) // stride + 1) * ((height - win_h) // stride + 1)
        processed = 0
        
        for y in range(0, height - win_h + 1, stride):
            for x in range(0, width - win_w + 1, stride):
                processed += 1
                if processed % 20 == 0:  # Show progress more frequently
                    progress = (processed / total_windows) * 100
                    print(f"   Progress: {progress:.1f}% ({processed}/{total_windows})")
                
                # Extract window
                window = blueprint[y:y+win_h, x:x+win_w]
                
                # Save window temporarily
                temp_path = f"temp_window.png"
                cv2.imwrite(temp_path, window)
                
                try:
                    # Get embedding for window
                    window_embedding = self.get_image_embedding(temp_path)
                    
                    if window_embedding is not None:
                        # Calculate cosine similarity
                        similarity = self.calculate_cosine_similarity(reference_embedding, window_embedding)
                        
                        if similarity > similarity_threshold:
                            detections.append({
                                'x': x,
                                'y': y,
                                'width': win_w,
                                'height': win_h,
                                'confidence': float(similarity),
                                'center_x': x + win_w // 2,
                                'center_y': y + win_h // 2
                            })
                            print(f"   🎯 Detection at ({x}, {y}) with similarity {similarity:.3f}")
                
                except Exception as e:
                    if processed % 100 == 0:  # Only print occasional errors to avoid spam
                        print(f"   ⚠️ Error processing window at ({x}, {y}): {e}")
                
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        print(f"✅ Found {len(detections)} potential matches")
        return detections
    
    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Reshape to ensure proper dimensions for cosine_similarity
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return similarity
    
    def detect_objects(self, blueprint_path: str, reference_path: str, 
                      window_size: tuple = (80, 80), stride: int = 40,
                      similarity_threshold: float = 0.85) -> dict:
        """Main detection method"""
        print("🚪 AZURE DINOV2 EMBEDDING-BASED DETECTION")
        print("-" * 50)
        
        # Analyze reference
        reference_data = self.analyze_reference_image(reference_path)
        if not reference_data:
            return {'error': 'Could not analyze reference image'}
        
        reference_embedding = reference_data['embedding']
        
        # Search blueprint
        detections = self.sliding_window_search(
            blueprint_path, reference_embedding, 
            window_size, stride, similarity_threshold
        )
        
        # Filter overlapping detections
        filtered_detections = self.apply_nms(detections)
        
        results = {
            'blueprint_path': blueprint_path,
            'reference_path': reference_path,
            'reference_embedding_shape': reference_data['shape'],
            'window_size': window_size,
            'stride': stride,
            'similarity_threshold': similarity_threshold,
            'raw_detections': len(detections),
            'final_detections': len(filtered_detections),
            'detections': filtered_detections
        }
        
        print(f"\n📊 RESULTS:")
        print(f"   Raw detections: {len(detections)}")
        print(f"   After NMS filtering: {len(filtered_detections)}")
        
        return results
    
    def apply_nms(self, detections: list, overlap_threshold: float = 0.5) -> list:
        """Apply non-maximum suppression"""
        if not detections:
            return []
        
        print(f"🔧 Applying NMS with overlap threshold: {overlap_threshold}")
        
        # Sort by confidence (similarity score)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for detection in detections:
            # Check overlap with kept detections
            should_keep = True
            for kept in keep:
                overlap = self.calculate_overlap(detection, kept)
                if overlap > overlap_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(detection)
        
        print(f"   Kept {len(keep)} detections after NMS")
        return keep
    
    def calculate_overlap(self, det1: dict, det2: dict) -> float:
        """Calculate IoU overlap between two detections"""
        x1_min, y1_min = det1['x'], det1['y']
        x1_max, y1_max = x1_min + det1['width'], y1_min + det1['height']
        
        x2_min, y2_min = det2['x'], det2['y']
        x2_max, y2_max = x2_min + det2['width'], y2_min + det2['height']
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = det1['width'] * det1['height']
        area2 = det2['width'] * det2['height']
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def visualize_results(self, results: dict, output_path: str = None) -> str:
        """Create visualization of detection results"""
        if not results or not results.get('detections'):
            print("❌ No detections to visualize")
            return None
        
        blueprint_path = results['blueprint_path']
        detections = results['detections']
        
        # Load image
        image = cv2.imread(blueprint_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        print(f"🎨 Creating visualization with {len(detections)} detections")
        
        # Draw detections
        for i, detection in enumerate(detections):
            x, y = detection['x'], detection['y']
            w, h = detection['width'], detection['height']
            confidence = detection['confidence']
            
            # Color based on confidence
            if confidence > 0.90:
                color = 'red'      # Very high confidence
            elif confidence > 0.87:
                color = 'orange'   # High confidence
            else:
                color = 'yellow'   # Medium confidence
            
            # Draw bounding box
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            
            # Draw label
            label = f"{i+1}: {confidence:.3f}"
            draw.text((x, y - 20), label, fill=color)
            
            # Draw center point
            center_x, center_y = x + w//2, y + h//2
            draw.ellipse([center_x-3, center_y-3, center_x+3, center_y+3], fill=color)
        
        # Save visualization
        if output_path is None:
            output_path = f"dinov2_embedding_detections.png"
        
        pil_image.save(output_path)
        print(f"💾 Visualization saved: {output_path}")
        
        return output_path

def test_dinov2_embedding():
    """Test DinoV2 embedding detection"""
    detector = AzureEmbeddingDetector()
    
    blueprint_path = "images/blueprints/blueprint_with_doors.png"
    reference_path = "images/templates/door_icon.png"
    
    if not os.path.exists(blueprint_path):
        print(f"❌ Blueprint not found: {blueprint_path}")
        return
    
    if not os.path.exists(reference_path):
        print(f"❌ Reference not found: {reference_path}")
        return
    
    # Run detection with optimized parameters
    results = detector.detect_objects(
        blueprint_path=blueprint_path,
        reference_path=reference_path,
        window_size=(60, 60),  # Smaller window for better precision
        stride=30,             # Smaller stride for better coverage
        similarity_threshold=0.85  # High threshold for DinoV2
    )
    
    # Save results
    with open('dinov2_embedding_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_copy = results.copy()
        if 'detections' in results_copy:
            for det in results_copy['detections']:
                if 'embedding' in det:
                    del det['embedding']  # Remove embeddings from saved results
        json.dump(results_copy, f, indent=2)
    
    # Create visualization
    viz_path = detector.visualize_results(results)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Blueprint: {blueprint_path}")
    print(f"   Reference: {reference_path}")
    print(f"   Objects detected: {len(results.get('detections', []))}")
    print(f"   Results saved: dinov2_embedding_results.json")
    if viz_path:
        print(f"   Visualization: {viz_path}")

if __name__ == "__main__":
    test_dinov2_embedding() 