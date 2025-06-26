#!/usr/bin/env python3
"""
Azure Vision API Object Detection for Blueprint Icons

This script demonstrates how to use Azure Vision API to detect objects in images,
specifically focusing on detecting and counting icons in blueprint images.

Author: AI Assistant
Date: 2024
"""

import os
import sys
from typing import List, Dict, Tuple
import json
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureObjectDetector:
    """Azure Vision API Object Detection wrapper class"""
    
    def __init__(self):
        """Initialize the Azure Vision client"""
        self.key = os.getenv('VISION_KEY')
        self.endpoint = os.getenv('VISION_ENDPOINT')
        
        if not self.key or not self.endpoint:
            raise ValueError("Please set VISION_KEY and VISION_ENDPOINT in your .env file")
        
        self.client = ImageAnalysisClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )
        
        logger.info(f"Azure Vision client initialized with endpoint: {self.endpoint}")
    
    def analyze_image_url(self, image_url: str) -> Dict:
        """
        Analyze an image from URL using Azure Vision API
        
        Args:
            image_url (str): URL of the image to analyze
            
        Returns:
            Dict: Analysis results
        """
        try:
            result = self.client.analyze_from_url(
                image_url=image_url,
                visual_features=[VisualFeatures.OBJECTS, VisualFeatures.TAGS],
                language="en"
            )
            
            return self._format_results(result)
            
        except HttpResponseError as e:
            logger.error(f"HTTP Error: {e.status_code} - {e.reason}")
            logger.error(f"Message: {e.error.message}")
            return None
    
    def analyze_image_file(self, image_path: str) -> Dict:
        """
        Analyze a local image file using Azure Vision API
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict: Analysis results
        """
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            result = self.client.analyze(
                image_data=image_data,
                visual_features=[VisualFeatures.OBJECTS, VisualFeatures.TAGS],
                language="en"
            )
            
            return self._format_results(result)
            
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return None
        except HttpResponseError as e:
            logger.error(f"HTTP Error: {e.status_code} - {e.reason}")
            logger.error(f"Message: {e.error.message}")
            return None
    
    def _format_results(self, result) -> Dict:
        """
        Format the Azure Vision API results into a more usable structure
        
        Args:
            result: Azure Vision API result object
            
        Returns:
            Dict: Formatted results
        """
        formatted_result = {
            'metadata': {
                'width': result.metadata.width,
                'height': result.metadata.height,
                'model_version': result.model_version
            },
            'objects': [],
            'tags': []
        }
        
        # Extract objects
        if result.objects is not None:
            for obj in result.objects.list:
                object_info = {
                    'name': obj.tags[0].name,
                    'confidence': obj.tags[0].confidence,
                    'bounding_box': {
                        'x': obj.bounding_box.x,
                        'y': obj.bounding_box.y,
                        'width': obj.bounding_box.width,
                        'height': obj.bounding_box.height
                    }
                }
                formatted_result['objects'].append(object_info)
        
        # Extract tags
        if result.tags is not None:
            for tag in result.tags.list:
                tag_info = {
                    'name': tag.name,
                    'confidence': tag.confidence
                }
                formatted_result['tags'].append(tag_info)
        
        return formatted_result
    
    def count_objects_by_name(self, results: Dict, object_name: str, min_confidence: float = 0.5) -> int:
        """
        Count how many times a specific object appears in the detection results
        
        Args:
            results (Dict): Detection results from analyze_image_*
            object_name (str): Name of the object to count
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            int: Number of detected objects matching the criteria
        """
        if not results or 'objects' not in results:
            return 0
        
        count = 0
        for obj in results['objects']:
            if (object_name.lower() in obj['name'].lower() and 
                obj['confidence'] >= min_confidence):
                count += 1
        
        return count
    
    def visualize_detections(self, image_path: str, results: Dict, output_path: str = None) -> str:
        """
        Visualize object detections on the image
        
        Args:
            image_path (str): Path to the original image
            results (Dict): Detection results
            output_path (str): Path to save the visualized image
            
        Returns:
            str: Path to the saved visualization
        """
        if not results or 'objects' not in results:
            logger.warning("No detection results to visualize")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Convert BGR to RGB for proper color display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes
        for obj in results['objects']:
            bbox = obj['bounding_box']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            # Draw rectangle
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add label
            label = f"{obj['name']}: {obj['confidence']:.2f}"
            cv2.putText(image_rgb, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Save visualization
        if output_path is None:
            output_path = f"detected_{os.path.basename(image_path)}"
        
        cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        logger.info(f"Visualization saved to: {output_path}")
        
        return output_path

def demo_object_detection():
    """
    Demo function to test Azure object detection
    """
    print("=== Azure Vision Object Detection Demo ===\n")
    
    # Initialize detector
    try:
        detector = AzureObjectDetector()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please create a .env file with your Azure Vision API credentials:")
        print("VISION_KEY=your_api_key")
        print("VISION_ENDPOINT=your_endpoint")
        return
    
    # Test with a sample image URL (Microsoft's sample)
    sample_url = "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"
    
    print(f"Analyzing sample image: {sample_url}")
    results = detector.analyze_image_url(sample_url)
    
    if results:
        print("\n=== Detection Results ===")
        print(f"Image dimensions: {results['metadata']['width']}x{results['metadata']['height']}")
        print(f"Model version: {results['metadata']['model_version']}")
        
        print(f"\nDetected {len(results['objects'])} objects:")
        for i, obj in enumerate(results['objects'], 1):
            print(f"{i}. {obj['name']} (confidence: {obj['confidence']:.3f})")
            bbox = obj['bounding_box']
            print(f"   Location: x={bbox['x']}, y={bbox['y']}, w={bbox['width']}, h={bbox['height']}")
        
        print(f"\nTags ({len(results['tags'])}):")
        for tag in sorted(results['tags'], key=lambda x: x['confidence'], reverse=True)[:10]:
            print(f"- {tag['name']} ({tag['confidence']:.3f})")
        
        # Example: Count specific objects
        laptop_count = detector.count_objects_by_name(results, "laptop", min_confidence=0.5)
        person_count = detector.count_objects_by_name(results, "person", min_confidence=0.5)
        
        print(f"\nObject counting examples:")
        print(f"- Laptops detected: {laptop_count}")
        print(f"- People detected: {person_count}")
    
    else:
        print("Failed to analyze image")

if __name__ == "__main__":
    demo_object_detection() 