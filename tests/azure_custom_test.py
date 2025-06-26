#!/usr/bin/env python3
"""
Azure Custom Vision Test

Quick script to test Azure Vision on any image file or URL.
Usage: python azure_custom_test.py
"""

import os
import json
from methods.azure_vision.azure_object_detection import AzureObjectDetector
import sys

def test_custom_image():
    """Test Azure Vision on custom image"""
    
    # Initialize detector
    try:
        detector = AzureObjectDetector()
        print("✅ Azure Vision connected!")
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Get image path from user
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("📁 Enter image path or URL: ")
    
    print(f"\n🔍 Analyzing: {image_path}")
    
    # Analyze image
    if image_path.startswith("http"):
        results = detector.analyze_image_url(image_path)
    else:
        results = detector.analyze_image_file(image_path)
    
    if results:
        print(f"\n📊 RESULTS:")
        print(f"Objects found: {len(results['objects'])}")
        print(f"Tags found: {len(results['tags'])}")
        
        for obj in results['objects']:
            print(f"  🎯 {obj['name']} ({obj['confidence']:.3f})")
        
        print(f"\nTop tags:")
        for tag in sorted(results['tags'], key=lambda x: x['confidence'], reverse=True)[:5]:
            print(f"  🏷️  {tag['name']} ({tag['confidence']:.3f})")
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    test_custom_image() 