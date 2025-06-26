#!/usr/bin/env python3
"""
Test Azure Vision API on Blueprint Images

This script tests how Azure Vision performs specifically on architectural blueprints
and door detection compared to template matching approaches.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from methods.azure_vision.azure_object_detection import AzureObjectDetector

def test_azure_on_blueprints():
    """Test Azure Vision API on blueprint images"""
    
    print("=" * 60)
    print("🏗️  TESTING AZURE VISION ON BLUEPRINT IMAGES")
    print("=" * 60)
    
    # Initialize Azure detector
    try:
        detector = AzureObjectDetector()
        print("✅ Azure Vision client connected successfully!")
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Test images (updated paths for reorganized structure)
    test_images = [
        "../images/blueprints/blueprint_with_doors.png",
        "../images/templates/door_icon.png"
    ]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            continue
        
        # Check image dimensions before analysis
        from PIL import Image
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width < 50 or height < 50:
                    print(f"\n⚠️  SKIPPING: {image_path}")
                    print("-" * 50)
                    print(f"📐 Image dimensions: {width}x{height} pixels")
                    print("❌ Azure Vision API requires images to be at least 50x50 pixels")
                    print("💡 This image is too small for Azure Vision analysis")
                    continue
        except Exception as e:
            print(f"⚠️  Could not read image dimensions: {image_path} - {e}")
            continue
            
        print(f"\n🔍 ANALYZING: {image_path}")
        print("-" * 50)
        
        # Analyze the image
        results = detector.analyze_image_file(image_path)
        
        # Save raw results to JSON file for detailed inspection
        if results:
            import json
            results_filename = f"azure_results_{os.path.basename(image_path).replace('.', '_')}.json"
            with open(results_filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Raw results saved to: {results_filename}")
        
        if results:
            print(f"📐 Image size: {results['metadata']['width']}x{results['metadata']['height']}")
            print(f"🤖 Model: {results['metadata']['model_version']}")
            
            # Objects detected
            print(f"\n🎯 OBJECTS DETECTED: {len(results['objects'])}")
            if results['objects']:
                for i, obj in enumerate(results['objects'], 1):
                    bbox = obj['bounding_box']
                    print(f"  {i}. {obj['name']} (confidence: {obj['confidence']:.3f})")
                    print(f"     📍 Position: x={bbox['x']}, y={bbox['y']}, size={bbox['width']}x{bbox['height']}")
            else:
                print("  ❌ No objects detected")
            
            # Tags detected
            print(f"\n🏷️  ALL TAGS DETECTED ({len(results['tags'])} total):")
            for i, tag in enumerate(sorted(results['tags'], key=lambda x: x['confidence'], reverse=True), 1):
                print(f"  {i:2d}. {tag['name']} (confidence: {tag['confidence']:.3f})")
            
            # Show raw detection results for debugging
            print(f"\n🔍 RAW AZURE DETECTION RESULTS:")
            print(f"  • Total objects detected: {len(results['objects'])}")
            print(f"  • Total tags detected: {len(results['tags'])}")
            if results['objects']:
                print("  • Object details:")
                for obj in results['objects']:
                    print(f"    - {obj}")
            
            # Show confidence score distribution
            if results['tags']:
                high_conf = [t for t in results['tags'] if t['confidence'] >= 0.8]
                med_conf = [t for t in results['tags'] if 0.5 <= t['confidence'] < 0.8]
                low_conf = [t for t in results['tags'] if t['confidence'] < 0.5]
                
                print(f"\n📊 CONFIDENCE DISTRIBUTION:")
                print(f"  • High confidence (≥0.8): {len(high_conf)} tags")
                print(f"  • Medium confidence (0.5-0.8): {len(med_conf)} tags") 
                print(f"  • Low confidence (<0.5): {len(low_conf)} tags")
            
            # Search for door-related terms
            door_terms = ['door', 'gate', 'entrance', 'opening', 'barrier', 'panel']
            door_objects = detector.count_objects_by_name(results, 'door', min_confidence=0.3)
            
            print(f"\n🚪 DOOR DETECTION ANALYSIS:")
            print(f"  • Direct 'door' objects found: {door_objects}")
            
            door_related_tags = [tag for tag in results['tags'] 
                               if any(term in tag['name'].lower() for term in door_terms)]
            
            if door_related_tags:
                print(f"  • Door-related tags:")
                for tag in door_related_tags:
                    print(f"    - {tag['name']} ({tag['confidence']:.3f})")
            else:
                print(f"  • No door-related tags found")
            
            # Save visualization if objects were detected
            if results['objects']:
                output_file = f"azure_detection_{os.path.basename(image_path)}"
                viz_path = detector.visualize_detections(image_path, results, output_file)
                if viz_path:
                    print(f"  💾 Visualization saved: {viz_path}")
        
        else:
            print("❌ Failed to analyze image")
    
    print(f"\n📊 AZURE VISION ASSESSMENT FOR BLUEPRINTS:")
    print("-" * 50)
    print("✅ STRENGTHS:")
    print("  • Excellent for general object detection")
    print("  • Great for recognizing people, furniture, vehicles")
    print("  • High accuracy on common objects")
    print("  • Provides detailed tags and descriptions")
    
    print("\n❌ LIMITATIONS FOR BLUEPRINTS:")
    print("  • Not trained on architectural symbols")
    print("  • Sees blueprints as 'diagrams' or 'drawings'")
    print("  • Cannot distinguish door icons from other lines/rectangles")
    print("  • Better suited for photographs than technical drawings")
    
    print(f"\n💡 RECOMMENDATION:")
    print("  Use Azure Vision for: General image analysis, photo classification")
    print("  Use Template Matching for: Specific architectural symbol detection")
    print("  Best approach: Enhanced multi-rotation detection for blueprint doors")

if __name__ == "__main__":
    test_azure_on_blueprints() 