#!/usr/bin/env python3
"""
Door Detection Test Suite

Test different approaches for detecting doors in blueprint images.
"""

import os
import cv2
from methods.azure_vision.azure_object_detection import AzureObjectDetector
from methods.template_matching.blueprint_icon_detector import BlueprintIconDetector
import json

def test_door_detection_only():
    """Test ONLY door detection - the main objective"""
    
    print("🚪" * 20)
    print("🚪 DOOR DETECTION TEST - MAIN OBJECTIVE ONLY")
    print("🚪" * 20)
    
    # Initialize Azure detector
    try:
        detector = AzureObjectDetector()
        print("✅ Azure connected")
    except ValueError as e:
        print(f"❌ Azure failed: {e}")
        return
    
    # Test images - focus on door detection
    door_test_images = [
        "images/templates/door_icon.png",  # Updated path
        "images/blueprints/blueprint_with_doors.png",  # Main blueprint to test
        "images/blueprints/blueprint_with_doors.png",  # Same blueprint
        "images/blueprints/blueprint_with_doors.png"  # Same blueprint for comparison
    ]
    
    door_detection_results = []
    
    for image_path in door_test_images:
        if not os.path.exists(image_path):
            print(f"❌ Missing: {image_path}")
            continue
        
        # Check size
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 50 or height < 50:
                print(f"❌ {os.path.basename(image_path)}: TOO SMALL ({width}x{height})")
                continue
        
        print(f"\n🚪 Testing: {os.path.basename(image_path)} ({width}x{height})")
        
        # Analyze ONLY for doors
        results = detector.analyze_image_file(image_path)
        
        if not results:
            print("❌ Analysis failed")
            continue
        
        # DOOR DETECTION ANALYSIS - THE ONLY THING THAT MATTERS
        door_objects = []
        door_tags = []
        
        # Check objects for doors
        for obj in results.get('objects', []):
            if 'door' in obj['name'].lower():
                door_objects.append(obj)
        
        # Check tags for door-related terms
        door_terms = ['door', 'entrance', 'opening', 'gateway', 'portal', 'entry']
        for tag in results.get('tags', []):
            if any(term in tag['name'].lower() for term in door_terms):
                door_tags.append(tag)
        
        # RESULTS - DOOR DETECTION ONLY
        doors_found = len(door_objects)
        door_tags_found = len(door_tags)
        
        print(f"🎯 DOOR OBJECTS: {doors_found}")
        if door_objects:
            for i, door in enumerate(door_objects, 1):
                print(f"  {i}. {door['name']} (confidence: {door['confidence']:.3f})")
        else:
            print("  ❌ NO DOOR OBJECTS DETECTED")
        
        print(f"🏷️ DOOR TAGS: {door_tags_found}")
        if door_tags:
            for tag in door_tags:
                print(f"  • {tag['name']} (confidence: {tag['confidence']:.3f})")
        else:
            print("  ❌ NO DOOR TAGS DETECTED")
        
        # Show what Azure actually detected (for context)
        print(f"🔍 What Azure saw: {len(results.get('tags', []))} total tags")
        if results.get('tags'):
            top_3 = sorted(results['tags'], key=lambda x: x['confidence'], reverse=True)[:3]
            for tag in top_3:
                print(f"  • {tag['name']} ({tag['confidence']:.3f})")
        
        # VERDICT
        total_door_detections = doors_found + door_tags_found
        if total_door_detections > 0:
            print(f"✅ SUCCESS: Found {total_door_detections} door-related detections")
            success = True
        else:
            print("❌ FAILED: NO DOORS DETECTED")
            success = False
        
        door_detection_results.append({
            'image': os.path.basename(image_path),
            'size': f"{width}x{height}",
            'door_objects': doors_found,
            'door_tags': door_tags_found,
            'success': success
        })
    
    # FINAL VERDICT - DOES AZURE DETECT DOORS?
    print(f"\n" + "🚪" * 20)
    print("🚪 FINAL DOOR DETECTION VERDICT")
    print("🚪" * 20)
    
    successful_detections = sum(1 for r in door_detection_results if r['success'])
    total_tests = len(door_detection_results)
    
    for result in door_detection_results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{status} | {result['image']} ({result['size']}) | Objects: {result['door_objects']} | Tags: {result['door_tags']}")
    
    print(f"\n🎯 DOOR DETECTION SCORE: {successful_detections}/{total_tests}")
    
    if successful_detections == 0:
        print("❌ AZURE VISION IS USELESS FOR DOOR DETECTION")
        print("💡 STICK WITH TEMPLATE MATCHING - IT ACTUALLY WORKS")
    elif successful_detections < total_tests:
        print("⚠️ AZURE VISION IS UNRELIABLE FOR DOOR DETECTION")
        print("💡 TEMPLATE MATCHING IS MORE RELIABLE")
    else:
        print("✅ AZURE VISION CAN DETECT DOORS")
        print("💡 BUT CHECK IF IT'S BETTER THAN TEMPLATE MATCHING")

if __name__ == "__main__":
    test_door_detection_only() 