#!/usr/bin/env python3
"""
Comparison of Detection Methods

This script compares basic template matching vs enhanced multi-rotation detection
to clearly show the improvements in detecting rotated icons.

Author: AI Assistant
Date: 2024
"""

import os
from blueprint_icon_detector import BlueprintIconDetector
from advanced_blueprint_detector import AdvancedBlueprintDetector

def compare_detection_methods():
    """Compare basic vs enhanced detection methods"""
    
    print("=" * 60)
    print("🔍 BLUEPRINT ICON DETECTION COMPARISON")
    print("=" * 60)
    
    blueprint_path = "demo_images/blueprint_with_doors.png"
    reference_path = "demo_images/door_icon.png"
    
    if not os.path.exists(blueprint_path):
        print("❌ Demo images not found. Run: python blueprint_icon_detector.py")
        return
    
    # Basic Detection
    print("\n1️⃣ BASIC TEMPLATE MATCHING")
    print("-" * 40)
    basic_detector = BlueprintIconDetector()
    basic_results = basic_detector.template_match_icons(
        blueprint_path, reference_path, threshold=0.6
    )
    
    print(f"Total detections: {len(basic_results)}")
    print("Detections by confidence:")
    for i, det in enumerate(basic_results, 1):
        print(f"  {i}. Position: ({det['x']}, {det['y']}) "
              f"Confidence: {det['confidence']:.3f}")
    
    # Enhanced Detection  
    print("\n2️⃣ ENHANCED MULTI-ROTATION DETECTION")
    print("-" * 40)
    enhanced_detector = AdvancedBlueprintDetector()
    enhanced_results = enhanced_detector.enhanced_detection(
        blueprint_path, reference_path, 
        confidence_threshold=0.6, 
        filter_false_positives=False
    )
    
    print(f"Total detections: {enhanced_results['summary']['total_matches']}")
    print(f"Detections by rotation: {enhanced_results['summary']['by_rotation']}")
    
    print("\nDetailed enhanced detections:")
    for i, det in enumerate(enhanced_results['template_matches'], 1):
        print(f"  {i}. Position: ({det['x']}, {det['y']}) "
              f"Rotation: {det['rotation']}° "
              f"Confidence: {det['confidence']:.3f}")
    
    # Analysis
    print("\n📊 ANALYSIS")
    print("-" * 40)
    
    # Count rotated detections
    rotated_detections = [d for d in enhanced_results['template_matches'] 
                         if d['rotation'] in [90, 270]]
    
    print(f"✅ Rotated doors found by enhanced method: {len(rotated_detections)}")
    print(f"❌ Rotated doors found by basic method: 0")
    
    print(f"\n🎯 IMPROVEMENT SUMMARY:")
    print(f"   Basic method:    {len(basic_results)} detections (no rotations)")
    print(f"   Enhanced method: {enhanced_results['summary']['total_matches']} detections (all rotations)")
    print(f"   Rotation coverage: {list(enhanced_results['summary']['by_rotation'].keys())}")
    
    # High confidence analysis
    print("\n🔍 HIGH-CONFIDENCE DETECTIONS (>0.8)")
    print("-" * 40)
    
    basic_high_conf = [d for d in basic_results if d['confidence'] > 0.8]
    enhanced_high_conf = [d for d in enhanced_results['template_matches'] if d['confidence'] > 0.8]
    
    print(f"Basic method: {len(basic_high_conf)} high-confidence detections")
    print(f"Enhanced method: {len(enhanced_high_conf)} high-confidence detections")
    
    print("\nHigh-confidence enhanced detections:")
    for det in enhanced_high_conf:
        print(f"  • Position: ({det['x']}, {det['y']}) "
              f"Rotation: {det['rotation']}° "
              f"Confidence: {det['confidence']:.3f}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    print("✅ Use Enhanced Method when:")
    print("   • Blueprint contains rotated symbols")
    print("   • Need comprehensive detection coverage")
    print("   • Can afford slightly longer processing time")
    
    print("\n⚡ Use Basic Method when:")
    print("   • All symbols have same orientation")
    print("   • Need maximum speed")
    print("   • Simple, consistent blueprints")
    
    print("\n🚀 PRODUCTION RECOMMENDATION:")
    print("   Use Enhanced Method with confidence threshold 0.8+")
    print("   This gives you complete rotation coverage with high accuracy!")
    
    return basic_results, enhanced_results

if __name__ == "__main__":
    compare_detection_methods() 