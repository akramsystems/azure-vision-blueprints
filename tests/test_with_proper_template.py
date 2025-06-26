#!/usr/bin/env python3
"""
Test with Proper Architectural Door Template

Tests template matching using an architectural door template that actually
matches the door symbols in the blueprint.
"""

import os
from blueprint_icon_detector import BlueprintIconDetector
from advanced_blueprint_detector import AdvancedBlueprintDetector

def test_proper_template():
    """Test door detection with proper architectural template"""
    
    print("🚪" * 20)
    print("🚪 TESTING WITH PROPER ARCHITECTURAL DOOR TEMPLATE")
    print("🚪" * 20)
    
    blueprint_path = "images/blueprints/blueprint_with_doors.png"
    template_path = "images/templates/door_icon.png"
    
    if not os.path.exists(blueprint_path):
        print("❌ Complex blueprint not found")
        return
    
    if not os.path.exists(template_path):
        print("❌ Architectural template not found")
        return
    
    print(f"📋 Blueprint: {blueprint_path}")
    print(f"🚪 Template: {template_path}")
    print(f"🎯 Expected: 15 doors at various angles")
    
    # Test Basic Template Matching
    print(f"\n1️⃣ BASIC TEMPLATE MATCHING")
    print("-" * 50)
    basic_detector = BlueprintIconDetector()
    basic_results = basic_detector.template_match_icons(
        blueprint_path, template_path, threshold=0.5
    )
    
    print(f"🎯 Doors found: {len(basic_results)}")
    if basic_results:
        print("Top detections:")
        for i, det in enumerate(sorted(basic_results, key=lambda x: x['confidence'], reverse=True)[:5], 1):
            print(f"  {i}. Position: ({det['x']}, {det['y']}) Confidence: {det['confidence']:.3f}")
    else:
        print("❌ No doors detected")
    
    # Test Enhanced Detection
    print(f"\n2️⃣ ENHANCED MULTI-ROTATION DETECTION")
    print("-" * 50)
    enhanced_detector = AdvancedBlueprintDetector()
    enhanced_results = enhanced_detector.ultimate_detection(
        blueprint_path, template_path, confidence_threshold=0.5
    )
    
    total_enhanced = enhanced_results['summary']['total_matches']
    print(f"🎯 Doors found: {total_enhanced}")
    
    if total_enhanced > 0:
        print(f"Detections by rotation: {enhanced_results['summary']['by_rotation']}")
        print("Top detections:")
        for i, det in enumerate(sorted(enhanced_results['template_matches'], key=lambda x: x['confidence'], reverse=True)[:5], 1):
            print(f"  {i}. Position: ({det['x']}, {det['y']}) Rotation: {det['rotation']}° Confidence: {det['confidence']:.3f}")
    else:
        print("❌ No doors detected")
    
    # Results Analysis
    print(f"\n🏆 RESULTS COMPARISON")
    print("-" * 50)
    print(f"📊 Basic Template:   {len(basic_results)} doors detected")
    print(f"📊 Enhanced Method:  {total_enhanced} doors detected")
    print(f"📊 Expected doors:   15 doors")
    
    # Calculate success rates
    basic_rate = (len(basic_results) / 15) * 100 if len(basic_results) <= 15 else 100
    enhanced_rate = (total_enhanced / 15) * 100 if total_enhanced <= 15 else 100
    
    print(f"\n📈 DETECTION RATES:")
    print(f"   Basic Method:    {basic_rate:.1f}% ({len(basic_results)}/15)")
    print(f"   Enhanced Method: {enhanced_rate:.1f}% ({total_enhanced}/15)")
    
    # Winner
    if total_enhanced > len(basic_results):
        print(f"\n🥇 WINNER: Enhanced Method!")
        print(f"   Found {total_enhanced - len(basic_results)} more doors than basic method")
    elif len(basic_results) > total_enhanced:
        print(f"\n🥇 WINNER: Basic Method!")
        print(f"   Found {len(basic_results) - total_enhanced} more doors than enhanced method")
    else:
        print(f"\n🤝 TIE: Both methods found {len(basic_results)} doors")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if total_enhanced > 10 or len(basic_results) > 10:
        print("✅ Template matching works well for architectural door detection!")
        print("🎯 Use enhanced method for rotated doors")
    elif total_enhanced > 5 or len(basic_results) > 5:
        print("⚠️ Template matching shows promise but needs tuning")
        print("🔧 Consider adjusting thresholds or template design")
    else:
        print("❌ Template matching struggling - template might need redesign")
        print("🎨 Consider creating multiple template variations")

if __name__ == "__main__":
    test_proper_template() 