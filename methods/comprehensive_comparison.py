#!/usr/bin/env python3
"""
Comprehensive Door Detection Comparison

Compares ALL detection methods:
1. Basic Template Matching
2. Enhanced Template Matching  
3. Azure Vision API (tags/objects)
4. Azure Vision Embedding (NEW)

Includes visualizations for all methods.
"""

import os
import json
from blueprint_icon_detector import BlueprintIconDetector
from advanced_blueprint_detector import AdvancedBlueprintDetector
from azure_embedding_detector import AzureEmbeddingDetector

def run_comprehensive_comparison():
    """Run comprehensive comparison of all detection methods"""
    
    print("=" * 80)
    print("🏆 COMPREHENSIVE DOOR DETECTION COMPARISON")
    print("=" * 80)
    
    blueprint_path = "images/blueprints/blueprint_with_doors.png"
    reference_path = "images/templates/door_icon.png"
    expected_doors = 15
    
    if not os.path.exists(blueprint_path) or not os.path.exists(reference_path):
        print("❌ Required images not found")
        return
    
    print(f"📋 Blueprint: {blueprint_path}")
    print(f"🚪 Reference: {reference_path}")
    print(f"🎯 Expected doors: {expected_doors}")
    
    results = {}
    
    # 1. Basic Template Matching
    print(f"\n1️⃣ BASIC TEMPLATE MATCHING")
    print("-" * 60)
    try:
        basic_detector = BlueprintIconDetector()
        basic_results = basic_detector.template_match_icons(
            blueprint_path, reference_path, threshold=0.7  # Higher threshold to reduce false positives
        )
        
        # Create visualization
        basic_viz = basic_detector.visualize_detections(
            blueprint_path, basic_results, "basic_template_detections.png"
        )
        
        results['basic_template'] = {
            'detections': len(basic_results),
            'method': 'Basic Template Matching',
            'visualization': basic_viz,
            'success_rate': min(100, (len(basic_results) / expected_doors) * 100),
            'raw_results': basic_results
        }
        
        print(f"✅ Detections: {len(basic_results)}")
        print(f"📊 Success rate: {results['basic_template']['success_rate']:.1f}%")
        print(f"💾 Visualization: {basic_viz}")
        
    except Exception as e:
        print(f"❌ Basic template matching failed: {e}")
        results['basic_template'] = {'detections': 0, 'error': str(e)}
    
    # 2. Enhanced Template Matching
    print(f"\n2️⃣ ENHANCED TEMPLATE MATCHING")
    print("-" * 60)
    try:
        enhanced_detector = AdvancedBlueprintDetector()
        enhanced_results = enhanced_detector.ultimate_detection(
            blueprint_path, reference_path, confidence_threshold=0.8  # Higher threshold
        )
        
        # Create visualization
        enhanced_viz = enhanced_detector.visualize_enhanced_results(
            blueprint_path, enhanced_results, "enhanced_template_detections.png"
        )
        
        total_enhanced = enhanced_results['summary']['total_matches']
        results['enhanced_template'] = {
            'detections': total_enhanced,
            'method': 'Enhanced Template Matching',
            'visualization': enhanced_viz,
            'success_rate': min(100, (total_enhanced / expected_doors) * 100),
            'rotations': enhanced_results['summary']['by_rotation'],
            'raw_results': enhanced_results
        }
        
        print(f"✅ Detections: {total_enhanced}")
        print(f"🔄 By rotation: {enhanced_results['summary']['by_rotation']}")
        print(f"📊 Success rate: {results['enhanced_template']['success_rate']:.1f}%")
        print(f"💾 Visualization: {enhanced_viz}")
        
    except Exception as e:
        print(f"❌ Enhanced template matching failed: {e}")
        results['enhanced_template'] = {'detections': 0, 'error': str(e)}
    
    # 3. Azure Vision API (Original)
    print(f"\n3️⃣ AZURE VISION API (TAGS/OBJECTS)")
    print("-" * 60)
    try:
        azure_basic = enhanced_detector.azure_detector
        azure_results = azure_basic.analyze_image_file(blueprint_path)
        
        azure_doors = 0
        azure_door_tags = 0
        
        if azure_results:
            # Count door objects
            for obj in azure_results.get('objects', []):
                if 'door' in obj['name'].lower():
                    azure_doors += 1
            
            # Count door-related tags
            door_terms = ['door', 'entrance', 'opening', 'gateway', 'portal', 'entry']
            for tag in azure_results.get('tags', []):
                if any(term in tag['name'].lower() for term in door_terms):
                    azure_door_tags += 1
        
        total_azure = azure_doors + azure_door_tags
        results['azure_basic'] = {
            'detections': total_azure,
            'method': 'Azure Vision API',
            'door_objects': azure_doors,
            'door_tags': azure_door_tags,
            'success_rate': (total_azure / expected_doors) * 100,
            'top_tags': [tag['name'] for tag in sorted(azure_results.get('tags', []), 
                        key=lambda x: x['confidence'], reverse=True)[:3]] if azure_results else []
        }
        
        print(f"✅ Door objects: {azure_doors}")
        print(f"✅ Door tags: {azure_door_tags}")
        print(f"✅ Total detections: {total_azure}")
        print(f"📊 Success rate: {results['azure_basic']['success_rate']:.1f}%")
        print(f"🏷️ Top tags: {results['azure_basic']['top_tags']}")
        
    except Exception as e:
        print(f"❌ Azure Vision API failed: {e}")
        results['azure_basic'] = {'detections': 0, 'error': str(e)}
    
    # 4. Azure Vision Embedding (NEW!)
    print(f"\n4️⃣ AZURE VISION EMBEDDING (NEW)")
    print("-" * 60)
    try:
        embedding_detector = AzureEmbeddingDetector()
        embedding_results = embedding_detector.detect_objects(blueprint_path, reference_path)
        
        # Create visualization
        embedding_viz = embedding_detector.visualize_results(
            embedding_results, "azure_embedding_detections.png"
        )
        
        total_embedding = embedding_results.get('final_detections', 0)
        results['azure_embedding'] = {
            'detections': total_embedding,
            'method': 'Azure Vision Embedding',
            'visualization': embedding_viz,
            'raw_detections': embedding_results.get('raw_detections', 0),
            'success_rate': min(100, (total_embedding / expected_doors) * 100),
            'raw_results': embedding_results
        }
        
        print(f"✅ Raw detections: {embedding_results.get('raw_detections', 0)}")
        print(f"✅ Final detections: {total_embedding}")
        print(f"📊 Success rate: {results['azure_embedding']['success_rate']:.1f}%")
        print(f"💾 Visualization: {embedding_viz}")
        
    except Exception as e:
        print(f"❌ Azure Vision Embedding failed: {e}")
        results['azure_embedding'] = {'detections': 0, 'error': str(e)}
    
    # Final Analysis
    print(f"\n" + "=" * 80)
    print("🏆 FINAL RESULTS COMPARISON")
    print("=" * 80)
    
    # Sort by number of detections
    methods = []
    for key, result in results.items():
        if 'error' not in result:
            methods.append((result['method'], result['detections'], result.get('success_rate', 0)))
    
    methods.sort(key=lambda x: x[1], reverse=True)
    
    print(f"📊 DETECTION RESULTS (Expected: {expected_doors} doors):")
    print("-" * 60)
    for i, (method, detections, success_rate) in enumerate(methods, 1):
        status = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{status} {i}. {method:<25} {detections:>3} doors ({success_rate:>5.1f}%)")
    
    print(f"\n📁 VISUALIZATION FILES CREATED:")
    print("-" * 60)
    for key, result in results.items():
        if 'visualization' in result and result['visualization']:
            print(f"• {result['method']}: {result['visualization']}")
    
    # Accuracy Analysis
    print(f"\n🎯 ACCURACY ANALYSIS:")
    print("-" * 60)
    best_method = None
    best_accuracy = float('inf')
    
    for key, result in results.items():
        if 'error' not in result:
            detections = result['detections']
            accuracy_error = abs(detections - expected_doors)
            accuracy_score = max(0, 100 - (accuracy_error / expected_doors) * 100)
            
            print(f"• {result['method']}: {detections} detected, error={accuracy_error}, accuracy={accuracy_score:.1f}%")
            
            if accuracy_error < best_accuracy:
                best_accuracy = accuracy_error
                best_method = result['method']
    
    if best_method:
        print(f"\n🎯 MOST ACCURATE: {best_method} (error: {best_accuracy} doors)")
    
    # Save complete results
    with open('comprehensive_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Complete results saved: comprehensive_comparison_results.json")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 60)
    
    # Find the method closest to expected number
    closest_method = min(methods, key=lambda x: abs(x[1] - expected_doors))
    
    if closest_method[1] == 0:
        print("❌ ALL METHODS FAILED - No doors detected by any method")
        print("🔧 SUGGESTIONS:")
        print("   • Check template design")
        print("   • Adjust detection thresholds") 
        print("   • Try different reference images")
    elif abs(closest_method[1] - expected_doors) <= 3:
        print(f"✅ BEST METHOD: {closest_method[0]}")
        print(f"   • Detected {closest_method[1]} doors (expected {expected_doors})")
        print(f"   • Accuracy: {closest_method[2]:.1f}%")
    else:
        print("⚠️ ALL METHODS NEED IMPROVEMENT")
        print(f"   • Closest: {closest_method[0]} with {closest_method[1]} doors")
        print(f"   • Consider hybrid approaches or parameter tuning")
    
    return results

if __name__ == "__main__":
    run_comprehensive_comparison() 