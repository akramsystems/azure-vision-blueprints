#!/usr/bin/env python3
"""
Detection Improvement Analysis

This script compares different detection methods to show improvements in door detection:
1. Basic template matching (original)
2. Enhanced multi-rotation detection 
3. Ultimate detection with multiple methods

Author: AI Assistant
Date: 2024
"""

import os
import cv2
import numpy as np
import json
from typing import List, Dict
import time
from methods.template_matching.blueprint_icon_detector import BlueprintIconDetector
from methods.template_matching.advanced_blueprint_detector import AdvancedBlueprintDetector

def analyze_detection_improvements():
    """
    Comprehensive analysis of detection improvements
    """
    print("=== DETECTION IMPROVEMENT ANALYSIS ===\n")
    
    blueprint_path = "images/blueprints/blueprint_with_doors.png"
    reference_path = "images/templates/door_icon.png"
    
    if not os.path.exists(blueprint_path):
        print("Creating demo images...")
        from .template_matching.blueprint_icon_detector import create_demo_images
        create_demo_images()
    
    # Initialize detectors
    basic_detector = BlueprintIconDetector()
    advanced_detector = AdvancedBlueprintDetector()
    
    results = {}
    
    # Method 1: Basic Template Matching (Original)
    print("1. Basic Template Matching (Original Method)")
    print("-" * 50)
    start_time = time.time()
    
    basic_matches = basic_detector.template_match_icons(
        blueprint_path, reference_path, threshold=0.7
    )
    
    basic_time = time.time() - start_time
    results['basic'] = {
        'matches': basic_matches,
        'count': len(basic_matches),
        'time': basic_time,
        'method': 'Basic template matching (single rotation)'
    }
    
    print(f"Found: {len(basic_matches)} doors")
    print(f"Time: {basic_time:.2f} seconds")
    
    # Method 2: Enhanced Multi-Rotation Detection 
    print(f"\n2. Enhanced Multi-Rotation Detection")
    print("-" * 50)
    start_time = time.time()
    
    enhanced_results = advanced_detector.template_match_multi_threshold(
        blueprint_path, reference_path, primary_threshold=0.7
    )
    
    enhanced_time = time.time() - start_time
    results['enhanced'] = {
        'matches': enhanced_results,
        'count': len(enhanced_results),
        'time': enhanced_time,
        'method': 'Multi-rotation + multi-threshold + template variations'
    }
    
    print(f"Found: {len(enhanced_results)} doors")
    print(f"Time: {enhanced_time:.2f} seconds")
    
    # Method 3: Ultimate Detection (All Methods Combined)
    print(f"\n3. Ultimate Detection (All Methods Combined)")
    print("-" * 50)
    start_time = time.time()
    
    ultimate_results = advanced_detector.ultimate_detection(
        blueprint_path, reference_path, confidence_threshold=0.7
    )
    
    ultimate_time = time.time() - start_time
    results['ultimate'] = {
        'matches': ultimate_results['template_matches'],
        'count': ultimate_results['summary']['total_matches'],
        'time': ultimate_time,
        'method': 'Multi-threshold + region-adaptive + advanced NMS',
        'by_method': ultimate_results['summary']['by_method']
    }
    
    print(f"Found: {ultimate_results['summary']['total_matches']} doors")
    print(f"Time: {ultimate_time:.2f} seconds")
    print(f"Method breakdown: {ultimate_results['summary']['by_method']}")
    
    # Improvement Analysis
    print(f"\n=== IMPROVEMENT ANALYSIS ===")
    print("-" * 50)
    
    basic_count = results['basic']['count']
    enhanced_count = results['enhanced']['count']
    ultimate_count = results['ultimate']['count']
    
    enhanced_improvement = ((enhanced_count - basic_count) / basic_count * 100) if basic_count > 0 else 0
    ultimate_improvement = ((ultimate_count - basic_count) / basic_count * 100) if basic_count > 0 else 0
    
    print(f"Basic Detection:     {basic_count} doors")
    print(f"Enhanced Detection:  {enhanced_count} doors (+{enhanced_improvement:.1f}%)")
    print(f"Ultimate Detection:  {ultimate_count} doors (+{ultimate_improvement:.1f}%)")
    
    # Rotation Analysis
    print(f"\n=== ROTATION COVERAGE ANALYSIS ===")
    print("-" * 50)
    
    def analyze_rotations(matches):
        rotation_counts = {}
        for match in matches:
            rotation = match.get('rotation', 0)
            rotation_counts[rotation] = rotation_counts.get(rotation, 0) + 1
        return rotation_counts
    
    basic_rotations = analyze_rotations(results['basic']['matches'])
    enhanced_rotations = analyze_rotations(results['enhanced']['matches'])
    ultimate_rotations = analyze_rotations(results['ultimate']['matches'])
    
    print("Basic Detection Rotations:", basic_rotations)
    print("Enhanced Detection Rotations:", enhanced_rotations)  
    print("Ultimate Detection Rotations:", ultimate_rotations)
    
    all_rotations = set()
    all_rotations.update(basic_rotations.keys())
    all_rotations.update(enhanced_rotations.keys())
    all_rotations.update(ultimate_rotations.keys())
    
    print(f"\nRotation Coverage:")
    print(f"Basic:    {len(basic_rotations)} different angles")
    print(f"Enhanced: {len(enhanced_rotations)} different angles")  
    print(f"Ultimate: {len(ultimate_rotations)} different angles")
    
    # Confidence Analysis
    print(f"\n=== CONFIDENCE ANALYSIS ===")
    print("-" * 50)
    
    def confidence_stats(matches):
        if not matches:
            return {'min': 0, 'max': 0, 'avg': 0}
        confidences = [m['confidence'] for m in matches]
        return {
            'min': min(confidences),
            'max': max(confidences),
            'avg': sum(confidences) / len(confidences)
        }
    
    basic_conf = confidence_stats(results['basic']['matches'])
    enhanced_conf = confidence_stats(results['enhanced']['matches'])
    ultimate_conf = confidence_stats(results['ultimate']['matches'])
    
    print(f"Basic:    {basic_conf['min']:.3f} - {basic_conf['max']:.3f} (avg: {basic_conf['avg']:.3f})")
    print(f"Enhanced: {enhanced_conf['min']:.3f} - {enhanced_conf['max']:.3f} (avg: {enhanced_conf['avg']:.3f})")
    print(f"Ultimate: {ultimate_conf['min']:.3f} - {ultimate_conf['max']:.3f} (avg: {ultimate_conf['avg']:.3f})")
    
    # Performance Analysis
    print(f"\n=== PERFORMANCE ANALYSIS ===")
    print("-" * 50)
    
    doors_per_second_basic = basic_count / basic_time if basic_time > 0 else 0
    doors_per_second_enhanced = enhanced_count / enhanced_time if enhanced_time > 0 else 0
    doors_per_second_ultimate = ultimate_count / ultimate_time if ultimate_time > 0 else 0
    
    print(f"Basic:    {doors_per_second_basic:.1f} doors detected per second")
    print(f"Enhanced: {doors_per_second_enhanced:.1f} doors detected per second")
    print(f"Ultimate: {doors_per_second_ultimate:.1f} doors detected per second")
    
    # Key Improvements Summary
    print(f"\n=== KEY IMPROVEMENTS IMPLEMENTED ===")
    print("-" * 50)
    
    improvements = [
        "✓ Multi-rotation detection (0°, 15°, 30°, ..., 345°)",
        "✓ Multi-scale detection (0.8x to 1.2x scaling)",
        "✓ Multi-threshold detection (0.7, 0.75, 0.8, 0.85)",
        "✓ Template variations (enhanced contrast, edge detection, blur)",
        "✓ Region-adaptive processing (9 overlapping regions)",
        "✓ Advanced Non-Maximum Suppression (spatial clustering)",
        "✓ Enhanced false positive filtering",
        "✓ Preprocessed blueprint variations",
        "✓ Robust error handling and boundary checks"
    ]
    
    for improvement in improvements:
        print(improvement)
    
    # Save comprehensive results
    analysis_results = {
        'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'methods_compared': {
            'basic': results['basic']['method'],
            'enhanced': results['enhanced']['method'], 
            'ultimate': results['ultimate']['method']
        },
        'detection_counts': {
            'basic': basic_count,
            'enhanced': enhanced_count,
            'ultimate': ultimate_count
        },
        'improvements': {
            'enhanced_vs_basic': f"{enhanced_improvement:.1f}%",
            'ultimate_vs_basic': f"{ultimate_improvement:.1f}%"
        },
        'rotation_coverage': {
            'basic': len(basic_rotations),
            'enhanced': len(enhanced_rotations),
            'ultimate': len(ultimate_rotations)
        },
        'confidence_stats': {
            'basic': basic_conf,
            'enhanced': enhanced_conf,
            'ultimate': ultimate_conf
        },
        'performance': {
            'basic_time': basic_time,
            'enhanced_time': enhanced_time,
            'ultimate_time': ultimate_time
        }
    }
    
    with open("detection_improvement_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n=== SUMMARY ===")
    print("-" * 50)
    print(f"🎯 BEST RESULT: Ultimate Detection found {ultimate_count} doors")
    print(f"📈 IMPROVEMENT: {ultimate_improvement:.1f}% more doors than basic method")
    print(f"🔄 ROTATION COVERAGE: {len(ultimate_rotations)} different angles detected")
    print(f"⚡ ANALYSIS COMPLETE: Results saved to detection_improvement_analysis.json")
    
    return analysis_results

if __name__ == "__main__":
    analyze_detection_improvements() 