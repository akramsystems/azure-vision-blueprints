# 🗂️ Project Structure

## Azure Vision Studio Blueprint Detection

This repository contains methods for detecting architectural elements (doors, windows, etc.) in blueprint images using various approaches including Azure Vision API and template matching.

## 📁 Directory Structure

```
azure-vision-studio-image-detection/
├── 📄 README.md                    # Main project documentation
├── 📄 APPROACH_SUMMARY.md          # Detailed approach analysis
├── 📄 PROJECT_STRUCTURE.md         # This file - project organization
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore patterns
│
├── 📁 methods/                     # Core detection algorithms
│   ├── 📁 azure_vision/           # Azure Vision API implementations
│   │   ├── azure_object_detection.py      # Main Azure Vision wrapper
│   │   ├── azure_embedding_detector.py    # Embedding-based approach
│   │   └── custom_vision_example.py       # Custom Vision integration
│   │
│   ├── 📁 template_matching/      # OpenCV template matching
│   │   ├── advanced_blueprint_detector.py # Multi-rotation detection
│   │   └── blueprint_icon_detector.py     # Basic template matching
│   │
│   ├── compare_detection_methods.py       # Method comparison utilities
│   ├── comprehensive_comparison.py        # Full benchmark suite
│   └── detection_improvement_analysis.py  # Performance analysis
│
├── 📁 tests/                      # Test scripts
│   ├── test_azure_blueprint.py            # Azure Vision tests
│   ├── test_with_proper_template.py       # Template matching tests
│   ├── azure_custom_test.py               # Custom Azure tests
│   └── door_detection_test.py             # General detection tests
│
├── 📁 scripts/                    # Utility scripts
│   └── create_complex_blueprint.py        # Blueprint generation tool
│
├── 📁 images/                     # Test images and templates
│   ├── 📁 blueprints/            # Test blueprint images
│   │   ├── blueprint_with_doors.png       # Main test blueprint
│   │   └── complex_blueprint_many_doors.png # Complex test case
│   │
│   └── 📁 templates/              # Door/window templates
│       ├── door_icon.png               # Basic door template
│       ├── door_icon.png                 # Icon-style template
│       └── architectural_door_template.png # Architectural template
│
└── 📁 results/                    # Output files and analysis
    ├── 📁 analysis/               # JSON analysis results
    │   ├── detection_results.json         # Basic detection results
    │   ├── enhanced_detection_results.json # Enhanced results
    │   ├── ultimate_detection_results.json # Final benchmark
    │   └── detection_improvement_analysis.json # Performance metrics
    │
    ├── 📁 images/                 # Generated visualization images
    │   ├── blueprint_detection_*.png      # Detection visualizations
    │   ├── enhanced_detection_*.png       # Enhanced detection results
    │   └── enhanced_template_detections.png # Template match results
    │
    ├── 📁 json_outputs/           # Structured JSON outputs
    └── 📁 visualizations/         # Generated charts and graphs
```

## 🚀 Quick Start

### 1. Core Detection Methods
```bash
# Azure Vision API detection
python methods/azure_vision/azure_object_detection.py

# Template matching detection  
python methods/template_matching/blueprint_icon_detector.py

# Compare all methods
python methods/compare_detection_methods.py
```

### 2. Run Tests
```bash
# Test Azure Vision
python tests/test_azure_blueprint.py

# Test template matching
python tests/test_with_proper_template.py
```

### 3. Generate Test Data
```bash
# Create complex blueprint with many doors
python scripts/create_complex_blueprint.py
```

## 📊 Method Performance Summary

| Method | Doors Detected | Success Rate | False Positives |
|--------|---------------|--------------|-----------------|
| **Azure Vision API** | 0 | 0% | None |
| **Basic Template** | 78 | 520% | High |
| **Enhanced Template** | 274 | 1827% | Very High |

> **Key Finding**: Azure Vision API is not effective for architectural symbol detection in technical drawings. Template matching with proper thresholds is more suitable for this use case.

## 🔧 Configuration

- **Azure Vision**: Requires `AZURE_VISION_KEY` and `AZURE_VISION_ENDPOINT` environment variables
- **Template Matching**: Configurable confidence thresholds in detection scripts
- **Test Images**: Standard test set in `images/blueprints/`

## 📈 Results Analysis

All detection results are automatically saved to:
- `results/analysis/` - JSON performance metrics
- `results/images/` - Visual detection overlays
- Console output with detailed analysis

## 🤝 Contributing

When adding new detection methods:
1. Place core algorithms in appropriate `methods/` subdirectory
2. Add tests to `tests/` folder
3. Update this documentation
4. Run comprehensive comparison to benchmark performance 