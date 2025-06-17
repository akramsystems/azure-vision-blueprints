# Azure Vision Studio Image Detection for Blueprint Icons

This project demonstrates how to use Azure Vision API to detect and count icons in blueprint images. It provides two main approaches:

1. **Out-of-the-box Object Detection** using Azure Vision API 4.0
2. **Template Matching** using OpenCV for precise icon detection
3. **Custom Vision** approach (training required)

## 🎯 Project Goal

Count multiple instances of architectural symbols (doors, windows, etc.) in blueprint images by:
- Using a reference image of the symbol
- Detecting all occurrences in the blueprint
- Handling rotations and scale variations
- Providing confidence scores and bounding boxes

## 📋 Prerequisites

- Azure subscription with Computer Vision resource
- Python 3.8+
- Azure Vision API key and endpoint

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download this repository
cd azure-vision-studio-image-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# Azure Vision API Configuration
VISION_KEY=your_azure_vision_api_key
VISION_ENDPOINT=your_azure_vision_endpoint
VISION_REGION=your_region

# Custom Vision Configuration (optional)
CUSTOM_VISION_TRAINING_KEY=your_training_key
CUSTOM_VISION_PREDICTION_KEY=your_prediction_key
CUSTOM_VISION_ENDPOINT=your_custom_vision_endpoint
CUSTOM_VISION_PROJECT_ID=your_project_id
```

### 3. Run the Demo

```bash
# Run the basic Azure object detection demo
python azure_object_detection.py

# Run the blueprint icon detection demo
python blueprint_icon_detector.py
```

## 📁 Project Structure

```
azure-vision-studio-image-detection/
├── azure_object_detection.py      # Core Azure Vision API wrapper
├── blueprint_icon_detector.py     # Blueprint-specific detection logic
├── custom_vision_example.py       # Custom Vision training example
├── requirements.txt                # Python dependencies
├── .env                           # Configuration (create this)
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
├── demo_images/                   # Generated demo images
│   ├── door_icon.png             # Reference door icon
│   └── blueprint_with_doors.png  # Sample blueprint
└── results/                       # Output images and JSON results
```

## 🔧 Features

### Azure Object Detection (`azure_object_detection.py`)
- Analyze images from URL or local file
- Extract objects with bounding boxes
- Filter architectural elements
- Visualize detections
- Count objects by type

### Blueprint Icon Detector (`blueprint_icon_detector.py`)
- Template matching for exact icon detection
- Combined Azure Vision + template matching
- Non-maximum suppression for overlapping detections
- Support for rotated icons
- Comprehensive result visualization

### Key Capabilities
- **Template Matching**: Precise detection using reference icons
- **Azure Vision**: General object detection with pre-trained models
- **Hybrid Approach**: Combines both methods for best results
- **Visualization**: Overlays bounding boxes on images
- **JSON Export**: Saves detailed results for analysis

## 📊 Detection Methods Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Azure Vision** | Pre-trained, recognizes common objects, no setup | Limited to known objects, may miss specific symbols | General architectural elements |
| **Template Matching** | Exact match detection, works with any icon | Sensitive to rotation/scale, requires clean reference | Specific blueprint symbols |
| **Custom Vision** | Trainable for specific use case, high accuracy | Requires training data and setup | Specialized icon sets |

## 🎮 Usage Examples

### Basic Object Detection
```python
from azure_object_detection import AzureObjectDetector

detector = AzureObjectDetector()
results = detector.analyze_image_file("blueprint.png")
door_count = detector.count_objects_by_name(results, "door")
print(f"Doors detected: {door_count}")
```

### Template Matching
```python
from blueprint_icon_detector import BlueprintIconDetector

detector = BlueprintIconDetector()
matches = detector.template_match_icons(
    "blueprint.png", 
    "door_reference.png", 
    threshold=0.7
)
print(f"Template matches: {len(matches)}")
```

### Combined Detection
```python
detector = BlueprintIconDetector()
results = detector.combined_detection(
    blueprint_path="blueprint.png",
    reference_icon_path="door_icon.png"
)
```

## 🔬 Understanding the Results

### Azure Vision Limitations for Blueprints
- **Pre-trained models**: May not recognize specific architectural symbols
- **Generic objects**: Better at detecting "door" as concept, not specific door symbols
- **Context dependency**: Works better with photographic images than line drawings
- **Scale sensitivity**: Small symbols (<5% of image) often missed

### Template Matching Advantages
- **Exact matching**: Finds precise symbol matches
- **Scale invariant**: Can handle different sizes with multi-scale detection
- **Rotation handling**: Built-in rotation detection
- **Custom symbols**: Works with any reference icon

### Recommended Approach

1. **Start with Template Matching** for known symbols
2. **Use Azure Vision** for general architectural element detection
3. **Combine both methods** for comprehensive coverage
4. **Consider Custom Vision** for specialized, high-volume use cases

## 🚀 Next Steps

### For Template Matching
- Add multi-scale detection for size variations
- Implement rotation-invariant matching
- Use feature-based matching (SIFT/ORB) for better robustness

### For Azure Vision
- Experiment with different confidence thresholds
- Use image preprocessing to enhance symbol visibility
- Combine with OCR for text-based legends

### For Custom Vision
- Collect training data (200+ images per symbol type)
- Create bounding box annotations
- Train custom object detection model
- Deploy for real-time detection

## 🐛 Troubleshooting

### Common Issues

1. **No objects detected**
   - Check image quality and resolution
   - Lower confidence thresholds
   - Verify API credentials in `.env`

2. **Template matching fails**
   - Ensure reference icon matches blueprint style
   - Adjust threshold (try 0.6-0.8 range)
   - Check for proper image preprocessing

3. **Azure API errors**
   - Verify API key and endpoint in `.env`
   - Check internet connection
   - Ensure image meets size requirements (<6MB)

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 API Reference

### AzureObjectDetector
- `analyze_image_url(url)`: Analyze image from URL
- `analyze_image_file(path)`: Analyze local image file
- `count_objects_by_name(results, name)`: Count specific objects
- `visualize_detections(image_path, results)`: Create visualization

### BlueprintIconDetector
- `template_match_icons(blueprint, reference, threshold)`: Template matching
- `detect_with_azure(blueprint)`: Azure Vision detection
- `combined_detection(blueprint, reference)`: Hybrid approach
- `visualize_combined_results(blueprint, results)`: Comprehensive visualization

## 📄 License

This project is for educational and demonstration purposes. Please ensure you comply with Azure service terms and your organization's policies when using the provided API keys.

## 🔗 References

- [Azure Computer Vision Documentation](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/)
- [Object Detection API Reference](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/concept-object-detection-40)
- [Custom Vision Service](https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/) 