#!/usr/bin/env python3
"""
Custom Vision Training Example for Blueprint Icons

This script demonstrates how to use Azure Custom Vision to train a model
specifically for detecting architectural icons in blueprints.

Note: This requires setting up Custom Vision resources in Azure and
collecting training data.

Author: AI Assistant
Date: 2024
"""

import os
import json
import time
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Note: These imports require azure-cognitiveservices-vision-customvision package
# pip install azure-cognitiveservices-vision-customvision

try:
    from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
    from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
    from azure.cognitiveservices.vision.customvision.training.models import ImageCreateEntry, Region
    from msrest.authentication import ApiKeyCredentials
    CUSTOM_VISION_AVAILABLE = True
except ImportError:
    CUSTOM_VISION_AVAILABLE = False
    print("Warning: Custom Vision SDK not installed. Install with:")
    print("pip install azure-cognitiveservices-vision-customvision")

load_dotenv()

class CustomVisionTrainer:
    """
    Custom Vision training and prediction wrapper
    """
    
    def __init__(self):
        """Initialize Custom Vision clients"""
        if not CUSTOM_VISION_AVAILABLE:
            raise ImportError("Custom Vision SDK not available")
        
        # Get credentials from environment
        self.training_key = os.getenv('CUSTOM_VISION_TRAINING_KEY')
        self.prediction_key = os.getenv('CUSTOM_VISION_PREDICTION_KEY')
        self.endpoint = os.getenv('CUSTOM_VISION_ENDPOINT')
        self.project_id = os.getenv('CUSTOM_VISION_PROJECT_ID')
        
        if not all([self.training_key, self.prediction_key, self.endpoint]):
            raise ValueError(
                "Please set CUSTOM_VISION_TRAINING_KEY, CUSTOM_VISION_PREDICTION_KEY, "
                "and CUSTOM_VISION_ENDPOINT in your .env file"
            )
        
        # Initialize clients
        training_credentials = ApiKeyCredentials(in_headers={"Training-key": self.training_key})
        prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": self.prediction_key})
        
        self.trainer = CustomVisionTrainingClient(self.endpoint, training_credentials)
        self.predictor = CustomVisionPredictionClient(self.endpoint, prediction_credentials)
        
        print(f"Custom Vision clients initialized with endpoint: {self.endpoint}")
    
    def create_project(self, project_name: str, description: str = None) -> str:
        """
        Create a new Custom Vision project for object detection
        
        Args:
            project_name (str): Name of the project
            description (str): Project description
            
        Returns:
            str: Project ID
        """
        try:
            # Create object detection project
            project = self.trainer.create_project(
                name=project_name,
                description=description or f"Blueprint icon detection project: {project_name}",
                domain_id="00000000-0000-0000-0000-000000000000",  # General object detection domain
                classification_type="MultiClass",
                target_export_platforms=["CoreML", "TensorFlow"]
            )
            
            print(f"Created project '{project_name}' with ID: {project.id}")
            return project.id
            
        except Exception as e:
            print(f"Error creating project: {e}")
            return None
    
    def create_tags(self, project_id: str, tag_names: List[str]) -> Dict[str, str]:
        """
        Create tags for object categories
        
        Args:
            project_id (str): Project ID
            tag_names (List[str]): List of tag names to create
            
        Returns:
            Dict[str, str]: Mapping of tag names to tag IDs
        """
        tag_map = {}
        
        for tag_name in tag_names:
            try:
                tag = self.trainer.create_tag(project_id, tag_name)
                tag_map[tag_name] = tag.id
                print(f"Created tag '{tag_name}' with ID: {tag.id}")
            except Exception as e:
                print(f"Error creating tag '{tag_name}': {e}")
        
        return tag_map
    
    def upload_training_images(self, project_id: str, images_data: List[Dict]) -> bool:
        """
        Upload training images with bounding box annotations
        
        Args:
            project_id (str): Project ID
            images_data (List[Dict]): List of image data with annotations
            
        Format of images_data:
        [
            {
                'image_path': 'path/to/image.jpg',
                'annotations': [
                    {
                        'tag_id': 'tag_id_for_door',
                        'left': 0.1,    # Normalized coordinates (0-1)
                        'top': 0.2,
                        'width': 0.1,
                        'height': 0.15
                    }
                ]
            }
        ]
        
        Returns:
            bool: Success status
        """
        try:
            image_entries = []
            
            for img_data in images_data:
                with open(img_data['image_path'], 'rb') as image_file:
                    image_bytes = image_file.read()
                
                # Create regions for bounding boxes
                regions = []
                for annotation in img_data.get('annotations', []):
                    region = Region(
                        tag_id=annotation['tag_id'],
                        left=annotation['left'],
                        top=annotation['top'],
                        width=annotation['width'],
                        height=annotation['height']
                    )
                    regions.append(region)
                
                # Create image entry
                image_entry = ImageCreateEntry(
                    name=os.path.basename(img_data['image_path']),
                    contents=image_bytes,
                    regions=regions
                )
                image_entries.append(image_entry)
            
            # Upload in batches (max 64 per batch)
            batch_size = 64
            for i in range(0, len(image_entries), batch_size):
                batch = image_entries[i:i + batch_size]
                result = self.trainer.create_images_from_data(project_id, batch)
                
                if result.is_batch_successful:
                    print(f"Successfully uploaded batch {i//batch_size + 1} ({len(batch)} images)")
                else:
                    print(f"Failed to upload batch {i//batch_size + 1}")
                    for img in result.images:
                        if img.status != "OK":
                            print(f"  Error with {img.source_url}: {img.status}")
            
            return True
            
        except Exception as e:
            print(f"Error uploading training images: {e}")
            return False
    
    def train_model(self, project_id: str) -> str:
        """
        Train the Custom Vision model
        
        Args:
            project_id (str): Project ID
            
        Returns:
            str: Iteration ID of trained model
        """
        try:
            print("Starting training...")
            iteration = self.trainer.train_project(project_id)
            
            # Wait for training to complete
            while iteration.status == "Training":
                print("Training in progress...")
                time.sleep(10)
                iteration = self.trainer.get_iteration(project_id, iteration.id)
            
            if iteration.status == "Completed":
                print(f"Training completed! Iteration ID: {iteration.id}")
                return iteration.id
            else:
                print(f"Training failed with status: {iteration.status}")
                return None
                
        except Exception as e:
            print(f"Error during training: {e}")
            return None
    
    def publish_iteration(self, project_id: str, iteration_id: str, 
                         publish_name: str = "blueprint_detection") -> bool:
        """
        Publish trained iteration for predictions
        
        Args:
            project_id (str): Project ID
            iteration_id (str): Iteration ID
            publish_name (str): Name for published model
            
        Returns:
            bool: Success status
        """
        try:
            self.trainer.publish_iteration(
                project_id, 
                iteration_id, 
                publish_name, 
                prediction_resource_id="your_prediction_resource_id"  # Replace with actual resource ID
            )
            print(f"Published iteration as '{publish_name}'")
            return True
            
        except Exception as e:
            print(f"Error publishing iteration: {e}")
            return False
    
    def predict_image(self, project_id: str, image_path: str, 
                     publish_name: str = "blueprint_detection") -> Dict:
        """
        Make predictions on a new image
        
        Args:
            project_id (str): Project ID
            image_path (str): Path to image for prediction
            publish_name (str): Name of published model
            
        Returns:
            Dict: Prediction results
        """
        try:
            with open(image_path, 'rb') as image_file:
                results = self.predictor.detect_image(
                    project_id, 
                    publish_name, 
                    image_file.read()
                )
            
            predictions = []
            for prediction in results.predictions:
                pred_data = {
                    'tag': prediction.tag_name,
                    'probability': prediction.probability,
                    'bounding_box': {
                        'left': prediction.bounding_box.left,
                        'top': prediction.bounding_box.top,
                        'width': prediction.bounding_box.width,
                        'height': prediction.bounding_box.height
                    }
                }
                predictions.append(pred_data)
            
            return {
                'predictions': predictions,
                'image_path': image_path
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

def create_sample_training_data():
    """
    Create sample training data structure for blueprint icons
    This shows the format needed for training Custom Vision
    """
    # Example training data structure
    sample_data = [
        {
            'image_path': 'training_data/blueprint1.jpg',
            'annotations': [
                {
                    'tag_id': 'door_tag_id',  # Will be replaced with actual tag ID
                    'left': 0.2,      # Door location as fraction of image width
                    'top': 0.3,       # Door location as fraction of image height  
                    'width': 0.05,    # Door width as fraction of image width
                    'height': 0.08    # Door height as fraction of image height
                },
                {
                    'tag_id': 'window_tag_id',
                    'left': 0.6,
                    'top': 0.4,
                    'width': 0.08,
                    'height': 0.05
                }
            ]
        },
        {
            'image_path': 'training_data/blueprint2.jpg',
            'annotations': [
                {
                    'tag_id': 'door_tag_id',
                    'left': 0.15,
                    'top': 0.5,
                    'width': 0.06,
                    'height': 0.09
                }
            ]
        }
    ]
    
    return sample_data

def demo_custom_vision_workflow():
    """
    Demonstrate the complete Custom Vision workflow
    """
    print("=== Custom Vision Blueprint Detection Demo ===\n")
    
    if not CUSTOM_VISION_AVAILABLE:
        print("Custom Vision SDK not available. Please install:")
        print("pip install azure-cognitiveservices-vision-customvision")
        print("\nTo use Custom Vision:")
        print("1. Create Custom Vision resources in Azure Portal")
        print("2. Get training and prediction keys")
        print("3. Update your .env file with the credentials")
        print("4. Collect and annotate training images (200+ per icon type)")
        print("5. Use this script to train your model")
        return
    
    try:
        trainer = CustomVisionTrainer()
    except Exception as e:
        print(f"Error initializing Custom Vision: {e}")
        print("\nTo set up Custom Vision:")
        print("1. Create Custom Vision Training and Prediction resources in Azure")
        print("2. Add the following to your .env file:")
        print("   CUSTOM_VISION_TRAINING_KEY=your_training_key")
        print("   CUSTOM_VISION_PREDICTION_KEY=your_prediction_key") 
        print("   CUSTOM_VISION_ENDPOINT=your_endpoint")
        return
    
    print("Custom Vision Training Workflow:")
    print("1. Create project")
    print("2. Create tags (door, window, etc.)")
    print("3. Upload training images with bounding box annotations") 
    print("4. Train model")
    print("5. Publish iteration")
    print("6. Make predictions")
    
    print("\nSample training data structure:")
    sample_data = create_sample_training_data()
    print(json.dumps(sample_data[0], indent=2))
    
    print("\nTraining Requirements:")
    print("- Minimum 15 images per object class")
    print("- Recommended 200+ images per class for best results")
    print("- Images should include various scales, rotations, lighting")
    print("- Precise bounding box annotations required")
    
    print("\nAdvantages of Custom Vision:")
    print("- Highly accurate for specific use cases")
    print("- Can detect custom architectural symbols")
    print("- Handles variations in style and drawing conventions")
    print("- Provides confidence scores and bounding boxes")
    
    print("\nNext Steps:")
    print("1. Collect training images of blueprints")
    print("2. Annotate with bounding boxes using Custom Vision portal")
    print("3. Train model using this script")
    print("4. Deploy for production use")

if __name__ == "__main__":
    demo_custom_vision_workflow() 