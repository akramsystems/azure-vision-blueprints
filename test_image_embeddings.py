#!/usr/bin/env python3
"""
Test Azure Vision Image Embeddings

Testing the multimodal image API approach to get actual image embeddings
instead of just tags and objects.
"""

import os
from typing import List

def get_image_vector(path: str) -> List[float]:
    """Get image embeddings using Azure Vision multimodal API"""
    try:
        from azure.ai.inference import ImageEmbeddingsClient
        from azure.ai.inference.models import ImageEmbeddingInput
        from azure.core.credentials import AzureKeyCredential
        import base64
        
        print(f"🔍 Getting embeddings for: {path}")
        print("📡 Using azure.ai.inference library...")
        
        client = ImageEmbeddingsClient(
            endpoint=os.getenv("VISION_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("VISION_KEY")),
        )
        
        # Read and encode image as base64 data URL
        with open(path, "rb") as image_file:
            image_data = image_file.read()
            
        # Get file extension for the data URL
        file_extension = path.split('.')[-1].lower()
        
        # Encode as base64 data URL
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
        data_url = f"data:image/{file_extension};base64,{base64_encoded}"
        
        # Create ImageEmbeddingInput with the data URL
        embedding_input = ImageEmbeddingInput(image=data_url)
        
        resp = client.embed(input=[embedding_input])
        
        embedding = resp.data[0].embedding
        print(f"✅ Got embedding vector of length: {len(embedding)}")
        return embedding
        
    except ModuleNotFoundError:
        print("📡 azure.ai.inference not found, using REST API fallback...")
        # Fallback to raw REST
        import requests
        
        with open(path, "rb") as f:
            img = f.read()
            
        # Use the correct Azure Vision API endpoint format
        endpoint_base = os.getenv('VISION_ENDPOINT').rstrip('/')
        url = f"{endpoint_base}/computervision/retrieval:vectorizeImage?api-version=2024-02-01&model-version=2023-04-15"
              
        print(f"🌐 Making REST request to: {url}")
        
        r = requests.post(url,
                          headers={"Ocp-Apim-Subscription-Key": os.getenv("VISION_KEY"),
                                   "Content-Type": "application/octet-stream"},
                          data=img, timeout=30)
        r.raise_for_status()
        
        vector = r.json()["vector"]
        print(f"✅ Got embedding vector of length: {len(vector)}")
        return vector
    
    except Exception as e:
        print(f"❌ Error getting embeddings: {e}")
        return []

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    import math
    
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def test_image_embeddings():
    """Test getting embeddings for our door icon and blueprint"""
    print("🧪 TESTING AZURE VISION IMAGE EMBEDDINGS")
    print("=" * 50)
    
    # Check environment variables
    vision_endpoint = os.getenv("VISION_ENDPOINT")
    vision_key = os.getenv("VISION_KEY")
    
    if not vision_endpoint or not vision_key:
        print("❌ Missing environment variables:")
        print(f"   VISION_ENDPOINT: {'✅' if vision_endpoint else '❌'}")
        print(f"   VISION_KEY: {'✅' if vision_key else '❌'}")
        return
    
    print(f"🔑 Using endpoint: {vision_endpoint}")
    
    # Test files
    door_icon_path = "images/templates/door_icon.png"
    blueprint_path = "images/blueprints/blueprint_with_doors.png"
    
    # Check if files exist
    if not os.path.exists(door_icon_path):
        print(f"❌ Door icon not found: {door_icon_path}")
        return
    
    if not os.path.exists(blueprint_path):
        print(f"❌ Blueprint not found: {blueprint_path}")
        return
    
    print(f"📁 Testing files:")
    print(f"   Door icon: {door_icon_path}")
    print(f"   Blueprint: {blueprint_path}")
    
    # Get embeddings for door icon
    print("\n1️⃣ Getting door icon embeddings...")
    door_embedding = get_image_vector(door_icon_path)
    
    if not door_embedding:
        print("❌ Failed to get door icon embeddings")
        return
    
    # Get embeddings for blueprint
    print("\n2️⃣ Getting blueprint embeddings...")
    blueprint_embedding = get_image_vector(blueprint_path)
    
    if not blueprint_embedding:
        print("❌ Failed to get blueprint embeddings")
        return
    
    # Calculate similarity
    print("\n3️⃣ Calculating similarity...")
    similarity = calculate_cosine_similarity(door_embedding, blueprint_embedding)
    
    print(f"\n📊 RESULTS:")
    print(f"   Door embedding length: {len(door_embedding)}")
    print(f"   Blueprint embedding length: {len(blueprint_embedding)}")
    print(f"   Cosine similarity: {similarity:.4f}")
    
    # Show first few values of embeddings
    print(f"\n🔍 Sample embedding values:")
    print(f"   Door (first 5): {door_embedding[:5]}")
    print(f"   Blueprint (first 5): {blueprint_embedding[:5]}")
    
    # Test with a smaller region of the blueprint
    print("\n4️⃣ Testing with blueprint region...")
    
    # Create a small region from the blueprint to test
    import cv2
    blueprint_img = cv2.imread(blueprint_path)
    if blueprint_img is not None:
        # Extract a 100x100 region from top-left area
        region = blueprint_img[80:180, 120:220]  # Where we know there's a door
        region_path = "temp_region.png"
        cv2.imwrite(region_path, region)
        
        region_embedding = get_image_vector(region_path)
        if region_embedding:
            region_similarity = calculate_cosine_similarity(door_embedding, region_embedding)
            print(f"   Region similarity: {region_similarity:.4f}")
        
        # Cleanup
        if os.path.exists(region_path):
            os.remove(region_path)
    
    print(f"\n✅ Embedding test complete!")
    print(f"💡 If similarity > 0.8, embeddings are working well!")
    print(f"💡 If similarity < 0.5, we may need to adjust the approach")

if __name__ == "__main__":
    test_image_embeddings() 