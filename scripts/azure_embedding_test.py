import os
from azure.ai.inference import ImageEmbeddingsClient, EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
import numpy as np

# Replace with your actual endpoint and key
endpoint = os.getenv('VISION_ENDPOINT')
key = os.getenv('VISION_KEY')
model_name = "cohere-embed-v3-english" # Example model name

client = ImageEmbeddingsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key),
)

print(client.get_model_info())

# # Replace with your image data (base64 encoded PNG)
# image_data = open("images/templates/door_icon.png", "rb").read()

# try:
#     response = client.embed(input=image_data)
#     embedding = response.data[0].embedding

#     # Print the embedding
#     print("Embedding:", embedding)
#     print("Shape:", np.asarray(embedding).shape)

# except Exception as e:
#     print(f"Error generating embedding: {e}")