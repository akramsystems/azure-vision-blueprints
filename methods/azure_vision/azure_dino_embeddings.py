import base64, json, requests, os, numpy as np

# 1.  encode the local file
with open("images/templates/door_icon_2.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "input_data": {
        "columns": ["image"],
        "index": [0],
        "data": [[b64]]          # list-of-rows, each row is list-of-cols
    }
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.environ['VISION_DINO_V2_KEY']}"
}

endpoint = os.environ.get('VISION_DINO_V2_ENDPOINT', 'https://quoto-measurement-detecti-wsmaj.eastus.inference.ml.azure.com/score')
resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
resp.raise_for_status()

vec = np.asarray(resp.json()[0]["image_features"])
print("Vector shape:", vec.shape)      # (768,)
print("Vector:", vec)
