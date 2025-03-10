import os
import requests

# Google Drive Direct Download Link
MODEL_URL = "https://drive.google.com/uc?id=1DF72bjGVnN6iNJrv6B18XulSdiCHBYtR"

# Define the model directory and file name
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognition_model.pth")

# Ensure the "models" directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def download_model():
    """Download the face recognition model from Google Drive."""
    print("📥 Downloading model file... Please wait.")
    response = requests.get(MODEL_URL, stream=True)

    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"✅ Model downloaded successfully and saved to {MODEL_PATH}")
    else:
        print("❌ Failed to download model. Please check the URL.")

# Run the function
if not os.path.exists(MODEL_PATH):
    download_model()
else:
    print(f"✅ Model already exists at {MODEL_PATH}")
