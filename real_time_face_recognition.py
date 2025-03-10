import os
import torch

# Check if model exists, otherwise download it
MODEL_PATH = "models/face_recognition_model.pth"

if not os.path.exists(MODEL_PATH):
    print("⚠ Model not found! Downloading...")
    os.system("python download_model.py")  # Automatically download model

# Now, proceed with the rest of the face recognition code
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
