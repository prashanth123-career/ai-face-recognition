import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import os
import requests
import time
import sys

# --- CareerUpskillers Branding ---
st.set_page_config(page_title="AI Face Recognition - CareerUpskillers", page_icon="🤖")
st.markdown(
    """
    <h1 style="text-align: center; color: #ff5733;">🤖 AI Face Recognition System</h1>
    <h3 style="text-align: center;">🚀 Developed by <a href='https://www.careerupskillers.com' target='_blank'>CareerUpskillers</a></h3>
    <p style="text-align: center;">📞 Contact: <a href="https://wa.me/917975931377" target="_blank">WhatsApp 917975931377</a></p>
    <hr style="border:1px solid #ff5733;">
    """,
    unsafe_allow_html=True
)

# --- Model Download Setup ---
MODEL_URL = "https://drive.google.com/uc?id=1DF72bjGVnN6iNJrv6B18XulSdiCHBYtR"
MODEL_PATH = "models/face_recognition_model.pth"

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Download the model if it does not exist
if not os.path.exists(MODEL_PATH):
    st.info("🔄 Downloading model... Please wait.")
    try:
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            st.success("✅ Model downloaded successfully!")
            st.info("Please refresh the page to load the model.")
            st.stop()
        else:
            st.error("❌ Failed to download model. Please check the URL.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Exception during download: {e}")
        st.stop()

# --- Load the Trained Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model architecture (must match the training script)
import torch.nn as nn

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set the dataset folder for class names
DATASET_PATH = "data_face_recognition/known"  # Folder that contains known face images
if os.path.exists(DATASET_PATH):
    classes = sorted(os.listdir(DATASET_PATH))
else:
    classes = ["Unknown"]

num_classes = len(classes)

model = FaceRecognitionModel(num_classes=num_classes).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    st.error(f"❌ Model file '{MODEL_PATH}' not found!")
    st.stop()

# --- Define Image Transformations ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Streamlit UI for Image Upload and Prediction ---
st.header("📤 Upload an Image for Face Recognition")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Prepare image for model
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    
    label = classes[predicted_class] if predicted_class < len(classes) else "Unknown"
    st.success(f"✅ Prediction: {label}")

# --- Footer Branding ---
st.markdown(
    """
    <hr style="border:1px solid #ff5733;">
    <h4 style="text-align: center;">🚀 AI Starter Kit by CareerUpskillers</h4>
    <p style="text-align: center;">For more AI tools, visit <a href='https://www.careerupskillers.com' target='_blank'>CareerUpskillers</a></p>
    """,
    unsafe_allow_html=True
)
