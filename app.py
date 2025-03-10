import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import os
import requests
import time

# Google Drive Model Link
MODEL_URL = "https://drive.google.com/uc?id=1DF72bjGVnN6iNJrv6B18XulSdiCHBYtR"
MODEL_PATH = "models/face_recognition_model.pth"

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Download model if not present
if not os.path.exists(MODEL_PATH):
    st.warning("🔄 Downloading Model... Please wait.")
    response = requests.get(MODEL_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        st.success("✅ Model Downloaded Successfully!")
        time.sleep(2)
        st.experimental_rerun()
    else:
        st.error("❌ Failed to download model.")

# Load Model (Dummy for Streamlit Display)
st.title("🤖 AI Face Recognition - CareerUpskillers")
st.write("🚀 Developed by CareerUpskillers | 📞 917975931377 | 🌐 www.careerupskillers.com")

# Upload Image for Testing
uploaded_file = st.file_uploader("📸 Upload an Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success("✅ Image Uploaded Successfully!")
