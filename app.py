import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision import models
import torch.nn as nn
import numpy as np

# 🎉 CareerUpskillers Branding
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

# Load trained model
MODEL_PATH = "models/face_recognition_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model
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

# Load dataset classes
DATASET_PATH = "data_face_recognition/train"
classes = os.listdir(DATASET_PATH) if os.path.exists(DATASET_PATH) else ["Unknown"]

# Load the trained model
num_classes = len(classes)
model = FaceRecognitionModel(num_classes=num_classes).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    st.error(f"❌ Model file '{MODEL_PATH}' not found!")
    st.stop()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Upload image
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict the class
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output).item()

    label = classes[predicted_class] if predicted_class < len(classes) else "Unknown"
    st.success(f"✅ Prediction: {label}")

# Footer
st.markdown(
    """
    <hr style="border:1px solid #ff5733;">
    <h4 style="text-align: center;">🚀 AI Starter Kit by CareerUpskillers</h4>
    <p style="text-align: center;">Get More AI Tools at <a href='https://www.careerupskillers.com' target='_blank'>www.careerupskillers.com</a></p>
    """,
    unsafe_allow_html=True
)
