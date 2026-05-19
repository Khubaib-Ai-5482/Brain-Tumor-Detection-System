import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# ---------------- DEVICE ----------------
device = torch.device("cpu")

# ---------------- MODEL ----------------
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

# ✅ FIXED PATH (IMPORTANT)
BASE_DIR = r"C:\Users\user\Videos\DATA ANALYSIS & SCIENCE PORTFOLIO\AI-Powered Brain Tumor Detection & Segmentation System"
MODEL_PATH = os.path.join(BASE_DIR, "tumor_model.pth")

# Load model safely
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# ---------------- UI ----------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Powered Brain Tumor Detection System")
st.subheader("Upload MRI Image to Detect Tumor Type")

col1, col2 = st.columns([2, 1])

with col1:
    file = st.file_uploader("Choose an MRI image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1) * 100
            _, pred = torch.max(outputs, 1)
            label = classes[pred.item()]

        st.success(f"Prediction: **{label}**")

        st.markdown("### Confidence Scores (%)")
        for i, cls in enumerate(classes):
            st.write(f"{cls}: {probs[0][i].item():.2f}%")

        df = pd.DataFrame({
            "Tumor Type": classes,
            "Confidence (%)": probs[0].cpu().numpy()
        })

        st.bar_chart(df.set_index("Tumor Type"))

with col2:
    st.markdown("### About Tumor Types")
    st.write("Glioma: Tumor in glial cells")
    st.write("Meningioma: Tumor in meninges")
    st.write("No Tumor: Healthy MRI scan")
    st.write("Pituitary: Tumor in pituitary gland")

st.markdown("---")
st.write("© 2026 Khubaib | Brain Tumor Detection System")