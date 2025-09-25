import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Face Recognition Attendance", layout="centered")

# --------------------------
# Model setup
# --------------------------
MODEL_PATH = "face_recognition_model.h5"

# Use DRIVE_FILE_ID from environment variable for security (optional)
DRIVE_FILE_ID = os.getenv("DRIVE_FILE_ID", "YOUR_DRIVE_FILE_ID")  # <-- Replace with your Google Drive file ID

# Download model if not present
if not os.path.exists(MODEL_PATH):
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        st.info("Downloading model...")
        gdown.download(url, MODEL_PATH, quiet=False)
    except Exception as e:
        st.error(f"Model not found and download failed: {e}")
        st.stop()

# Load Keras model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("🎓 Face Recognition Attendance System")
st.write("Upload a face image to detect student ID.")

# --------------------------
# File upload
# --------------------------
uploaded_file = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Could not read the image. Make sure it's a valid image file.")
    else:
        # Preprocess
        img_resized = cv2.resize(img, (100, 100))
        img_norm = img_resized / 255.0
        img_input = img_norm.reshape(1, 100, 100, 1)

        # Prediction
        pred = model.predict(img_input)
        confidence = float(np.max(pred))
        predicted_class = int(np.argmax(pred)) + 1  # 1–31

        # Display uploaded image
        st.image(img_resized, caption="Uploaded (preprocessed)", width=250, channels="GRAY")

        # Threshold for unknown detection
        threshold = 0.60
        if confidence < threshold:
            st.error(f"⚠️ Unknown person (highest confidence {confidence*100:.2f}%)")
        else:
            st.success(f"✅ Predicted Student ID: {predicted_class} (Confidence: {confidence*100:.2f}%)")

