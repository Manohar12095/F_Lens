import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import google.generativeai as genai
import gdown
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🍎",
    layout="centered"
)

# ---------------- MODEL DOWNLOAD ----------------
MODEL_PATH = "f&v_product_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model..."):
        url = "https://drive.google.com/uc?id=1zFL5xeBEjXfGRpHnJZjvXJjlEv7zUi_p"
        gdown.download(url, MODEL_PATH, quiet=False)

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)
st.success("✅ Model loaded successfully")

# ⚠️ MUST match model output layer
class_names = ['Fruit', 'Vegetable', 'Other']

# ---------------- GEMINI CONFIG ----------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-pro")

def gemini_analysis(item):
    prompt = f"""
    Food Item: {item}

    • Is it a fruit or vegetable?
    • Key health benefits
    • Nutritional value
    • Daily uses
    • Is it good for human health?

    Explain in simple bullet points.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- UI ----------------
st.title("🍎 Fruit & Vegetable Recognition System")
st.markdown("Upload an image **or** capture using your camera")

input_method = st.radio(
    "Choose input method",
    ["Upload Image", "Use Camera"]
)

image = None

# ---- Upload Image ----
if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file)

# ---- Camera Input ----
if input_method == "Use Camera":
    camera_image = st.camera_input("Capture image")
    if camera_image:
        image = Image.open(camera_image)

# ---------------- PREDICTION ----------------
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    predicted_class = class_names[predicted_index]

    st.success(f"### 🧠 Detected: {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.progress(confidence)

    # ---------------- GEMINI AI ----------------
    with st.spinner("🤖 Analyzing with Gemini AI..."):
        analysis = gemini_analysis(predicted_class)

    st.subheader("🔍 Gemini AI Health Analysis")
    st.write(analysis)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("CNN + Gemini AI | College Mini Project")
