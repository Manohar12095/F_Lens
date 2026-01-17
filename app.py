import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai
from tensorflow.keras.models import load_model
import gdown
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🍎",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("🍎 Fruit & Vegetable Recognition System")
st.markdown("Upload an image **or** capture using your camera")

# ---------------- GEMINI CONFIG ----------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ✅ IMPORTANT: Use model name exactly like this
gemini_model = genai.GenerativeModel("models/gemini-1.5-pro")

def gemini_analysis(item):
    prompt = f"""
    Food Item: {item}

    Give:
    - Nutrient content
    - Health benefits
    - Vitamins & minerals
    - Daily uses
    - Is it good for human health?

    Explain in simple bullet points.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception:
        return "⚠️ Nutrient data temporarily unavailable."

# ---------------- MODEL DOWNLOAD ----------------
MODEL_PATH = "model_F.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading trained CNN model..."):
        gdown.download(
            "https://drive.google.com/uc?id=1aC4cEJyUKbAQGk79SF9oyhRg4yISTY1I",
            MODEL_PATH,
            quiet=False
        )

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)

# ⚠️ MUST MATCH TRAINING ORDER
class_names = ["Fruit", "Vegetable", "Other"]

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- UI INPUT ----------------
input_method = st.radio(
    "Choose input method",
    ["Upload Image", "Use Camera"]
)

image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file)

if input_method == "Use Camera":
    camera_image = st.camera_input("Capture image")
    if camera_image:
        image = Image.open(camera_image)

# ---------------- PREDICTION ----------------
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    predicted_class = class_names[predicted_index]

    st.success(f"### 🧠 Detected: {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.progress(confidence)

    # ---------------- GEMINI AI ----------------
    with st.spinner("🤖 Fetching nutrient details..."):
        analysis = gemini_analysis(predicted_class)

    st.subheader("🥗 Nutrient & Health Information")
    st.write(analysis)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("CNN + Gemini AI | College Mini Project")
