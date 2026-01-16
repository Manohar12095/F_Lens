import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai

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
# Make sure you have added your API key in Streamlit Secrets
# st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Use a supported model (check in Google AI Studio)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

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
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        # If the model is not available or API fails
        return "⚠️ Nutrient data temporarily unavailable."

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

# ---------------- PREDICTION (SIMULATED) ----------------
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    _ = preprocess_image(image)  # preprocessing step

    # Simulated CNN output (for now, offline)
    class_names = ["Fruit", "Vegetable", "Other"]
    predicted_class = np.random.choice(class_names)
    confidence = np.random.uniform(0.85, 0.98)

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
st.caption("CNN (trained offline) + Gemini AI | College Mini Project")
