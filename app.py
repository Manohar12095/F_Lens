import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import google.generativeai as genai

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🍎",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "f&v_product_model.h5"   # put model in same folder
model = load_model(MODEL_PATH)

# ⚠️ IMPORTANT:
# These class names MUST match your dataset folders exactly
class_names = ['Fruit', 'Vegetable', 'Other']

# ---------------- GEMINI CONFIG ----------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-pro")

def gemini_analysis(item):
    prompt = f"""
    Food Item: {item}

    1. Is it a fruit or vegetable?
    2. Health benefits
    3. Nutritional value
    4. Uses in daily life
    5. Is it good for human health?
    Explain in simple bullet points.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
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

# ---- Upload from computer ----
if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file)

# ---- Camera input ----
if input_method == "Use Camera":
    camera_image = st.camera_input("Capture image")
    if camera_image:
        image = Image.open(camera_image)

# ---------------- PREDICTION ----------------
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)

    confidence = float(np.max(prediction))
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"### Detected: {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.progress(confidence)

    # ---------------- GEMINI AI OUTPUT ----------------
    with st.spinner("Analyzing with Gemini AI..."):
        analysis = gemini_analysis(predicted_class)

    st.subheader("🤖 Gemini AI Health Analysis")
    st.write(analysis)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("CNN + Gemini AI | College Mini Project")
