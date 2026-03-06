import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ----------------------------
# Page settings
# ----------------------------
st.set_page_config(
    page_title="Package Quality Detector",
    page_icon="📦",
    layout="centered"
)

# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = "package_quality_model.h5"
model = load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

CLASS_LABELS = {0: "Damaged", 1: "Good"}

# ----------------------------
# Header
# ----------------------------
st.title("📦 Package Quality Detector")

st.markdown(
"""
Upload a **package image** and the AI model will predict whether the package is:

✅ **Good**  
❌ **Damaged**
"""
)

st.divider()

# ----------------------------
# Upload image
# ----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload package image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# Prediction
# ----------------------------
if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Package Image", use_container_width=True)

    # Preprocess image
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = CLASS_LABELS[1]
        confidence = prediction
        st.success("✅ Package Status: GOOD")
    else:
        label = CLASS_LABELS[0]
        confidence = 1 - prediction
        st.error("❌ Package Status: DAMAGED")

    confidence_percent = confidence * 100

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Prediction", label)

    with col2:
        st.metric("Confidence", f"{confidence_percent:.2f}%")

    st.progress(int(confidence_percent))

st.divider()

st.caption("AI Model: ResNet50 Transfer Learning | Built with Streamlit")