import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image

# load model
model = tf.keras.models.load_model("model.h5")

# class names
class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# explanation
defect_info = {
    "crazing": "Surface cracks due to stress caused by improper cooling.",
    "inclusion": "Foreign particles trapped in metal during manufacturing.",
    "patches": "Uneven coating due to poor surface treatment.",
    "pitted_surface": "Small holes caused by corrosion or gas bubbles.",
    "rolled-in_scale": "Oxide scale pressed into metal during rolling.",
    "scratches": "Surface damage caused by friction or handling."
}

# suggestion
suggestion = {
    "crazing": "Improve cooling process and reduce thermal stress.",
    "inclusion": "Use high-quality raw materials and better filtration.",
    "patches": "Improve coating process and surface cleaning.",
    "pitted_surface": "Control corrosion and improve environment conditions.",
    "rolled-in_scale": "Enhance descaling process before rolling.",
    "scratches": "Handle materials carefully and reduce friction."
}

# title
st.title("🏭 AI Steel Defect Detection System")

st.markdown("Upload a steel surface image to detect defect and get AI-based analysis.")

# upload
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    # preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # prediction
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # results section
    st.subheader("🔍 Prediction Result")

    st.success(f"Defect Type: {predicted_class.upper()}")
    st.write(f"Confidence Score: {confidence:.2f}")

    st.subheader("🧠 AI Analysis")
    st.info(defect_info[predicted_class])

    st.subheader("🛠 Recommendation")
    st.warning(suggestion[predicted_class])

    # business insight (IMPORTANT 🔥)
    st.subheader("📊 Business Impact")
    st.write(
        "This defect may affect product quality and increase rejection rate. "
        "Early detection helps reduce cost and improve production efficiency."
    )