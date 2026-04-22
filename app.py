import streamlit as st
from PIL import Image
import numpy as np

# Title
st.title("Steel Defect Detection 🔍")

# Description
st.write("Upload a steel surface image and detect defects using AI (Demo Version)")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    
    # Open Image
    img = Image.open(uploaded_file)
    
    # Show Image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Dummy prediction list
    defects = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"
    ]
    
    # Random prediction
    prediction = np.random.choice(defects)
    
    # Show Result
    st.success(f"Prediction: {prediction}")