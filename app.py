# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model and labels
model = tf.keras.models.load_model("waste_model.h5")
with open("labels.txt", "r") as f:
    class_names = f.read().splitlines()

# Streamlit UI
st.set_page_config(page_title="Waste Classifier", page_icon="🗑️")
st.title("🧠 Waste Classification App")
st.write("Upload an image of waste and the model will predict its type.")

# File uploader
uploaded_file = st.file_uploader("📷 Upload Waste Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    preds = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.subheader("🧠 Prediction Result:")
    st.success(f"**{predicted_class.capitalize()}** ({confidence:.2f}% confidence)")
