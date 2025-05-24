import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Dataset directory paths
dataset_dir_path = "cat-breeds"

# Load model
model = load_model('cat_breed_model.keras')

# Load breed names from breeds directory
breed_labels = list(os.listdir(dataset_dir_path))

st.title("Cat Breed Classifier")

uploaded_file = st.file_uploader("Upload a cat image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(128, 128))
    st.image(img, caption='Uploaded Image', use_container_width=True)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    prediction = model.predict(x)
    predicted_class = breed_labels[np.argmax(prediction)]
    
    st.write(f"Predicted Cat Breed: **{predicted_class}**")
