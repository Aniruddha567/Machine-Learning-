from ultralytics import YOLO
import streamlit as st
from PIL import Image
import os
import shutil

# Load model
model = YOLO("train9/weights/best.pt")  # Change path if needed

# Title and description
st.title("⚽ Football Detection with YOLOv8")
st.write("Upload an image, and the model will detect football(s) in it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    # Save uploaded image to a temporary file
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run YOLOv8 prediction
    results = model.predict(source=img_path, save=True, conf=0.25)

    # Find latest output directory (e.g., runs/detect/predict4/)
    output_dir = results[0].save_dir

    # Get the predicted image file path
    output_image_path = os.path.join(output_dir, os.path.basename(img_path))

    # Check if the image exists and display
    if os.path.exists(output_image_path):
        st.image(output_image_path, caption='Detected Football(s)', use_container_width=True)
    else:
        st.warning("Output image not found.")
