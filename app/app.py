import os
import sys
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import IMG_SIZE


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "unet_skin_lesion.keras")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    image = np.array(image)

    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    normalized = normalized.astype(np.float32)

    input_image = np.expand_dims(normalized, axis=0)

    return image, input_image


def predict_mask(model, input_image, threshold):
    pred = model.predict(input_image)[0]
    mask = (pred > threshold).astype(np.uint8) * 255
    return mask


def create_overlay(original_image, mask):
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

    overlay = original_image.copy()
    overlay[mask_resized.squeeze() > 0] = [255, 0, 0]

    blended = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
    return blended


st.set_page_config(
    page_title="Skin Lesion Segmentation",
    layout="wide"
)

st.title("Skin Lesion Segmentation using U-Net")

st.write(
    "Upload a dermoscopic skin lesion image and the model will predict the lesion region at pixel level."
)

st.warning(
    "This project is for educational computer vision segmentation only. It is not a medical diagnosis tool."
)

st.markdown(
    """
    **Test image source:** This model works best with dermoscopic ISIC-style images.  
    You can download sample-compatible images from the Kaggle ISIC 2018 Task 1 Segmentation dataset:  
    https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation
    """
)

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

threshold = st.slider(
    "Mask threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Higher threshold usually makes the predicted mask smaller. Try 0.7 or 0.8 if the model over-segments."
)

if uploaded_file is not None:
    model = load_model()

    original_image, input_image = preprocess_image(uploaded_file)
    predicted_mask = predict_mask(model, input_image, threshold)
    overlay = create_overlay(original_image, predicted_mask)

    lesion_pixels = np.sum(predicted_mask > 0)
    total_pixels = predicted_mask.shape[0] * predicted_mask.shape[1]
    lesion_area_percentage = (lesion_pixels / total_pixels) * 100

    st.metric(
        label="Predicted Lesion Area",
        value=f"{lesion_area_percentage:.2f}% of image"
    )

    if lesion_area_percentage < 5:
        st.info(
            "The predicted mask covers a small part of the image. If the lesion is missing, try lowering the threshold.")
    elif lesion_area_percentage > 60:
        st.warning(
            "The predicted mask covers a large part of the image. Try increasing the threshold to reduce over-segmentation.")
    else:
        st.success("The predicted mask size looks visually reasonable. Please inspect the overlay carefully.")

    # Prepare files for download
    mask_image = Image.fromarray(predicted_mask.squeeze())
    overlay_image = Image.fromarray(overlay)

    mask_download_path = os.path.join(BASE_DIR, "outputs", "app_predicted_mask.png")
    overlay_download_path = os.path.join(BASE_DIR, "outputs", "app_overlay.png")

    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

    mask_image.save(mask_download_path)
    overlay_image.save(overlay_download_path)

    with open(mask_download_path, "rb") as file:
        st.download_button(
            label="Download Predicted Mask",
            data=file,
            file_name="predicted_mask.png",
            mime="image/png"
        )

    with open(overlay_download_path, "rb") as file:
        st.download_button(
            label="Download Overlay Image",
            data=file,
            file_name="overlay.png",
            mime="image/png"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

    with col2:
        st.subheader("Predicted Mask")
        st.image(predicted_mask, use_container_width=True, clamp=True)

    with col3:
        st.subheader("Overlay")
        st.image(overlay, use_container_width=True)

    st.success("Segmentation completed successfully.")
else:
    st.info("Please upload an image to start segmentation.")