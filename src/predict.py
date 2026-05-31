import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import read_image


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_SIZE = 128


def predict_mask(image_path, model_path=None):
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "models", "unet_skin_lesion.keras")

    model = tf.keras.models.load_model(model_path, compile=False)

    image = read_image(image_path)
    input_image = np.expand_dims(image, axis=0)

    predicted_mask = model.predict(input_image, verbose=1)[0]
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

    return predicted_mask


def create_overlay(original_image, predicted_mask):
    mask_resized = cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0]))

    overlay = original_image.copy()
    overlay[mask_resized.squeeze() > 0] = [255, 0, 0]

    blended = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)

    return blended


def main():
    image_path = os.path.join(BASE_DIR, "data", "images", "ISIC_0000000.jpg")

    output_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    original_bgr = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    predicted_mask = predict_mask(image_path)
    overlay = create_overlay(original_rgb, predicted_mask)

    cv2.imwrite(os.path.join(output_dir, "predicted_mask.png"), predicted_mask)
    cv2.imwrite(
        os.path.join(output_dir, "overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask.squeeze(), cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    comparison_path = os.path.join(output_dir, "comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    print(f"Saved predicted mask to {output_dir}/predicted_mask.png")
    print(f"Saved overlay to {output_dir}/overlay.png")
    print(f"Saved comparison to {comparison_path}")


if __name__ == "__main__":
    main()