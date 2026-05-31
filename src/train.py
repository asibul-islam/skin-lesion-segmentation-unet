import os
import sys
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import build_unet
from utils import read_image, read_mask


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 10


def load_dataset(image_dir, mask_dir):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    images = []
    masks = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        image_id = image_name.replace(".jpg", "")
        mask_path = os.path.join(mask_dir, image_id + "_segmentation.png")

        if not os.path.exists(mask_path):
            continue

        image = read_image(image_path)
        mask = read_mask(mask_path)

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)


def dice_coef_tf(y_true, y_pred):
    smooth = 1e-6

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

    return dice


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef_tf(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)

    return bce + dice_loss(y_true, y_pred)


def main():
    image_dir = os.path.join(BASE_DIR, "data", "images")
    mask_dir = os.path.join(BASE_DIR, "data", "masks")
    model_path = os.path.join(BASE_DIR, "models", "unet_skin_lesion.keras")

    print("Loading dataset...")
    print("Image directory:", image_dir)
    print("Mask directory:", mask_dir)

    X, y = load_dataset(image_dir, mask_dir)

    print("Images shape:", X.shape)
    print("Masks shape:", y.shape)

    if len(X) == 0:
        print("No image-mask pairs found.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=bce_dice_loss,
        metrics=[
            "accuracy",
            dice_coef_tf
        ]
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()