import cv2
import numpy as np


IMG_SIZE = 128


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = mask / 255.0
    mask = (mask > 0.5).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    intersection = np.sum(y_true * y_pred)
    dice = (2.0 * intersection + smooth) / (
        np.sum(y_true) + np.sum(y_pred) + smooth
    )
    return dice


def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou