# Skin Lesion Segmentation using U-Net

A deep learning-based computer vision project for **pixel-level skin lesion segmentation** using a U-Net architecture. The model predicts the lesion region from dermoscopic skin images and displays the original image, predicted binary mask, and overlay visualization through a Streamlit web app.

> This project is for educational computer vision purposes only. It is not a medical diagnosis tool.

---

## Links

- Live App: https://huggingface.co/spaces/asibul60/skin-lesion-segmentation-unet
- Dataset: https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation

## Project Overview

Skin lesion segmentation is a semantic segmentation task where the goal is to identify the exact pixels that belong to a lesion region in a skin image.

Unlike image classification or object detection, segmentation provides a pixel-level output mask.

| Task | Output |
|---|---|
| Image Classification | Class label |
| Object Detection | Bounding box |
| Image Segmentation | Pixel-level mask |

This project uses a U-Net model to segment lesion regions from dermoscopic images.

---

## Demo Features

The Streamlit app allows users to:

- Upload a skin lesion image
- Predict the lesion region
- View the original image
- View the predicted segmentation mask
- View the mask overlay on the original image
- Adjust the mask threshold using a slider
- See predicted lesion area percentage
- Download the predicted mask and overlay image

---

## Model Architecture

This project uses a stronger U-Net architecture with an encoder-decoder structure and skip connections.

Architecture filter progression:

```text
Encoder: 32 → 64 → 128 → 256
Bridge: 512
Decoder: 256 → 128 → 64 → 32

Model size:

Total parameters: 7,760,097
Input shape: 128 × 128 × 3
Output shape: 128 × 128 × 1

The final output uses a sigmoid activation for binary segmentation.

Dataset

The model was trained using a subset of the ISIC 2018 Challenge Task 1 skin lesion segmentation dataset.

Dataset source:

https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation

Training setup used in this project:

Training image-mask pairs: 1000
Image size: 128 × 128
Train-validation split: 80/20
Training images: 800
Validation images: 200

The dataset and image files are not included in this repository because of size limitations.

Training Details

The model was trained in Google Colab using GPU acceleration.

Loss function:

Binary Cross-Entropy + Dice Loss

Optimizer:

Adam optimizer
Learning rate: 1e-4

Training configuration:

Batch size: 8
Epochs: 15
Early stopping: enabled
Best model checkpoint: enabled

Best validation result:

Validation Dice Coefficient: 0.8902
Validation Accuracy: 0.9529
Validation Loss: 0.2416
Project Structure
skin-lesion-segmentation/
│
├── app/
│   └── app.py
│
├── data/
│   ├── images/
│   │   └── .gitkeep
│   └── masks/
│       └── .gitkeep
│
├── models/
│   └── unet_skin_lesion.keras   # Not included in GitHub repo
│
├── outputs/
│   └── .gitkeep
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── requirements.txt
├── .gitignore
└── README.md
Installation

Clone the repository:

git clone https://github.com/asibul-islam/skin-lesion-segmentation-unet.git
cd skin-lesion-segmentation-unet

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Running the App

Place the trained model file here:

models/unet_skin_lesion.keras

Then run:

streamlit run app/app.py

Open the local Streamlit URL in your browser and upload a dermoscopic skin lesion image.

## Deployment

The app is deployed on Hugging Face Spaces using Streamlit.

Deployment files include:

streamlit_app.py
requirements.txt
src/utils.py
models/unet_skin_lesion.keras

Running Prediction Script

To run prediction on a sample image:

python src/predict.py

The script saves outputs to:

outputs/predicted_mask.png
outputs/overlay.png
outputs/comparison.png
Important Notes

This model works best with dermoscopic ISIC-style skin lesion images.

It may not perform well on regular phone camera images because phone images often have different lighting, distance, skin texture, shadows, and image quality compared to dermoscopic dataset images.

This project should not be used for medical decision-making.

Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy
Matplotlib
Streamlit
Pillow
Google Colab GPU
Computer Vision Concepts Demonstrated
Semantic segmentation
Pixel-level classification
U-Net architecture
Encoder-decoder networks
Skip connections
Binary mask prediction
Dice coefficient
BCE-Dice loss
Image preprocessing
Segmentation overlay visualization
Streamlit model deployment
Future Improvements
Train on the full ISIC 2018 dataset
Add a separate test set evaluation
Add IoU metric during training
Compare U-Net with DeepLabV3+ or Attention U-Net
Add data augmentation
Add Grad-CAM or explainability visualizations
Disclaimer

This project is for educational and portfolio purposes only. It is not intended to diagnose, treat, or provide medical advice. Always consult a qualified medical professional for skin-related concerns.
