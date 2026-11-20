import os
os.environ["TF_CPP_MIN_LOC_LEVEL"]="3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="1"

import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# --
# 1. Load models
# --

@st.cache_resource
def load_models():
    return {
        "organ": tf.keras.models.load_model("Model/Organ_classifier_256.keras"),
        "Brain": tf.keras.models.load_model("Model/Brain_classifier_256.keras"),
        "Lung": tf.keras.models.load_model("Model/Lung_classifier_256.keras"),
        "Breast": tf.keras.models.load_model("Model/Breast_Cancer_classifier_256.keras"),
        "Eye": tf.keras.models.load_model("Model/Eye_classifier_256.keras"),
        "Skin": tf.keras.models.load_model("Model/Skin_classifier_256.keras"),
    }

models = load_models()


# --
# 2. Class names
# --

organ_classes = [ "Brain", "Breast", "Lung",  "eye","skin"]

disease_classes = {
    "Brain": [
        "Glioma",
        "Meningioma",
        "No Tumor",
        "Pituitary"
    ],

    "Breast": [
        "Benign",
        "Malignant"
    ],

    "Skin": [
        "Benign",
        "Malignant"
    ],

    "Lung": [
        "Colon Adenocarcinoma",
        "Colon Benign",
        "Lung Adenocarcinoma",
        "Lung Benign",
        "Lung Squamous Cell Carcinoma"
    ],

    "Eye": [
        "Age Related Degeneration",
        "Cataract",
        "Diabetic Retinopathy",
        "Glaucoma",
        "Hypertensive Retinopathy",
        "Myopia",
        "Normal"
    ]
}

IMG_SIZE = (256, 256)

# -
# Image Preprocessing
# -
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype("float32")
    img = np.expand_dims(img, axis=0)
    return img

# opening of images
image = Image.open("./sample_images/G_367.jpg")   # put your image path here.
img = preprocess_image(image)

# Organ Prediction 
organ_prediction = models["organ"].predict(img, verbose=0)

organ_index = np.argmax(organ_prediction)
predicted_organ = organ_classes[organ_index]
organ_confidence = (organ_prediction[0][organ_index] * 100)

#  Disease Prediction 
disease_model = models[predicted_organ]
disease_prediction = disease_model.predict(img,verbose=0)
disease_index = np.argmax(disease_prediction)
predicted_disease = disease_classes[predicted_organ][disease_index]
# Disease confidence
disease_confidence = ( disease_prediction[0][disease_index] * 100)
print(f"Disease is: {predicted_disease} with confidence: {disease_confidence}")