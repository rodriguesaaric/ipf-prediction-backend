# ===========================
# FVC Prediction Backend API
# ===========================

import os
import io
import numpy as np
import requests
import cv2
import pydicom
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3

# ======================================
# CONFIG
# ======================================

MODEL_URL = os.environ.get("MODEL_URL")
WEIGHTS_PATH = "/tmp/fvc_model_weights.weights.h5"

IMG_SIZE = 256
TABULAR_FEATURES = [
    'Weeks', 'Age', 'FVC',
    'Sex_male',
    'SmokingStatus_Ex-smoker',
    'SmokingStatus_Never Smoker'
]


# ======================================
# DOWNLOAD MODEL WEIGHTS
# ======================================

def download_model():
    """Download weights from GitHub Release into /tmp (Render allowed)."""
    if os.path.exists(WEIGHTS_PATH):
        print("✔ Weights already downloaded.")
        return WEIGHTS_PATH

    if MODEL_URL is None:
        raise RuntimeError("MODEL_URL env variable not set.")

    print(f"⬇ Downloading model weights from: {MODEL_URL}")

    r = requests.get(MODEL_URL)
    if r.status_code != 200:
        raise RuntimeError("Failed to download model weights.")

    with open(WEIGHTS_PATH, "wb") as f:
        f.write(r.content)

    print("✔ Model weights downloaded successfully.")
    return WEIGHTS_PATH


# ======================================
# CUSTOM LOSS (needed to load model)
# ======================================

def laplace_log_likelihood(y_true, y_pred):
    mu = y_pred[:, 0]
    log_sigma = y_pred[:, 1]
    sigma = K.exp(log_sigma)

    y_true_fvc = y_true[:, 0]
    loss = (K.abs(y_true_fvc - mu) / sigma) + K.log(2 * sigma)

    return K.mean(loss)


# ======================================
# BUILD MODEL ARCHITECTURE
# ======================================

def build_model():
    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image_input")

    # Convert 1-channel into 3-channel for EfficientNet
    x = Concatenate()([img_input, img_input, img_input])

    CNN = EfficientNetB3(
        weights=None,
        include_top=False,
        input_tensor=x
    )
    CNN.trainable = False

    img_features = CNN.output
    img_features = GlobalAveragePooling2D()(img_features)
    img_features = Dense(64, activation="relu")(img_features)

    tab_input = Input(shape=(len(TABULAR_FEATURES),), name="tabular_input")
    tab_features = Dense(32, activation="relu")(tab_input)

    merged = Concatenate()([img_features, tab_input])
    merged = Dense(64, activation="relu")(merged)

    mu = Dense(1, name="mu")(merged)
    sigma = Dense(1, name="log_sigma")(merged)

    output = Concatenate(axis=-1)([mu, sigma])

    model = Model(inputs=[img_input, tab_input], outputs=output)
    model.compile(optimizer="adam", loss=laplace_log_likelihood)

    return model


# ======================================
# LOAD MODEL ONCE
# ======================================

MODEL = None

def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    weights_file = download_model()
    MODEL = build_model()
    MODEL.load_weights(weights_file)

    print("✔ Model fully loaded and ready.")

    return MODEL


# ======================================
# DICOM PREPROCESSING
# ======================================

def preprocess_dicom(dicom_bytes: bytes):
    dcm = pydicom.dcmread(io.BytesIO(dicom_bytes))
    img = dcm.pixel_array.astype(np.int16)

    # Apply HU conversion
    if "RescaleSlope" in dcm and "RescaleIntercept" in dcm:
        img = img * dcm.RescaleSlope + dcm.RescaleIntercept

    # Windowing (matching your training)
    MIN_HU, MAX_HU = -1000, -400
    img = np.clip(img, MIN_HU, MAX_HU)
    img = (img - MIN_HU) / (MAX_HU - MIN_HU)
    img = (img * 255).astype(np.uint8)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    # Expand dims -> (1, 256, 256, 1)
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# ======================================
# TABULAR PREPROCESSING
# ======================================

def prepare_tabular_data(age, sex, smoking, weeks, fvc):
    df = pd.DataFrame([{
        "Weeks": weeks,
        "Age": age,
        "FVC": fvc,
        "Sex": sex,
        "SmokingStatus": smoking
    }])

    df = pd.get_dummies(df, columns=["Sex", "SmokingStatus"])

    # Ensure all columns exist
    for col in TABULAR_FEATURES:
        if col not in df:
            df[col] = 0

    return df[TABULAR_FEATURES].values.astype(np.float32)


# ======================================
# FASTAPI SERVER
# ======================================

app = FastAPI(title="IPF FVC Prediction API")

@app.on_event("startup")
def on_startup():
    try:
        load_model()
    except Exception as e:
        print("❌ Startup failed:", e)


@app.get("/")
def root():
    return {"status": "running", "model_loaded": MODEL is not None}


@app.post("/predict")
async def predict(
    ctScan: UploadFile = File(...),
    weeks: int = Form(...),
    fvc: float = Form(...),
    age: int = Form(...),
    sex: str = Form(...),
    smokingStatus: str = Form(...)
):
    if MODEL is None:
        raise HTTPException(500, "Model not loaded.")

    # Process CT
    try:
        img_input = preprocess_dicom(await ctScan.read())
    except Exception as e:
        raise HTTPException(400, f"DICOM error: {e}")

    # Process tabular
    try:
        tab_input = prepare_tabular_data(age, sex, smokingStatus, weeks, fvc)
    except Exception as e:
        raise HTTPException(400, f"Tabular error: {e}")

    # Predict
    pred = MODEL.predict([img_input, tab_input], verbose=0)
    mu = float(pred[0, 0])
    sigma = float(np.exp(pred[0, 1]))

    return {
        "status": "success",
        "FVC": round(mu, 2),
        "Confidence": round(sigma, 2)
    }
