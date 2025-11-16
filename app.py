# app.py
import os
import io
import tempfile
import requests
import numpy as np
import pydicom
import cv2
import pandas as pd
from typing import Dict, Any

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.applications import EfficientNetB3

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# Configuration
# -------------------------
app = FastAPI(title="FVC Prediction API", version="1.0")

# Allow CORS - change origins to your Vercel domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with ["https://your-vercel-site.com"] for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# If you uploaded the weights to repo root (not recommended), it will use that.
# Otherwise it will download from Google Drive direct link into a temp file on startup.
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'fvc_model_weights.weights.h5')
FVC_MODEL = None

# Google Drive direct download link (replace with your own if different)
# Example: "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
DRIVE_MODEL_URL = os.getenv("DRIVE_MODEL_URL", "https://drive.google.com/uc?export=download&id=1Gg-QlOWS4qRAJHhcaZzeq0D2KHve2d7c")

# Preprocessing constants (match training)
IMG_SIZE = 256
TABULAR_FEATURES = ['Weeks', 'Age', 'FVC', 'Sex_male', 'SmokingStatus_Ex-smoker', 'SmokingStatus_Never Smoker']

# -------------------------
# Custom loss
# -------------------------
def laplace_log_likelihood(y_true, y_pred):
    mu = y_pred[:, 0]
    log_sigma = y_pred[:, 1]
    sigma = K.exp(log_sigma)
    y_true_fvc = y_true[:, 0]
    loss = (K.abs(y_true_fvc - mu) / sigma) + K.log(2 * sigma)
    return K.mean(loss)

# -------------------------
# Model builder & loader
# -------------------------
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), tabular_dim=len(TABULAR_FEATURES)):
    img_input = Input(shape=input_shape, name='image_input')
    x = Concatenate()([img_input, img_input, img_input])
    cnn = EfficientNetB3(weights=None, include_top=False, input_tensor=x)
    cnn.trainable = False

    image_features = cnn.output
    image_features = GlobalAveragePooling2D()(image_features)
    image_features = Dense(64, activation='relu')(image_features)

    tabular_input = Input(shape=(tabular_dim,), name='tabular_input')
    tabular_features = Dense(32, activation='relu')(tabular_input)

    fused = Concatenate()([image_features, tabular_input])
    fused = Dense(64, activation='relu')(fused)

    mu_output = Dense(1, name='mu')(fused)
    sigma_output = Dense(1, name='log_sigma')(fused)

    combined_output = Concatenate(axis=-1)([mu_output, sigma_output])

    model = Model(inputs=[img_input, tabular_input], outputs=combined_output)
    model.compile(optimizer='adam', loss=laplace_log_likelihood)
    return model

def download_weights_from_drive(drive_url: str, dest_path: str, chunk_size: int = 32768):
    """
    Download large file by streaming (saves to dest_path).
    """
    resp = requests.get(drive_url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return dest_path

def load_model_from_weights():
    """
    Build model architecture then load weights. If weights file missing, try to download from DRIVE_MODEL_URL.
    """
    # if file present in repo root or path configured
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        # try download into tmp file
        try:
            tmpdir = tempfile.gettempdir()
            tmp_path = os.path.join(tmpdir, "fvc_model_weights.weights.h5")
            print(f"Model weights not found at {MODEL_WEIGHTS_PATH}. Attempting to download to {tmp_path} ...")
            download_weights_from_drive(DRIVE_MODEL_URL, tmp_path)
            weights_path_to_use = tmp_path
        except Exception as e:
            raise RuntimeError(f"Failed to download model weights: {e}")
    else:
        weights_path_to_use = MODEL_WEIGHTS_PATH

    try:
        model = build_model()
        model.load_weights(weights_path_to_use)
        print("✅ Model weights loaded from:", weights_path_to_use)
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load model weights: {e}")

# -------------------------
# DICOM preprocessing
# -------------------------
def preprocess_dicom(dicom_bytes: bytes) -> np.ndarray:
    try:
        dcm_data = pydicom.dcmread(io.BytesIO(dicom_bytes))
        img = dcm_data.pixel_array.astype(np.int16)

        if 'RescaleSlope' in dcm_data and 'RescaleIntercept' in dcm_data:
            img = img * float(dcm_data.RescaleSlope) + float(dcm_data.RescaleIntercept)

        MIN_HU = -1000.0
        MAX_HU = -400.0
        img = np.clip(img, MIN_HU, MAX_HU)
        img = (img - MIN_HU) / (MAX_HU - MIN_HU)

        img = (img * 255).astype(np.uint8)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img_final = img_resized.astype(np.float32) / 255.0
        img_final = np.expand_dims(np.expand_dims(img_final, axis=0), axis=-1)
        return img_final
    except Exception as e:
        print(f"DICOM processing error: {e}")
        raise ValueError(f"Could not process DICOM file: {e}")

# -------------------------
# Tabular preparation
# -------------------------
def prepare_tabular_data(age: int, sex: str, smoking_status: str, weeks: int, fvc: float) -> np.ndarray:
    try:
        input_data = {'Weeks': [weeks], 'Age': [age], 'FVC': [fvc], 'Sex': [sex], 'SmokingStatus': [smoking_status]}
        df = pd.DataFrame(input_data)
        df = pd.get_dummies(df, columns=['Sex', 'SmokingStatus'], prefix=['Sex', 'SmokingStatus'])
        expected_features = TABULAR_FEATURES
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        tabular_input_vector = df[expected_features].values.astype(np.float32)
        return tabular_input_vector
    except Exception as e:
        print(f"Tabular data preparation error: {e}")
        raise ValueError(f"Could not prepare tabular data: {e}")

# -------------------------
# App startup: load model once
# -------------------------
@app.on_event("startup")
def startup_event():
    global FVC_MODEL
    try:
        FVC_MODEL = load_model_from_weights()
        # warmup
        dummy_img = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        dummy_tab = np.zeros((1, len(TABULAR_FEATURES)), dtype=np.float32)
        FVC_MODEL.predict([dummy_img, dummy_tab], verbose=0)
        print("✅ Model warmup complete.")
    except Exception as e:
        print(f"❌ API startup failed: {e}")
        FVC_MODEL = None

# -------------------------
# Endpoints
# -------------------------
@app.get("/")
def read_root():
    return {"message": "FVC Prediction API", "model_loaded": FVC_MODEL is not None}

@app.post("/predict")
async def predict_fvc(
    ctScan: UploadFile = File(...),
    weeks: int = Form(...),
    fvc: float = Form(...),
    age: int = Form(...),
    sex: str = Form(...),
    smokingStatus: str = Form(...)
) -> Dict[str, Any]:
    if FVC_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check logs.")

    # 1. DICOM bytes -> image tensor
    try:
        dicom_bytes = await ctScan.read()
        image_input = preprocess_dicom(dicom_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image file error: {e}")

    # 2. Tabular
    try:
        tabular_input = prepare_tabular_data(age=age, sex=sex, smoking_status=smokingStatus, weeks=weeks, fvc=fvc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tabular data error: {e}")

    # 3. Predict
    try:
        prediction_output = FVC_MODEL.predict([image_input, tabular_input], verbose=0)
        mu_pred = float(prediction_output[0, 0])
        log_sigma_pred = float(prediction_output[0, 1])
        confidence_ml = float(np.exp(log_sigma_pred))
    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    return {
        "status": "success",
        "FVC": round(mu_pred, 2),
        "Confidence": round(confidence_ml, 2),
        "Patient_Week_Stub": f"P_WK_{weeks}"
    }
