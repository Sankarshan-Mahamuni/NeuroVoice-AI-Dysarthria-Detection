"""
config.py
Configuration parameters for the Flask Dysarthria Detection web application.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY")
    ROOT_DIR = BASE_DIR
    
    # Audio Upload Constraints
    MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024
    MAX_REQUEST_SIZE_BYTES = 16 * 1024 * 1024
    MAX_CONTENT_LENGTH = MAX_REQUEST_SIZE_BYTES
    ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "webm", "m4a"}
    
    # Model Artifacts
    MODELS_DIR = BASE_DIR / "models" / "selected_model"
    MODEL_PATH = MODELS_DIR / "model.joblib"
    SCALER_PATH = MODELS_DIR / "scaler.joblib"
    CONFIG_PATH = MODELS_DIR / "model_config.json"
    FEATURE_NAMES_PATH = BASE_DIR / "data" / "processed" / "feature_names.json"
    
    # Sample Audio for testing
    SAMPLE_AUDIO_DIR = BASE_DIR / "data" / "sample_audio"
    VALIDATION_METADATA_PATH = BASE_DIR / "data" / "processed" / "validation_metadata.csv"
    
    # Audio Signal Parameters
    TARGET_SAMPLE_RATE = 16000
    MIN_DURATION_SEC = 0.2
    MAX_DURATION_SEC = 15.0
