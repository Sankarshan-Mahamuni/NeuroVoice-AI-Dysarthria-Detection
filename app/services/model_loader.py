"""
model_loader.py
Thread-safe singleton model and artifact loader.
Loads the trained model, standard scaler, and metadata once into memory.
"""
import json
import logging
from pathlib import Path
import joblib
from config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    _model = None
    _scaler = None
    _config = None
    _feature_names = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_artifacts()
        return cls._instance

    def load_artifacts(self):
        logger.info("Initializing ModelManager: loading artifacts...")
        
        if not Config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at: {Config.MODEL_PATH}")
        if not Config.SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler artifact not found at: {Config.SCALER_PATH}")
        if not Config.CONFIG_PATH.exists():
            raise FileNotFoundError(f"Model configuration not found at: {Config.CONFIG_PATH}")

        self._model = joblib.load(Config.MODEL_PATH)
        self._scaler = joblib.load(Config.SCALER_PATH)

        with open(Config.CONFIG_PATH, "r", encoding="utf-8") as f:
            self._config = json.load(f)

        if Config.FEATURE_NAMES_PATH.exists():
            with open(Config.FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
                self._feature_names = json.load(f)
        else:
            self._feature_names = [f"feature_{i}" for i in range(144)]

        logger.info(f"Model '{self._config.get('model_name')}' and scaler loaded successfully.")

    @property
    def model(self):
        if self._model is None:
            self.load_artifacts()
        return self._model

    @property
    def scaler(self):
        if self._scaler is None:
            self.load_artifacts()
        return self._scaler

    @property
    def config(self):
        if self._config is None:
            self.load_artifacts()
        return self._config

    @property
    def feature_names(self):
        return self._feature_names
