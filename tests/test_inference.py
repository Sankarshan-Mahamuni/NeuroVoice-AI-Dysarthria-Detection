"""
test_inference.py
Tests model loading and inference execution on synthetic and real test recordings.
"""
import io
import numpy as np
import pytest
import soundfile as sf
from app.services.model_loader import ModelManager
from app.services.inference import run_dysarthria_inference

def test_model_loader_singleton():
    mgr1 = ModelManager.get_instance()
    mgr2 = ModelManager.get_instance()
    assert mgr1 is mgr2
    assert mgr1.model is not None
    assert mgr1.scaler is not None
    assert mgr1.config is not None
    assert "model_name" in mgr1.config

def test_inference_on_sample_audio():
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    signal = 0.5 * np.sin(2 * np.pi * 120 * time)
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, signal, sample_rate, format="WAV")
    result = run_dysarthria_inference(audio_buffer.getvalue(), filename="test.wav")

    assert "prediction" in result
    assert result["prediction"] in ["Healthy Speech", "Dysarthric Speech"]
    assert 50.0 <= result["model_score_pct"] <= 100.0
    assert abs((result["score_healthy_pct"] + result["score_dysarthric_pct"]) - 100.0) < 0.2
    assert "not clinically validated" in result["summary_note"]
    assert "biomarkers" not in result
    assert "spectrogram" in result
    assert "signal_metadata" in result
    assert result["spectrogram"]["n_mels"] == 40
