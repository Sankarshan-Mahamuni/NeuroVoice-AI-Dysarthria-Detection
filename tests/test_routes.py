"""Integration tests for the Flask pages and audio analysis endpoint."""
import io
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest
import soundfile as sf

from app import create_app
from app.routes import get_validation_sample_names
from config import Config


def make_wav_bytes():
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    signal = 0.5 * np.sin(2 * np.pi * 120 * time)
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, signal, sample_rate, format="WAV")
    return audio_buffer.getvalue()


def test_pages_and_health_endpoint():
    client = create_app().test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert b"Speech Classification Demonstration" in response.data
    assert b"Read this clinical speech passage" in response.data
    assert b"Rainbow Passage" in response.data

    response = client.get("/about")
    assert response.status_code == 200
    assert b"87.36%" in response.data
    assert b"not calibrated probabilities" in response.data
    assert b"with VAD" not in response.data

    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"
    assert response.json["model_loaded"] is True


def test_uploaded_recording_inference_returns_model_scores():
    client = create_app().test_client()
    response = client.post(
        "/analyze",
        data={"audio_file": (io.BytesIO(make_wav_bytes()), "test.wav")},
        content_type="multipart/form-data",
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 200
    assert response.json["success"] is True
    assert response.json["prediction"] in {"Healthy Speech", "Dysarthric Speech"}
    assert "model_score_pct" in response.json
    assert "confidence_score" not in response.json
    assert "biomarkers" not in response.json


def test_standard_form_upload_renders_classification_result():
    response = create_app().test_client().post(
        "/analyze",
        data={"audio_file": (io.BytesIO(make_wav_bytes()), "test.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Top model score (not calibrated)" in response.data
    assert b"not a diagnosis" in response.data


def test_test_speaker_presets_are_not_served():
    response = create_app().test_client().post(
        "/analyze",
        data={"sample_name": "healthy_fc10_fc10_06_Cella.wav"},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 404


def test_validation_sample_is_available_when_extracted():
    available_samples = get_validation_sample_names()
    sample_name = next(
        (
            name for name in sorted(available_samples)
            if (Config.SAMPLE_AUDIO_DIR / name).is_file()
        ),
        None
    )
    if sample_name is None:
        pytest.skip("Validation demonstration audio has not been extracted.")

    client = create_app().test_client()
    audio_response = client.get(f"/sample-audio/{quote(sample_name, safe='')}")
    assert audio_response.status_code == 200

    result = client.post(
        "/analyze",
        data={"sample_name": sample_name},
        headers={"Accept": "application/json"},
    )
    assert result.status_code == 200
    assert result.json["success"] is True


def test_upload_errors_are_controlled():
    client = create_app().test_client()

    missing = client.post("/analyze", headers={"Accept": "application/json"})
    assert missing.status_code == 400
    assert "No audio" in missing.json["error"]

    unsupported = client.post(
        "/analyze",
        data={"audio_file": (io.BytesIO(b"audio"), "recording.txt")},
        content_type="multipart/form-data",
        headers={"Accept": "application/json"},
    )
    assert unsupported.status_code == 400
    assert "Unsupported audio format" in unsupported.json["error"]

    corrupt = client.post(
        "/analyze",
        data={"audio_file": (io.BytesIO(b"not audio"), "recording.wav")},
        content_type="multipart/form-data",
        headers={"Accept": "application/json"},
    )
    assert corrupt.status_code == 422
    assert corrupt.json["error"] == (
        "Could not decode audio. Upload a valid file in a supported audio format."
    )
    assert "BytesIO" not in corrupt.json["error"]


def test_upload_size_limit():
    client = create_app().test_client()
    too_large = b"x" * (15 * 1024 * 1024 + 1)
    response = client.post(
        "/analyze",
        data={"audio_file": (io.BytesIO(too_large), "large.wav")},
        content_type="multipart/form-data",
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 413
    assert "15 MiB limit" in response.json["error"]
