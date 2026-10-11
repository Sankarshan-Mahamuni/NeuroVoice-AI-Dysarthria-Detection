"""
test_preprocessing.py
Unit tests for audio validation, resampling, and 144-dim acoustic feature extraction.
"""
import io
import subprocess
import numpy as np
import pytest
import soundfile as sf
import imageio_ffmpeg
from app.services.audio_preprocessing import (
    load_and_validate_audio, AudioProcessingError, TARGET_SAMPLE_RATE, MIN_DURATION_SEC
)
from app.services.feature_extraction import (
    extract_acoustic_features, compute_log_mel_spectrogram, FEATURE_DIM
)

@pytest.fixture
def synthetic_speech_wav():
    """Generates a 1.0s synthetic harmonic speech-like wave at 8000 Hz."""
    sr = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental frequency (120 Hz) with harmonics (240 Hz, 360 Hz)
    signal = 0.5 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 240 * t)
    bio = io.BytesIO()
    sf.write(bio, signal.astype(np.float32), sr, format='WAV')
    bio.seek(0)
    return bio.read()

def test_load_and_validate_audio_resampling(synthetic_speech_wav):
    signal, sr, meta = load_and_validate_audio(synthetic_speech_wav)
    assert sr == TARGET_SAMPLE_RATE, f"Expected {TARGET_SAMPLE_RATE}, got {sr}"
    assert signal.ndim == 1, "Signal should be 1-dimensional mono"
    assert len(signal) == int(TARGET_SAMPLE_RATE * 1.0)
    assert meta["duration_sec"] == 1.0
    assert not meta["clipping_detected"]
    assert np.isfinite(signal).all()


def test_webm_audio_decode(synthetic_speech_wav):
    encoded = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "wav",
            "-i", "pipe:0", "-c:a", "libopus", "-f", "webm", "pipe:1"
        ],
        input=synthetic_speech_wav,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    assert encoded.returncode == 0, encoded.stderr.decode(errors="replace")

    signal, sr, metadata = load_and_validate_audio(encoded.stdout)
    assert sr == TARGET_SAMPLE_RATE
    assert signal.ndim == 1
    assert metadata["duration_sec"] > 0.9


def test_empty_audio_raises_error():
    bio = io.BytesIO()
    sf.write(bio, np.array([], dtype=np.float32), 16000, format='WAV')
    bio.seek(0)
    with pytest.raises(AudioProcessingError) as exc_info:
        load_and_validate_audio(bio.read())
    assert "empty" in str(exc_info.value).lower()


def test_missing_audio_path_raises_controlled_error():
    with pytest.raises(AudioProcessingError, match="Could not decode audio"):
        load_and_validate_audio("data/sample_audio/does-not-exist.wav")


def test_too_short_audio_raises_error():
    # 0.05 seconds audio (shorter than MIN_DURATION_SEC = 0.2s)
    short_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.05, 800))
    bio = io.BytesIO()
    sf.write(bio, short_signal.astype(np.float32), 16000, format='WAV')
    bio.seek(0)
    with pytest.raises(AudioProcessingError) as exc_info:
        load_and_validate_audio(bio.read())
    assert "too short" in str(exc_info.value).lower()


def test_too_long_audio_raises_error():
    long_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 15.1, int(16000 * 15.1), endpoint=False))
    bio = io.BytesIO()
    sf.write(bio, long_signal.astype(np.float32), 16000, format="WAV")
    with pytest.raises(AudioProcessingError) as exc_info:
        load_and_validate_audio(bio.getvalue())
    assert "exceeds" in str(exc_info.value).lower()


def test_silent_audio_raises_error():
    silence = np.zeros(16000, dtype=np.float32)
    bio = io.BytesIO()
    sf.write(bio, silence, 16000, format='WAV')
    bio.seek(0)
    with pytest.raises(AudioProcessingError) as exc_info:
        load_and_validate_audio(bio.read())
    assert "silent" in str(exc_info.value).lower()

def test_stereo_to_mono_conversion():
    stereo = np.ones((16000, 2), dtype=np.float32) * 0.5
    bio = io.BytesIO()
    sf.write(bio, stereo, 16000, format='WAV')
    bio.seek(0)
    signal, sr, meta = load_and_validate_audio(bio.read())
    assert signal.ndim == 1

def test_feature_extraction_dimension_and_validity(synthetic_speech_wav):
    signal, sr, _ = load_and_validate_audio(synthetic_speech_wav)
    feats = extract_acoustic_features(signal, sr)
    assert feats.shape == (FEATURE_DIM,), f"Expected shape ({FEATURE_DIM},), got {feats.shape}"
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all(), "Features must contain no NaN or Inf values"

def test_spectrogram_dimensions(synthetic_speech_wav):
    signal, sr, _ = load_and_validate_audio(synthetic_speech_wav)
    spec = compute_log_mel_spectrogram(signal, sr, n_mels=40, max_frames=120)
    assert spec.shape == (40, 120)
    assert spec.min() >= 0.0 and spec.max() <= 1.0
