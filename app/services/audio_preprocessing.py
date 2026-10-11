"""
audio_preprocessing.py
Shared audio validation, loading, decoding, resampling, and normalization module.
Used by both offline training and real-time Flask web inference.
"""
import io
import numpy as np
import soundfile as sf
import librosa
import subprocess
from pathlib import Path
import imageio_ffmpeg

TARGET_SAMPLE_RATE = 16000
MIN_DURATION_SEC = 0.2
MAX_DURATION_SEC = 15.0

class AudioProcessingError(Exception):
    """Custom exception for audio validation and preprocessing errors."""
    pass

def load_and_validate_audio(audio_input, target_sr=TARGET_SAMPLE_RATE):
    """
    Decodes and validates raw audio from either:
      - File path (str or Path)
      - Raw bytes (e.g., from web upload or Parquet)
      - File-like object (BytesIO)
      
    Returns:
      audio_signal: np.ndarray, float32 1D array normalized to [-1.0, 1.0]
      sample_rate: int, TARGET_SAMPLE_RATE (16000 Hz)
      metadata: dict of signal health properties (duration, rms, clipping, etc.)
    """
    try:
        if isinstance(audio_input, (bytes, bytearray)):
            bio = io.BytesIO(audio_input)
            data, sr = sf.read(bio)
        elif isinstance(audio_input, io.BytesIO):
            audio_input.seek(0)
            data, sr = sf.read(audio_input)
        elif isinstance(audio_input, (str, Path)):
            data, sr = sf.read(audio_input)
        else:
            raise AudioProcessingError("Unsupported audio input format.")
    except (RuntimeError, ValueError, OSError):
        try:
            if isinstance(audio_input, (bytes, bytearray)):
                encoded_audio = bytes(audio_input)
            elif isinstance(audio_input, io.BytesIO):
                audio_input.seek(0)
                encoded_audio = audio_input.read()
            elif isinstance(audio_input, (str, Path)):
                encoded_audio = Path(audio_input).read_bytes()
            else:
                raise AudioProcessingError("Unsupported audio input format.")

            decoded = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", "pipe:0",
                    "-t", "15.1", "-ac", "1", "-ar", str(target_sr),
                    "-f", "wav", "-acodec", "pcm_s16le", "pipe:1"
                ],
                input=encoded_audio,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False
            )
            if decoded.returncode != 0 or not decoded.stdout:
                raise AudioProcessingError(
                    "Could not decode audio. Upload a valid file in a supported audio format."
                )
            data, sr = sf.read(io.BytesIO(decoded.stdout))
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as err:
            raise AudioProcessingError(
                "Could not decode audio. Upload a valid file in a supported audio format."
            ) from err

    if data is None or len(data) == 0:
        raise AudioProcessingError("Audio signal is empty.")

    # Convert to mono if multi-channel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Cast to float32
    data = data.astype(np.float32)

    # Check for NaN or Inf
    if not np.isfinite(data).all():
        raise AudioProcessingError("Audio data contains invalid (NaN or Inf) values.")

    # Resample if sample rate does not match target
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    duration = len(data) / sr

    if duration < MIN_DURATION_SEC:
        raise AudioProcessingError(
            f"Audio duration ({duration:.2f}s) is too short. Minimum required is {MIN_DURATION_SEC}s."
        )

    if duration > MAX_DURATION_SEC:
        raise AudioProcessingError(
            f"Audio duration ({duration:.2f}s) exceeds the {MAX_DURATION_SEC:.0f}s maximum."
        )

    # Check amplitude and silence
    peak_amp = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data**2)))

    if peak_amp < 1e-4 or rms < 1e-4:
        raise AudioProcessingError("Audio is virtually silent or below detectable signal threshold.")

    # Check clipping (> 0.99)
    clipping_ratio = float(np.mean(np.abs(data) >= 0.999))
    has_clipping = clipping_ratio > 0.02  # more than 2% samples clipped

    # Peak normalization with headroom to prevent digital distortion
    if peak_amp > 0:
        normalized_data = data / peak_amp * 0.95
    else:
        normalized_data = data

    metadata = {
        "sample_rate": sr,
        "duration_sec": round(duration, 3),
        "peak_amplitude": round(peak_amp, 4),
        "rms_energy": round(rms, 5),
        "clipping_detected": has_clipping,
        "clipping_ratio": round(clipping_ratio, 4),
        "num_samples": len(normalized_data)
    }

    return normalized_data, sr, metadata
