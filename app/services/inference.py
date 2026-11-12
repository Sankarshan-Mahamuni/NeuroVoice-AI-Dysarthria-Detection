"""
inference.py
Inference service that encapsulates audio validation, preprocessing,
feature extraction, scaling, and classification scoring.
Returns a binary classifier result, uncalibrated model scores, and signal metadata.
"""
import numpy as np
from app.services.audio_preprocessing import load_and_validate_audio, AudioProcessingError
from app.services.feature_extraction import extract_acoustic_features, compute_log_mel_spectrogram
from app.services.model_loader import ModelManager

def run_dysarthria_inference(audio_input, filename="uploaded_audio"):
    """
    Executes the complete inference pipeline on an input audio source.
    
    Parameters:
      audio_input: bytes, bytearray, or path string.
      filename: str, source identifier for logging and reporting.
      
    Returns:
      dict containing prediction, uncalibrated class scores,
      signal health metrics, and a spectrogram array.
    """
    # 1. Validation and Preprocessing
    signal, sr, signal_meta = load_and_validate_audio(audio_input)

    # 2. Extract Acoustic Feature Vector (144-dim)
    features = extract_acoustic_features(signal, sr)

    # 3. Generate Spectrogram for UI Visualization (downsampled for fast JSON serialization)
    spec_2d = compute_log_mel_spectrogram(signal, sr, n_mels=40, max_frames=120)
    spectrogram_data = {
        "n_mels": 40,
        "n_frames": spec_2d.shape[1],
        "values": [[round(float(v), 3) for v in row] for row in spec_2d]
    }

    # 4. Model Prediction
    model_mgr = ModelManager.get_instance()
    model = model_mgr.model
    scaler = model_mgr.scaler
    config = model_mgr.config

    # Scale using exact training scaler
    features_scaled = scaler.transform(features.reshape(1, -1))

    # These are classifier scores, not calibrated clinical probabilities.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_scaled)[0]
        score_healthy = float(probabilities[0])
        score_dysarthric = float(probabilities[1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(features_scaled)[0])
        score_dysarthric = 1.0 / (1.0 + np.exp(-score))
        score_healthy = 1.0 - score_dysarthric
    else:
        pred_raw = int(model.predict(features_scaled)[0])
        score_dysarthric = 1.0 if pred_raw == 1 else 0.0
        score_healthy = 1.0 - score_dysarthric

    predicted_class = "Dysarthric Speech" if score_dysarthric >= 0.50 else "Healthy Speech"

    if predicted_class == "Dysarthric Speech":
        alert_level = "warning"
    else:
        alert_level = "success"
        
    summary_note = (
        f"The model classified this recording as {predicted_class.lower()}. "
        "This research model is not clinically validated; the result is not a diagnosis."
    )

    # Input quality warnings
    warnings = []
    if signal_meta["clipping_detected"]:
        warnings.append("Input audio contains noticeable clipping (>2% clipped samples), which may introduce acoustic artifacts.")
    if signal_meta["duration_sec"] < 0.6:
        warnings.append("The recording is very short (<0.6s). Longer speech samples (1-5s) offer higher diagnostic stability.")

    result = {
        "filename": filename,
        "prediction": predicted_class,
        "predicted_target": 1 if predicted_class == "Dysarthric Speech" else 0,
        "model_score_pct": round(max(score_healthy, score_dysarthric) * 100, 1),
        "score_healthy_pct": round(score_healthy * 100, 1),
        "score_dysarthric_pct": round(score_dysarthric * 100, 1),
        "alert_level": alert_level,
        "summary_note": summary_note,
        "signal_metadata": signal_meta,
        "warnings": warnings,
        "spectrogram": spectrogram_data,
        "model_used": config.get("model_name", "Trained Classifier")
    }

    return result
