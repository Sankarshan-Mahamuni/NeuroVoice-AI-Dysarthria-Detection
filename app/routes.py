"""
routes.py
Flask blueprints and route handlers for AI-Based Dysarthria Detection Web Application.
Supports:
- Clinical screening dashboard with audio upload and real-time microphone recording
- REST API endpoint for inference with full JSON response
- File serving for audio playback
- Preset audio sample loader for instant testing
- Health-check endpoint
- Detailed methodology, biomarker science, and evaluation reports
- Commercial pricing and subscription plans
"""
import json
import logging
import csv
from flask import (
    Blueprint, render_template, request, jsonify,
    send_from_directory, url_for, abort
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from config import Config
from app.services.audio_preprocessing import AudioProcessingError
from app.services.inference import run_dysarthria_inference
from app.services.model_loader import ModelManager

logger = logging.getLogger(__name__)
bp = Blueprint("main", __name__)

ENGLISH_PRESETS = [
    ("healthy_speech_sample_rainbow_01.wav", "Healthy Speech: Rainbow Passage (01)", "success", True, "Adult Control 01"),
    ("healthy_speech_sample_vowel_02.wav", "Healthy Speech: Sustained /a/ Vowel (02)", "success", True, "Adult Control 02"),
    ("healthy_speech_sample_conversational_03.wav", "Healthy Speech: Conversational (03)", "success", True, "Adult Control 03"),
    ("healthy_speech_sample_passage_04.wav", "Healthy Speech: Phonetic Agility (04)", "success", True, "Adult Control 04"),
    ("healthy_speech_sample_articulation_05.wav", "Healthy Speech: Syllable Repetition (05)", "success", True, "Adult Control 05"),
    ("dysarthric_speech_sample_mild_01.wav", "Dysarthric Speech: Mild Articulation (01)", "warning", False, "Clinical Patient 01"),
    ("dysarthric_speech_sample_moderate_02.wav", "Dysarthric Speech: Moderate Phonation (02)", "warning", False, "Clinical Patient 02"),
    ("dysarthric_speech_sample_moderate_03.wav", "Dysarthric Speech: Moderate Prosody (03)", "warning", False, "Clinical Patient 03"),
    ("dysarthric_speech_sample_severe_04.wav", "Dysarthric Speech: Severe Motor Impairment (04)", "warning", False, "Clinical Patient 04"),
    ("dysarthric_speech_sample_severe_05.wav", "Dysarthric Speech: Severe Slurred Speech (05)", "warning", False, "Clinical Patient 05"),
]

def load_test_metrics():
    final_metrics = Config.ROOT_DIR / "reports" / "final_test_metrics.json"
    if final_metrics.exists():
        with final_metrics.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "accuracy": 0.8736,
        "precision": 0.9196,
        "recall": 0.8281,
        "f1_score": 0.8715,
        "specificity": 0.9223,
        "roc_auc": 0.9545,
        "confusion_matrix": {"tn": 2292, "fp": 193, "fn": 458, "tp": 2207}
    }

def load_benchmarks():
    bench_file = Config.ROOT_DIR / "reports" / "comprehensive_benchmarks.json"
    if bench_file.exists():
        with bench_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_validation_sample_names():
    sample_names = set()
    if Config.VALIDATION_METADATA_PATH.exists():
        with Config.VALIDATION_METADATA_PATH.open("r", encoding="utf-8", newline="") as metadata_file:
            for row in csv.DictReader(metadata_file):
                if row.get("label") and row.get("speaker"):
                    sample_names.add(f"{row['label']}_{row['speaker']}_{row['filename']}")
                elif row.get("filename"):
                    sample_names.add(row["filename"])
    
    # Always include all curated English preset sample names
    for fname, _, _, _, _ in ENGLISH_PRESETS:
        sample_names.add(fname)
        
    return sample_names

get_available_sample_names = get_validation_sample_names

@bp.app_errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    message = "The upload exceeds the 15 MiB limit. Choose a smaller audio file."
    if (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
        or request.is_json
    ):
        return jsonify({"success": False, "error": message}), 413
    return render_template("index.html", error=message, config={}, samples=[], samples_en=[], samples_it=[], test_metrics=load_test_metrics()), 413

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@bp.route("/")
def index():
    """Render main detection dashboard."""
    model_mgr = ModelManager.get_instance()
    cfg = model_mgr.config
    
    samples = []
    if Config.SAMPLE_AUDIO_DIR.exists():
        for fname, display_label, badge, is_healthy, speaker_info in ENGLISH_PRESETS:
            if (Config.SAMPLE_AUDIO_DIR / fname).is_file():
                samples.append({
                    "filename": fname,
                    "label": display_label,
                    "badge": badge,
                    "speaker_info": speaker_info,
                    "url": url_for("main.serve_sample_audio", filename=fname),
                    "is_healthy": is_healthy
                })

    return render_template(
        "index.html",
        config=cfg,
        samples=samples,
        samples_en=samples[:5],
        samples_it=[],
        test_metrics=load_test_metrics(),
        benchmarks=load_benchmarks()
    )

@bp.route("/analyze", methods=["POST"])
def analyze_audio():
    """
    Primary analysis endpoint.
    Accepts:
      - Uploaded file (multipart/form-data) via 'audio_file'
      - Preset sample selection via 'sample_name'
      - Raw audio blob from live microphone
    Returns:
      - JSON response if requested via AJAX / Accept header
      - Rendered result template if submitted via standard form
    """
    is_json_request = (
        request.headers.get("X-Requested-With") == "XMLHttpRequest" or
        "application/json" in request.headers.get("Accept", "") or
        request.is_json
    )

    try:
        audio_bytes = None
        source_filename = "recording.wav"

        # Check if user selected a preset sample
        sample_name = request.form.get("sample_name") or (request.get_json(silent=True) or {}).get("sample_name")
        if sample_name:
            if sample_name not in get_validation_sample_names():
                return jsonify({"success": False, "error": "The selected demonstration sample is not available."}), 404
            sample_path = Config.SAMPLE_AUDIO_DIR / sample_name
            if not sample_path.is_file():
                return jsonify({"success": False, "error": "The selected demonstration sample is not available."}), 404
            with open(sample_path, "rb") as f:
                audio_bytes = f.read()
            source_filename = sample_name

        # Otherwise check uploaded file
        elif "audio_file" in request.files:
            file = request.files["audio_file"]
            if file.filename == "":
                return jsonify({"success": False, "error": "No file was selected for upload."}), 400
                
            if not allowed_file(file.filename):
                allowed_str = ", ".join(sorted(list(Config.ALLOWED_EXTENSIONS)))
                return jsonify({
                    "success": False,
                    "error": f"Unsupported audio format. Allowed formats: {allowed_str}"
                }), 400
                
            source_filename = secure_filename(file.filename)
            audio_bytes = file.stream.read(Config.MAX_UPLOAD_SIZE_BYTES + 1)
            if len(audio_bytes) > Config.MAX_UPLOAD_SIZE_BYTES:
                return jsonify({
                    "success": False,
                    "error": "The upload exceeds the 15 MiB limit. Choose a smaller audio file."
                }), 413

        elif request.data and len(request.data) > 0:
            # Raw audio blob (e.g. from live microphone recording)
            audio_bytes = request.data
            source_filename = "microphone_recording.wav"

        else:
            return jsonify({"success": False, "error": "No audio file or data received."}), 400

        if not audio_bytes or len(audio_bytes) == 0:
            return jsonify({"success": False, "error": "Uploaded audio file is empty (0 bytes)."}), 400

        # Execute inference pipeline
        result = run_dysarthria_inference(audio_bytes, filename=source_filename)
        result["success"] = True

        if is_json_request:
            return jsonify(result)
        else:
            return render_template("result.html", result=result)

    except AudioProcessingError as ape:
        logger.warning(f"Audio validation failed: {str(ape)}")
        if is_json_request:
            return jsonify({"success": False, "error": str(ape)}), 422
        return render_template("index.html", error=str(ape)), 422

    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        user_msg = "An unexpected error occurred while processing the audio recording. Please ensure it is a valid speech file."
        if is_json_request:
            return jsonify({"success": False, "error": user_msg}), 500
        return render_template("index.html", error=user_msg), 500

@bp.route("/sample-audio/<filename>")
def serve_sample_audio(filename):
    """Serves sample audio files for in-browser playback."""
    if filename not in get_validation_sample_names():
        abort(404)
    return send_from_directory(Config.SAMPLE_AUDIO_DIR, filename)

@bp.route("/api/health")
def health_check():
    """Health check endpoint to verify server and model readiness."""
    try:
        model_mgr = ModelManager.get_instance()
        return jsonify({
            "status": "healthy",
            "model_loaded": model_mgr.model is not None,
            "model_architecture": model_mgr.config.get("model_name"),
            "feature_dim": model_mgr.config.get("feature_dim"),
            "target_sample_rate": model_mgr.config.get("sample_rate"),
            "validation_accuracy": model_mgr.config.get("validation_metrics", {}).get("accuracy")
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@bp.route("/about")
def about():
    """Renders methodology, dataset documentation, and evaluation report."""
    model_mgr = ModelManager.get_instance()
    cfg = model_mgr.config
    return render_template(
        "about.html",
        config=cfg,
        test_metrics=load_test_metrics(),
        benchmarks=load_benchmarks()
    )

@bp.route("/api/demo-action", methods=["POST"])
def demo_action():
    """Endpoint for interactive commercial actions (e.g. subscribing or booking demo)."""
    data = request.get_json(silent=True) or {}
    plan = data.get("plan", "Standard")
    return jsonify({
        "success": True,
        "message": f"Successfully initiated demonstration access for {plan} tier! A confirmation package has been prepared.",
        "plan": plan
    }), 200
