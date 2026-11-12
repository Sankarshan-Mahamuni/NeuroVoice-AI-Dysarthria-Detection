# AI-Based Dysarthria Detection System: Final Engineering & Research Report

**Project Title:** AI-Based Dysarthria Detection System  
**Framework:** Multilingual Acoustic Speech Screening (English + Italian)  
**Authors:** Machine Learning & Speech Processing Engineering Team  
**Date:** October 2026  
**License:** Open Academic & Educational Use  

---

## Executive Summary

This project delivers a complete, reproducible, end-to-end **AI-Based Dysarthria Detection System** engineered from scratch. The system analyzes human speech recordings to perform non-invasive screening for motor speech disorders (dysarthria) associated with neurological conditions such as Cerebral Palsy, Amyotrophic Lateral Sclerosis (ALS), Stroke, and Parkinson's disease.

Crucially, this implementation directly addresses and overcomes the core pitfalls of previous attempts:
1. **Multilingual & Multi-Speaker Scale:** Rather than relying exclusively on a 15-speaker English subset (TORGO) or an Italian-only cohort (EasyCall), we integrated **BOTH English (TORGO) and Italian (EasyCall)** speech corpora. The resulting dataset comprises **70 independent human subjects** (39 dysarthric patients, 31 healthy control subjects) and **37,771 verified audio recordings**.
2. **Zero Speaker Leakage:** A strict speaker-grouped split was enforced. Every speaker exists in exactly one split across training, validation, and testing.
3. **Reproducible Shared Preprocessing:** A single 144-dimensional acoustic biomarker pipeline (`app/services/audio_preprocessing.py` and `app/services/feature_extraction.py`) handles both batch offline feature extraction and real-time online web inference.
4. **Independent Test Verification:** The frozen multilingual model was evaluated once on a held-out test set of **17 completely unseen human speakers** (4 English, 13 Italian), achieving **78.10% overall accuracy**, **79.20% dysarthria recall**, and **0.8616 ROC-AUC** across both languages.
5. **Full-Stack Web Demonstration:** A responsive Flask application featuring HTML5 live microphone recording, drag-and-drop file upload, audio waveform preview, real-time Log-Mel spectrogram canvas rendering, and test presets for both English and Italian.
6. **Rigorous Automated Testing:** 19 automated pytest tests covering audio resampling, WebM decoding, feature shapes, error handling, route contracts, and file upload limits (100% pass rate).

---

## 1. Dataset Report

### 1.1 Source Corpora & Licensing
| Property | Corpus 1: EasyCall | Corpus 2: TORGO | Joint Multilingual Dataset |
| :--- | :--- | :--- | :--- |
| **Language** | Italian (`it`) | English (`en`) | **Multilingual (`en` + `it`)** |
| **Institution** | Italian Institute of Tech (IIT), Univ. of Ferrara, Sant'Anna Hospital | Univ. of Toronto & Toronto Rehabilitation Institute | Academic Research Collaboration |
| **Citation** | Turrisi et al., *INTERSPEECH 2021* | Rudzicz et al., *JSLHR 2012 / LRE 2012* | Unified Corpus |
| **License** | CC BY-NC 2.0 | Academic Research Non-Commercial | Non-Commercial Research |
| **Total Speakers** | 55 (31 Dysarthric, 24 Control) | 15 (8 Dysarthric, 7 Control) | **70 Independent Speakers** |
| **Total Utterances**| 21,386 recordings | 16,552 recordings | **37,938 total recordings** |
| **Native Audio** | 8 kHz PCM WAV mono | 16 kHz PCM WAV (headMic & arrayMic) | Standardized to 16 kHz mono |
| **Acoustic Task** | Smartphone voice commands & phonetically balanced words | Digits, Frenchay assessment words, Grandfather passage | Fixed & variable duration speech |

### 1.2 Speaker-Independent Split Design
To ensure zero speaker identity leakage, speakers were partitioned strictly by subject ID:

| Split | English Speakers (TORGO) | Italian Speakers (EasyCall) | Total Speakers | Total Samples |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `F01`, `F03`, `M01`, `M02`, `M04`, `FC01`, `FC02`, `MC01`, `MC02` (9 spks) | 18 dysarthric, 13 healthy controls (31 spks) | **40 Speakers** | 21,117 samples |
| **Validation** | `M03`, `FC03` (2 spks) | 6 dysarthric, 5 healthy controls (11 spks) | **13 Speakers** | 6,969 samples |
| **Held-Out Test** | `F04`, `M05`, `MC03`, `MC04` (4 spks) | 7 dysarthric, 6 healthy controls (13 spks) | **17 Speakers** | 9,685 samples |
| **TOTAL** | **15 Speakers (8 dys / 7 ctrl)** | **55 Speakers (31 dys / 24 ctrl)** | **70 Speakers** | **37,771 clean samples** |

*Verification:* Automated integrity checks confirm zero speaker intersection between train, validation, and test sets.

---

## 2. Acoustic Feature Pipeline & Preprocessing

The system extracts a **144-dimensional acoustic biomarker vector** per utterance:
1. **Mel-Frequency Cepstral Coefficients (MFCCs 1–20) [40 features]:**
   - Frame-level mean (20) and standard deviation (20) capturing vocal tract resonances and formant configurations.
2. **First Derivative Velocity (Delta MFCCs 1–20) [40 features]:**
   - Mean (20) and standard deviation (20) capturing articulatory transition speed.
3. **Second Derivative Acceleration (Delta-Delta MFCCs 1–20) [40 features]:**
   - Mean (20) and standard deviation (20) capturing articulatory acceleration and motor speech inertia.
4. **Spectral Descriptors [20 features]:**
   - Spectral Centroid (mean, std) [2]
   - Spectral Bandwidth (mean, std) [2]
   - Spectral Rolloff at 85% (mean, std) [2]
   - Spectral Flatness (mean, std) [2] – key biomarker for vocal breathiness and friction turbulence
   - Spectral Contrast across 6 sub-bands (mean [6], std [6]) [12]
5. **Prosodic & Energy Stability [4 features]:**
   - Root Mean Square (RMS) energy (mean, std) [2] – vocal volume tremor & respiratory stability
   - Zero-Crossing Rate (ZCR) (mean, std) [2] – voiced-to-unvoiced phonation ratio

All features are normalized using a `StandardScaler` fitted strictly on training speakers only.

---

## 3. Model Training, Validation, and Test Results

### 3.1 Validation Benchmark (13 Unseen Speakers, 6,969 Recordings)
Models were trained on the 40 training speakers and evaluated on the 13 validation speakers:

| Model Architecture | Overall Validation Accuracy | Overall F1-Score | English Accuracy | Italian Accuracy | ROC-AUC | Selected |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP (Neural Network)** | **63.60%** | **57.92%** | **56.33%** | **68.28%** | **0.6767** | **BEST (Selected)** |
| Logistic Regression (L2) | 64.28% | 57.78% | 56.81% | 69.11% | 0.6642 | Baseline |
| Random Forest (150 trees) | 64.13% | 55.47% | 61.27% | 65.97% | 0.6647 | Candidate |
| HistGradientBoosting (GBDT)| 62.52% | 53.61% | 58.38% | 65.19% | 0.6444 | Candidate |

### 3.2 Synthetic Data Experiment Findings
We evaluated feature-level convex interpolation (within-class Mixup) on the training set to simulate inter-speaker acoustic continua without cross-class label corruption.
- *Result:* Mixup generated 2,000 synthetic acoustic vectors, but produced no meaningful improvement on unseen validation speakers (+0.3% accuracy, within noise variance).
- *Decision:* In strict accordance with the master guidelines ("Reject synthetic methods that fail to improve intended metrics; do not include them in the final pipeline"), synthetic data was excluded from the production pipeline.

### 3.3 Final Held-Out Test Evaluation (17 Unseen Speakers, 9,685 Recordings)
The selected Regularized Multi-Layer Perceptron was frozen and evaluated on the held-out test set:

| Evaluation Metric | Overall Multilingual Test | English (TORGO) Test | Italian (EasyCall) Test |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **78.10%** | **69.46%** | **85.71%** |
| **Dysarthria Recall** | **79.20%** | **75.48%** | **80.94%** |
| **Dysarthria Precision**| **70.34%** | **46.61%** | **90.44%** |
| **F1-Score** | **74.50%** | **57.63%** | **85.43%** |
| **Specificity** | **77.36%** | **67.17%** | **90.82%** |
| **ROC-AUC** | **0.8616** | **0.7942** | **0.9424** |

#### Multilingual Confusion Matrix (9,685 Recordings)
- **True Negatives (Healthy predicted as Healthy):** 4,465 / 5,772 (77.4%)
- **False Positives (Healthy predicted as Dysarthric):** 1,307 / 5,772 (22.6%)
- **False Negatives (Dysarthric predicted as Healthy):** 814 / 3,913 (20.8%)
- **True Positives (Dysarthric predicted as Dysarthric):** 3,099 / 3,913 (79.2%)

#### Per-Speaker Breakdown (17 Unseen Test Speakers)
| Speaker | Language | Pathology Class | Total Recordings | Screening Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| `it_mc11` | Italian | Healthy Control | 414 | **98.1%** |
| `it_mc07` | Italian | Healthy Control | 461 | **97.8%** |
| `it_f09` | Italian | Dysarthric (Severity 1) | 414 | **97.3%** |
| `it_fc04` | Italian | Healthy Control | 394 | **96.7%** |
| `it_f04` | Italian | Dysarthric | 233 | **95.3%** |
| `it_fc10` | Italian | Healthy Control | 410 | **95.1%** |
| `it_m11` | Italian | Dysarthric (Severity 3) | 371 | **93.5%** |
| `it_f06` | Italian | Dysarthric (Severity 1) | 414 | **92.3%** |
| `it_mc04` | Italian | Healthy Control | 395 | **87.6%** |
| `en_M05` | English | Dysarthric (Clinical) | 575 | **87.3%** |
| `it_m12` | Italian | Dysarthric (Severity 1) | 414 | **85.7%** |
| `en_MC03` | English | Healthy Control | 1,667 | **80.6%** |
| `it_m10` | Italian | Dysarthric (Severity 2) | 405 | **75.6%** |
| `it_fc07` | Italian | Healthy Control | 411 | **68.9%** |
| `en_F04` | English | Dysarthric (Clinical) | 673 | **65.4%** |
| `en_MC04` | English | Healthy Control | 1,620 | **53.4%** |
| `it_m17` | Italian | Dysarthric (Severity 1) | 414 | **34.3%** |

---

## 4. Full-Stack Application Architecture

### 4.1 Architecture Diagram
```text
                      [USER]
                        │
                        ▼
            RESPONSIVE FRONTEND (HTML5 + CSS + JS)
       ┌─────────────────┴──────────────────┐
       ▼                                    ▼
 [Live Microphone Recording]        [Audio File Upload]
 (MediaRecorder API / WebM)          (WAV, MP3, OGG, FLAC)
       │                                    │
       └─────────────────┬──────────────────┘
                         ▼
             POST /analyze (Flask Backend)
        ┌────────────────┴─────────────────┐
        ▼                                  ▼
[Upload Validation & Temp File]    [File Size & MIME Sanitization]
        │
        ▼
   AUDIO PREPROCESSING SERVICE (app/services/audio_preprocessing.py)
   - Decode -> Mono conversion -> 16 kHz Resample -> DC Offset & Clipping Checks
        │
        ▼
   FEATURE EXTRACTION SERVICE (app/services/feature_extraction.py)
   - Extract 144-dimensional acoustic biomarker vector
   - Compute Log-Mel Spectrogram (40x120) for visual heatmap rendering
        │
        ▼
   INFERENCE ENGINE (app/services/inference.py)
   - Pre-loaded Singleton ModelManager (joblib)
   - Apply StandardScaler -> Multi-Layer Perceptron Forward Pass
   - Output Predicted Class & Uncalibrated Decision Scores
        │
        ▼
   DYNAMIC RESULTS INTERFACE
   - Status badge (Healthy vs Dysarthric)
   - Score progress meter
   - Time-frequency canvas spectrogram display
   - Signal health metadata (duration, RMS, clipping)
   - Clear medical research disclaimers
```

### 4.2 Key Software Components Created
- `config.py`: Centralized configuration (upload limits, file extensions, artifact paths).
- `app/services/audio_preprocessing.py`: Shared audio loading, decoding, resampling, and clipping detection.
- `app/services/feature_extraction.py`: 144-dim acoustic vector extraction and 2D Log-Mel spectrogram generation.
- `app/services/model_loader.py`: Singleton model loader caching model weights and scaler in memory.
- `app/services/inference.py`: End-to-end inference execution pipeline returning predictions, scores, and spectrogram arrays.
- `app/routes.py`: Flask blueprint handling `/`, `/analyze`, `/sample-audio/<file>`, `/api/health`, and `/about`.
- `app/templates/index.html`: Responsive dashboard with bilingual speech reading prompts, presets, live recorder, and visualizer.
- `app/templates/about.html`: In-depth scientific report with confusion matrix and per-speaker results.
- `app/templates/result.html`: Non-AJAX fallback view for standard form submissions.
- `app/static/css/style.css`: Clean, accessible dark medical-tech styling with mobile responsiveness.
- `app/static/js/main.js`: Client-side logic for drag-and-drop, MediaRecorder, AJAX fetch, and canvas rendering.

---

## 5. Automated Test Suite Verification

Pytest was executed across the codebase with the following results:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
collected 19 items

tests/test_inference.py::test_model_loader_singleton PASSED              [  5%]
tests/test_inference.py::test_inference_on_sample_audio PASSED           [ 10%]
tests/test_preprocessing.py::test_load_and_validate_audio_resampling PASSED [ 15%]
tests/test_preprocessing.py::test_webm_audio_decode PASSED               [ 21%]
tests/test_preprocessing.py::test_empty_audio_raises_error PASSED        [ 26%]
tests/test_preprocessing.py::test_missing_audio_path_raises_controlled_error PASSED [ 31%]
tests/test_preprocessing.py::test_too_short_audio_raises_error PASSED    [ 36%]
tests/test_preprocessing.py::test_too_long_audio_raises_error PASSED     [ 42%]
tests/test_preprocessing.py::test_silent_audio_raises_error PASSED       [ 47%]
tests/test_preprocessing.py::test_stereo_to_mono_conversion PASSED       [ 52%]
tests/test_preprocessing.py::test_feature_extraction_dimension_and_validity PASSED [ 57%]
tests/test_preprocessing.py::test_spectrogram_dimensions PASSED          [ 63%]
tests/test_routes.py::test_pages_and_health_endpoint PASSED              [ 68%]
tests/test_routes.py::test_uploaded_recording_inference_returns_model_scores PASSED [ 73%]
tests/test_routes.py::test_standard_form_upload_renders_classification_result PASSED [ 78%]
tests/test_routes.py::test_test_speaker_presets_are_not_served PASSED    [ 84%]
tests/test_routes.py::test_validation_sample_is_available_when_extracted PASSED [ 89%]
tests/test_routes.py::test_upload_errors_are_controlled PASSED           [ 94%]
tests/test_routes.py::test_upload_size_limit PASSED                      [100%]

============================= 19 passed in 4.98s ==============================
```

- **Tests Executed:** 19
- **Tests Passed:** 19 (100%)
- **Tests Failed:** 0

---

## 6. Reproducible Execution Instructions

### Step 1: Environment Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install exact pinned dependencies
pip install -r requirements.txt
```

### Step 2: Dataset Download
```powershell
# Download EasyCall (Italian)
python scripts/download_dataset.py

# Download TORGO (English)
python scripts/download_torgo.py
```

### Step 3: Dataset Preparation & Feature Extraction
```powershell
# Validate EasyCall metadata
python scripts/validate_dataset.py

# Build unified multilingual catalog (70 speakers)
python scripts/prepare_multilingual_dataset.py

# Parallel acoustic feature extraction (144 features)
python scripts/extract_multilingual_features.py
```

### Step 4: Model Training & Evaluation
```powershell
# Train and benchmark models across 40 training speakers
python scripts/train_multilingual_models.py

# Evaluate frozen model on 17 held-out test speakers
python scripts/evaluate_multilingual_model.py

# Run test suite
pytest -v
```

### Step 5: Start the Flask Application
```powershell
# Run local development server
python app.py
```
Open **`http://127.0.0.1:5000`** in any modern web browser.

---

## 7. Clinical Limitations & Future Directions

1. **Acoustic vs. Perceptual Severity:** Mild dysarthric speakers who preserve articulatory compensation (such as speaker `it_m17`) remain challenging to detect from short isolated commands without conversational or diadochokinetic (DDK) syllable tasks.
2. **Screening Prototype Boundary:** This tool is an academic screening prototype designed for preliminary triage and motor speech research. It is **not a diagnostic medical device** and cannot replace comprehensive examination by a certified Speech-Language Pathologist (SLP) or neurologist.
3. **Future Multi-Modal Expansion:** Integrating synchronized lip-movement video (computer vision) or submental surface electromyography (sEMG) with acoustic features will further boost sensitivity on borderline mild dysarthric individuals.
