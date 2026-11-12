# 🎙️ NeuroVoice AI: Clinical Speech Disorder & Dysarthria Detection System

[![Python 3.11](https://img.shields.io/badge/Python-3.11%20%7C%203.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Pytest-19%2F19%20Passing-10b981?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)
[![Accuracy](https://img.shields.io/badge/Held--Out%20Accuracy-87.36%25-0284c7?style=for-the-badge)](https://github.com/)
[![Specificity](https://img.shields.io/badge/Specificity-92.23%25-059669?style=for-the-badge)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **An open-source, reproducible AI-powered speech screening platform capable of detecting acoustic motor speech disorders (dysarthria) from voice recordings using a 144-dimensional biomarker extraction pipeline and zero-leakage speaker-independent machine learning.**

---

## 📌 Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. What is Dysarthria?](#2-what-is-dysarthria)
- [3. The 144 Acoustic Biomarkers](#3-the-144-acoustic-biomarkers)
- [4. Model Benchmarks & Evaluation](#4-model-benchmarks--evaluation)
- [5. Zero-Leakage Speaker-Independent Protocol](#5-zero-leakage-speaker-independent-protocol)
- [6. Application Architecture & Tech Stack](#6-application-architecture--tech-stack)
- [7. Directory Structure](#7-directory-structure)
- [8. Quick Start: How to Run This Project](#8-quick-start-how-to-run-this-project)
- [9. Feature Walkthrough & Demonstration](#9-feature-walkthrough--demonstration)
- [10. REST API Documentation](#10-rest-api-documentation)
- [11. Testing & Code Quality](#11-testing--code-quality)
- [12. Ethical & Medical Disclaimer](#12-ethical--medical-disclaimer)
- [13. License](#13-license)

---

## 1. Executive Summary

Dysarthria is a motor speech disorder characterized by impaired muscle movement in the vocal tract, lips, tongue, or diaphragm. Traditional clinical diagnosis requires manual evaluation by Speech-Language Pathologists (SLPs), which can be inaccessible in rural, telehealth, or emergency settings.

**NeuroVoice AI** bridges this gap by offering a non-invasive, sub-300ms acoustic screening tool:
- **No Identity Leakage:** Trained and validated using strict human speaker grouping. No speaker in the test cohort was ever seen during training or feature normalization.
- **Pre-trained Checkpoints Included:** Model weights (`model.joblib`), standard scalers, and configuration files are fully versioned in the repository. **You can clone and immediately run inference without retraining.**
- **Production-Ready Web Application:** Modern, responsive medical SaaS UI featuring live microphone waveform rendering, Log-Mel spectrogram visualization, standardized clinical reading protocols, and interactive symptom checklists.

---

## 2. What is Dysarthria?

Dysarthria results from neurological injury affecting the muscles responsible for speech production. Intellect, thought formulation, and language comprehension remain intact, but neuromuscular control is compromised.

```
Neurological Event (Stroke, PD, ALS)
                 │
                 ▼
Impaired Speech Motor Control (Cranial Nerves V, VII, IX, X, XII)
                 │
  ┌──────────────┼──────────────┬──────────────┐
  ▼              ▼              ▼              ▼
Respiration    Phonation   Articulation     Prosody
(Short breath) (Tremor)   (Slurred vowels) (Monotone)
                 │
                 ▼
144-D Acoustic Biomarker Extraction Engine
                 │
                 ▼
Clinical Triage & Spectrogram Visualization (<300ms)
```

### The 6 Clinical Subtypes
| Subtype | Primary Lesion Site | Auditory Acoustic Traits | Associated Conditions |
| :--- | :--- | :--- | :--- |
| **Flaccid** | Lower Motor Neuron | Breathy vocal quality, hypernasality, audible inspiration | Bulbar ALS, Myasthenia Gravis, Bell's Palsy |
| **Spastic** | Bilateral Upper Motor Neuron | Harsh, strained-strangled vocal quality, slow rate | Pseudobulbar palsy, Multiple Strokes, CP |
| **Ataxic** | Cerebellum | Slurred cadence, irregular articulatory breakdown | Cerebellar stroke, Multiple Sclerosis, Ataxia |
| **Hypokinetic**| Basal Ganglia (Dopamine Depletion) | Monopitch, hypophonia (quiet voice), rapid rushes of words | Parkinson's Disease, PSP |
| **Hyperkinetic**| Basal Ganglia (Excess Movement) | Unpredictable voice arrests, sudden loud blasts, tremors | Huntington's Disease, Dystonia |
| **Mixed** | Multiple Motor Systems | Combination of spastic stiffness and flaccid weakness | Advanced ALS, Severe Traumatic Brain Injury |

---

## 3. The 144 Acoustic Biomarkers

Raw audio is normalized to **16,000 Hz 16-bit mono** and mapped into a comprehensive 144-dimensional mathematical feature space:

1. **Static Mel-Frequency Cepstral Coefficients (MFCCs 1–20) — 40 dimensions:**
   - Mean and standard deviation across time capturing vocal tract geometric filter shape and vowel formant positions ($F_1, F_2$).
2. **First-Order Velocity Dynamics ($\Delta$ MFCCs 1–20) — 40 dimensions:**
   - Temporal polynomial derivative measuring articulatory velocity and speed of consonant transitions.
3. **Second-Order Acceleration Dynamics ($\Delta\Delta$ MFCCs 1–20) — 40 dimensions:**
   - Measures muscular acceleration, vocal cord stability, and articulatory jerkiness/ataxia.
4. **Spectral Entropy & Flatness Dynamics — 20 dimensions:**
   - Wiener Spectral Flatness and Spectral Centroid measuring breathy vocal fold air leakage and high-frequency friction turbulence.
5. **Prosodic Energy & Pitch Perturbation — 4 dimensions:**
   - Root-Mean-Square (RMS) energy distribution, peak amplitude stability, and Zero-Crossing Rate (ZCR) assessing phonatory endurance.

---

## 4. Model Benchmarks & Evaluation

All classifiers were trained on **31 independent speakers (11,881 recordings)**, tuned on **11 held-out validation speakers (4,237 recordings)**, and tested on **13 completely quarantined speakers (5,150 recordings)**:

| Model Architecture | Validation Acc | Held-Out Test Acc | Dysarthria Recall | Specificity (Control) | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Clinical Soft-Voting Ensemble (LR + GBDT + MLP)** | 70.52% | **87.79%** | 82.66% | **93.28%** | **0.9597** | Highest Consensus AUC |
| **Balanced Logistic Regression ($C=0.02$)** | 71.23% | **87.61%** | 82.44% | 93.16% | 0.9583 | Optimal Decision Boundary |
| **Logistic Regression + Synthetic Mixup Augmentation** | 70.57% | **87.50%** | **82.85%** | 92.47% | 0.9555 | **Highest Pathology Recall** |
| **Production Selected: Logistic Regression ($L_2, C=0.1$)** | **70.85%** | **87.36%** | **82.81%** | **92.23%** | **0.9545** | **Active Production Engine** |
| **Histogram Gradient Boosting (GBDT)** | 65.28% | 87.15% | 81.28% | 93.44% | 0.9525 | Non-Linear Baseline |
| **Multi-Layer Perceptron (64 $\to$ 32, $L_2=0.1$)** | 68.92% | 85.17% | 80.08% | 90.62% | 0.9395 | Deep Neural Representation |

### Confusion Matrix on Unseen Held-Out Test Set (5,150 Samples)
- **True Negatives (Healthy Controls correctly identified):** `2,292 / 2,485` (**92.2%**)
- **False Positives (Controls flagged as dysarthric):** `193 / 2,485` (**7.8% low false-alarm rate**)
- **True Positives (Dysarthric patients caught):** `2,207 / 2,665` (**82.8% sensitivity**)
- **False Negatives (Dysarthric patients missed):** `458 / 2,665` (**17.2%**)

---

## 5. Zero-Leakage Speaker-Independent Protocol

 многих speech AI papers report artificially inflated accuracies (>98%) because they perform random utterance splits. If the same person's voice appears in both train and test partitions, the model simply memorizes the speaker's pitch, microphone, and room acoustics rather than pathology biomarkers.

**Our Guarantee:**
- **Zero Speaker Overlap:** Splits are strictly partitioned by human identity.
- **Zero Normalization Leakage:** The `StandardScaler` is fitted exclusively on the training corpus and frozen before validation or test evaluation.
- **Quarantined Test Set:** The test set of 13 unseen human subjects was sealed until the final benchmark evaluation.

---

## 6. Application Architecture & Tech Stack

- **Backend:** Python 3.11, Flask 3.0, Werkzeug
- **Audio Processing:** Librosa 0.10, SoundFile, NumPy, SciPy, `imageio-ffmpeg`
- **Machine Learning:** Scikit-Learn (Regularized Linear Model, GBDT, MLP), Joblib
- **Frontend:** Vanilla JavaScript (ES6+), HTML5 Web Audio API (`AudioContext`, `AnalyserNode`), HTML5 Canvas
- **Design System:** Responsive CSS3 medical design system with light clinical theme, glassmorphism, and print stylesheet
- **Testing:** Pytest 8.0, AnyIO

---

## 7. Directory Structure

```text
Dysarthia_Detection_Project/
├── app.py                      # Flask application entry point
├── config.py                   # Centralized configuration & file paths
├── conftest.py                 # Pytest environment fixtures
├── requirements.txt            # Project dependencies
├── requirements-lock.txt       # Frozen dependency versions
├── README.md                   # Complete system documentation
│
├── app/
│   ├── routes.py               # Application endpoints & API handlers
│   ├── services/
│   │   ├── audio_preprocessing.py   # Ingestion, validation, 16kHz resampling
│   │   ├── feature_extraction.py    # 144-D vector & spectrogram extraction
│   │   ├── model_loader.py          # Singleton model artifact loader
│   │   └── inference.py             # Inference pipeline & scoring engine
│   ├── templates/
│   │   ├── base.html           # Layout shell, navigation, & disclaimer
│   │   ├── index.html          # Dashboard, 6-tab knowledge hub, presets
│   │   ├── about.html          # Methodology, math formulas, & benchmarks
│   │   └── result.html         # Non-JS fallback result view
│   └── static/
│       ├── css/style.css       # Clean medical light-mode stylesheet
│       └── js/main.js          # Audio recording, live waveform, & AJAX
│
├── data/
│   ├── sample_audio/           # 10 Curated English clinical WAV files
│   ├── processed/              # Validation metadata & feature definitions
│   └── raw/                    # (Optional) Raw dataset archives
│
├── models/
│   └── selected_model/
│       ├── model.joblib        # Pre-trained classifier checkpoint
│       ├── scaler.joblib       # Fitted StandardScaler
│       └── model_config.json   # Model configuration & metadata
│
├── reports/
│   ├── final_test_metrics.json # Test metrics with per-speaker breakdown
│   ├── comprehensive_benchmarks.json # Multi-model comparison metrics
│   └── evaluation_report.md    # Clinical evaluation documentation
│
├── scripts/
│   ├── train_and_compare.py    # Baseline & advanced model training
│   ├── evaluate_model.py       # Frozen test evaluation script
│   ├── fast_model_experiments.py # Fast multi-model benchmarking script
│   └── smoke_test_live.py      # End-to-end API verification script
│
└── tests/
    ├── test_inference.py       # Inference pipeline & singleton tests
    ├── test_preprocessing.py   # Resampling, WebM decode, & feature tests
    └── test_routes.py          # Route validation & upload error tests
```

---

## 8. Quick Start: How to Run This Project

Follow these steps to run the project on any computer (**Windows, macOS, or Linux**):

### Prerequisites
- **Python 3.10 or 3.11** installed. *(During Windows installation, make sure to check "Add python.exe to PATH")*.
- **Git** installed.

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/Sankarshan-Mahamuni/NeuroVoice-AI-Dysarthria-Detection.git
cd NeuroVoice-AI-Dysarthria-Detection
```

---

### Step 2: Create a Virtual Environment
```bash
# On Windows (PowerShell or Command Prompt):
python -m venv venv

# On macOS / Linux:
python3 -m venv venv
```

---

### Step 3: Activate the Virtual Environment
```powershell
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# (If PowerShell displays an Execution Policy restriction, run:)
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# Then re-run: .\venv\Scripts\Activate.ps1

# On Windows (Command Prompt - cmd.exe):
venv\Scripts\activate.bat

# On macOS / Linux:
source venv/bin/activate
```

---

### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> **Note:** Audio decoding uses `imageio-ffmpeg`, which automatically provisions the required FFmpeg binaries inside your environment. You do not need to install system FFmpeg manually.

---

### Step 5: Run Automated Tests
Verify that all 19 tests pass on your machine:
```bash
pytest -v
```
**Expected Output:**
```text
============================= 19 passed in ~10s =============================
```

---

### Step 6: Start the Web Application
```bash
python app.py
```
**Expected Terminal Output:**
```text
[INFO] app.services.model_loader: Model 'Clinical Acoustic Speech Benchmark (16 kHz, 144 Biomarkers)' and scaler loaded successfully.
[INFO] dysarthria_app: Starting NeuroVoice AI Dysarthria Detection Server on port 5000...
 * Running on http://127.0.0.1:5000
```

---

### Step 7: Open the Dashboard
Open your web browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

## 9. Feature Walkthrough & Demonstration

1. **Instant Presets:** On the left sidebar, click any preset button (e.g. *Healthy: Rainbow Passage* or *Dysarthric: Moderate Phonation*). The sample loads and triggers instant inference.
2. **Live Microphone Recording:** Click **"Record Live Microphone"**, read the standard clinical passage, and observe the live audio waveform canvas. Click stop to analyze.
3. **File Upload:** Drag-and-drop any `.wav`, `.mp3`, `.flac`, `.ogg`, or `.webm` speech file (0.2s to 15s).
4. **Classification & Triage:** View the triage pill (*Normal Phonation* vs. *Pathological Markers Detected*), model score bar, clinical recommendation, and signal metadata.
5. **Formant Spectrogram:** Inspect the 40-band thermal Log-Mel Spectrogram generated in real time via HTML5 canvas.
6. **Report Export:** Click **"Print / Save Clinical Report"** for a clean, printer-friendly PDF medical summary, or **"Copy Biomarker JSON"** to export raw data.
7. **Knowledge Hub:** Switch between the 6 educational tabs at the top to explore the simplified patient guide, 6 dysarthria subtypes, 144 biomarkers, model comparison benchmarks, system flowchart, and the interactive symptom checklist.

---

## 10. REST API Documentation

### 1. Ingest & Analyze Audio
- **Endpoint:** `POST /analyze`
- **Content-Type:** `multipart/form-data` or `application/json`
- **Parameters:**
  - `audio_file`: Audio file binary (`.wav`, `.mp3`, `.flac`, `.ogg`, `.webm`, max 15 MiB).
  - OR `sample_name`: Name of verified demonstration sample (e.g. `healthy_speech_sample_rainbow_01.wav`).
- **Response (`200 OK`):**
```json
{
  "success": true,
  "prediction": "Healthy Speech",
  "predicted_target": 0,
  "model_score_pct": 96.7,
  "score_healthy_pct": 96.7,
  "score_dysarthric_pct": 3.3,
  "summary_note": "Acoustic features demonstrate stable phonatory frequency and steady resonance contours consistent with healthy control speech.",
  "signal_metadata": {
    "duration_sec": 2.5,
    "sample_rate": 16000,
    "peak_amplitude": 0.65,
    "rms_energy": 0.0821,
    "clipping_detected": false
  },
  "spectrogram": {
    "num_mel_bins": 40,
    "num_time_frames": 120,
    "values": [...]
  }
}
```

### 2. Health Check
- **Endpoint:** `GET /api/health`
- **Response (`200 OK`):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_architecture": "Clinical Acoustic Speech Benchmark (16 kHz, 144 Biomarkers)",
  "feature_dim": 144,
  "target_sample_rate": 16000,
  "validation_accuracy": 0.8736
}
```

---

## 11. Testing & Code Quality

The system includes automated tests located in `tests/`:
- `test_inference.py`: Verifies singleton model loading, prediction formatting, and confidence bounds.
- `test_preprocessing.py`: Validates sample rate conversion (16 kHz), WebM decoding, silent audio detection, clipping detection, and 144-D vector shapes.
- `test_routes.py`: Verifies Flask endpoints, error handling (400, 404, 413, 422), file upload security, and HTML rendering.

To execute tests:
```bash
pytest -v
```

---

## 12. Ethical & Medical Disclaimer

> ⚠️ **Important Clinical Declaration**
> 
> NeuroVoice AI is distributed solely for academic research, educational exploration, and proof-of-concept clinical screening support.
> - This software is **not a certified diagnostic medical device** under FDA (SaMD) or EU MDR regulations.
> - It must **never** be used as an independent diagnostic tool or to replace comprehensive evaluation by a licensed Speech-Language Pathologist (SLP) or Board-Certified Neurologist.
> - Model scores represent empirical classification likelihoods, not calibrated medical diagnostic probabilities.

---

## 13. License

Distributed under the **MIT License**. See `LICENSE` for more information.

Developed by **[Sankarshan Mahamuni](https://github.com/Sankarshan-Mahamuni)**.
