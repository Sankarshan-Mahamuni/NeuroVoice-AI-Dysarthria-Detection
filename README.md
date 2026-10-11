# AI-Based Dysarthria Detection System

An academic Flask demonstration that classifies short speech recordings as
healthy or dysarthric using a Logistic Regression model and 144 summary acoustic
features. The model is a research prototype, not a clinically validated
screening or diagnostic device.

## Current project status

- Dataset: EasyCall Dysarthric Speech Corpus, with 21,386 catalogued recordings
  from 55 speakers (31 dysarthric, 24 healthy controls).
- Partitions: 31 training, 11 validation, and 13 held-out test speakers. The
  speaker IDs in the saved arrays have no overlap between partitions.
- Selected model: Logistic Regression, selected using validation data. The
  evaluated saved model obtained 87.36% accuracy, 91.96% precision, 82.81%
  dysarthria recall, 87.15% F1, 92.23% specificity, and 0.9545 ROC-AUC on
  5,150 utterances from the held-out test speakers. Per-speaker variation is
  substantial; see [the evaluation report](reports/evaluation_report.md).
- The synthetic feature-mixup experiment did not improve validation results and
  is not used by the selected model.
- Historical feature exports contain 21,268 of the 21,386 catalogued records.
  The earlier extraction run did not preserve per-record failure reasons. New
  runs write failed recording IDs and error messages to
  `data/metadata/feature_extraction_failures.csv`.
  `scripts/audit_feature_exports.py` lists the records absent from the existing
  saved feature exports without inventing a cause for those omissions.

## Requirements and installation (Windows / PowerShell)

Python 3.11 is the tested interpreter version. From the project root:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` records the package versions installed in the
development environment. `requirements.txt` lists the direct project
dependencies with compatible version ranges.

## Dataset and model workflow

The dataset download is provided by Hugging Face:
`https://huggingface.co/datasets/changelinglab/easycall-dysarthria`.
The project report records the dataset's stated CC BY-NC 2.0 license. Review
the dataset card and license before downloading or using the corpus. It is
non-commercially licensed; do not redistribute the downloaded audio or derived
speaker-level data without confirming the applicable terms.

Run the preparation and training stages in this order:

```powershell
.\venv\Scripts\python.exe scripts\download_dataset.py
.\venv\Scripts\python.exe scripts\validate_dataset.py
.\venv\Scripts\python.exe scripts\extract_features.py
.\venv\Scripts\python.exe scripts\audit_feature_exports.py
.\venv\Scripts\python.exe scripts\train_and_compare.py
.\venv\Scripts\python.exe scripts\evaluate_model.py
```

The download and extraction stages require internet access and substantial local
disk space. The downloaded Parquet files are approximately 0.8 GB in total.
Feature extraction uses multiple processes and can take several minutes. The
dataset-provided partitions are kept speaker-independent. Training selects a
model on the validation partition; the evaluation script then evaluates the
selected model on the held-out test partition.

The extraction script writes processed arrays under `data/processed`, sample
recordings under `data/sample_audio`, and a failure log under
`data/metadata/feature_extraction_failures.csv`. Raw data and generated arrays
are excluded from version control by `.gitignore`.

## Run the Flask application

Start the app from the project root:

```powershell
.\venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000/`. The web app supports audio file selection,
drag-and-drop, microphone recordings where the browser supports them, and locally extracted validation-speaker sample recordings. Held-out test audio is
not offered as an interactive preset. Supported extensions are WAV, MP3, OGG, FLAC, WebM,
and M4A; an FFmpeg binary is included for decoding these formats. Uploads are
limited to 15 MiB and accepted recordings must be 0.2-15 seconds long. Longer
recordings are rejected rather than silently truncated. `/api/health` reports
the loaded model status.

The application loads the saved model once at startup and does not train during
requests. Uploaded recordings are processed in memory and are not retained.
Model scores shown in the interface are uncalibrated classifier scores, not
clinical probabilities.

The landing page includes the short Italian test sentence “Ciao, come stai?”
with a simple pronunciation hint and English meaning. Read it aloud, choose
**Record Live Microphone**, stop the recording, and then choose **Analyze Speech
Sample**. The model is an academic prototype trained on Italian speech.

## Tests

Run preprocessing, inference, and Flask route tests:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Tests require the saved model artifacts, which are available in the provided
project folder. The inference and upload tests generate temporary speech-like
audio themselves. Re-run the dataset and training workflow if you are setting
the project up from source without the saved model.

## Limitations

- There are only 55 independent speakers, and performance varies considerably
  by held-out speaker. Utterance counts do not replace independent subjects.
- EasyCall consists of Italian speech recordings and smartphone commands; the
  model's results should not be generalized to other languages or recording
  conditions without new evaluation.
- The reported metrics are utterance-level metrics from this corpus, not
  estimates of real-world clinical performance.
- The test set has already been evaluated and its reported results should not be
  used to select new models or tune thresholds.
- This application is for academic demonstration only and must not replace
  assessment by qualified health professionals.

See [the dataset report](reports/dataset_report.md) and
[the evaluation report](reports/evaluation_report.md) for source, protocol, and
speaker-level results.
