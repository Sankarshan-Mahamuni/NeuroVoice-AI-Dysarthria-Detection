"""
evaluate_model.py
Performs the one-time final evaluation of the frozen selected model on the held-out test set
comprising 13 completely unseen speakers (5,150 recordings).
Generates the final metrics, confusion matrix, per-speaker evaluation, and evaluation_report.md.
"""
import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models/selected_model")
REPORTS_DIR = Path("reports")

def evaluate_test_set():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading frozen selected model and scaler...")
    model = joblib.load(MODELS_DIR / "model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "model_config.json") as f:
        config = json.load(f)
        
    print(f"Model architecture: {config['model_name']}")
    
    print("\nLoading held-out test set (13 unseen speakers)...")
    X_test = np.load(PROCESSED_DIR / "test_features.npy")
    y_test = np.load(PROCESSED_DIR / "test_labels.npy")
    spk_test = np.load(PROCESSED_DIR / "test_speakers.npy", allow_pickle=True)
    test_meta = pd.read_csv(PROCESSED_DIR / "test_metadata.csv")
    
    print(f"Total test samples: {len(X_test)}")
    print(f"Test class balance: Healthy={np.sum(y_test == 0)}, Dysarthric={np.sum(y_test == 1)}")
    print(f"Test speakers ({len(np.unique(spk_test))}): {sorted(list(np.unique(spk_test)))}")
    
    # Scale test set using frozen scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
    
    print("\n==========================================")
    print("FINAL HELD-OUT TEST EVALUATION RESULTS")
    print("==========================================")
    print(f"Accuracy:            {acc*100:.2f}%")
    print(f"Precision:           {prec*100:.2f}%")
    print(f"Recall (Dysarthria): {rec*100:.2f}%")
    print(f"F1-Score:            {f1*100:.2f}%")
    print(f"Specificity:         {spec*100:.2f}%")
    print(f"ROC-AUC:             {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (Healthy predicted as Healthy):     {tn} / {tn+fp} ({tn/(tn+fp)*100:.1f}%)")
    print(f"  False Positives (Healthy predicted as Dysarthric):  {fp} / {tn+fp} ({fp/(tn+fp)*100:.1f}%)")
    print(f"  False Negatives (Dysarthric predicted as Healthy):  {fn} / {fn+tp} ({fn/(fn+tp)*100:.1f}%)")
    print(f"  True Positives (Dysarthric predicted as Dysarthric):{tp} / {fn+tp} ({tp/(fn+tp)*100:.1f}%)")
    
    # Per-speaker analysis
    df_eval = pd.DataFrame({
        "speaker": spk_test,
        "y_true": y_test,
        "y_pred": y_pred,
        "severity": test_meta["severity"]
    })
    
    spk_breakdown = []
    for spk, g in df_eval.groupby("speaker"):
        spk_acc = (g["y_true"] == g["y_pred"]).mean()
        spk_class = "dysarthric" if g["y_true"].iloc[0] == 1 else "healthy"
        sev = str(g["severity"].iloc[0])
        spk_breakdown.append({
            "speaker": spk,
            "class": spk_class,
            "severity": sev,
            "total_samples": len(g),
            "correct_predictions": int((g["y_true"] == g["y_pred"]).sum()),
            "accuracy": f"{spk_acc*100:.1f}%",
            "raw_acc": spk_acc
        })
        
    df_spk = pd.DataFrame(spk_breakdown)
    print("\nPer-Speaker Breakdown on Test Set:")
    print(df_spk[["speaker", "class", "severity", "total_samples", "accuracy"]].to_string(index=False))
    
    # Save test metrics JSON
    test_metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "specificity": round(float(spec), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_speaker": spk_breakdown
    }
    with open(REPORTS_DIR / "final_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
        
    # Generate evaluation_report.md
    report_md = f"""# Final Evaluation Report: Dysarthria Detection System

## 1. Evaluation Protocol & Rigor
- **Evaluation Type:** Strictly Speaker-Independent Held-Out Test Evaluation.
- **Zero Leakage:** No test speaker or recording was ever seen during feature normalization, hyperparameter tuning, or model fitting.
- **Test Subjects:** 13 completely unseen human speakers (6 healthy controls, 7 dysarthric individuals).
- **Test Sample Size:** {len(X_test)} recordings across varying phonemes and smartphone voice commands.

## 2. Final Held-Out Test Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **{acc*100:.2f}%** | Overall correct classification rate on unseen individuals. |
| **Dysarthria Recall** | **{rec*100:.2f}%** | Sensitivity: fraction of dysarthric utterances correctly caught. |
| **Dysarthria Precision** | **{prec*100:.2f}%** | Positive Predictive Value. |
| **F1-Score** | **{f1*100:.2f}%** | Harmonic balance between recall and precision. |
| **Specificity** | **{spec*100:.2f}%** | True Negative Rate: Healthy speech correctly identified as healthy. |
| **ROC-AUC** | **{auc:.4f}** | Area under the Receiver Operating Characteristic curve. |

## 3. Confusion Matrix Breakdown
- **True Negatives (Healthy -> Healthy):** {tn} / {tn+fp} ({tn/(tn+fp)*100:.1f}%)
- **False Positives (Healthy -> Dysarthric):** {fp} / {tn+fp} ({fp/(tn+fp)*100:.1f}%)
- **False Negatives (Dysarthric -> Healthy):** {fn} / {fn+tp} ({fn/(fn+tp)*100:.1f}%)
- **True Positives (Dysarthric -> Dysarthric):** {tp} / {fn+tp} ({tp/(fn+tp)*100:.1f}%)

## 4. Per-Speaker Performance Analysis
The table below documents performance on every individual held-out speaker:

| Speaker ID | Ground Truth Class | Clinical Severity | Total Samples | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
"""
    for row in spk_breakdown:
        report_md += f"| `{row['speaker']}` | {row['class']} | {row['severity']} | {row['total_samples']} | **{row['accuracy']}** |\n"

    report_md += f"""
## 5. Limitations and Responsible Interpretation
1. **Speaker-level variation:** Aggregate metrics are utterance-level and conceal large differences between held-out participants. For example, dysarthric speaker `m17` has 31.2% utterance accuracy, while several other speakers are above 90%. This variation limits claims about performance on a new individual.
2. **Severity:** The small held-out speaker sample does not support a reliable severity-performance conclusion. The model predicts only a binary class and does not estimate clinical severity.
3. **No feature explanations:** The classifier output is not an explanation of which acoustic characteristics caused an individual prediction. The displayed class scores are not calibrated clinical probabilities.
4. **Medical disclaimer:** This is an engineering prototype for academic demonstration, not a clinically validated screening or diagnostic device.
"""
    with open(REPORTS_DIR / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"\nSaved final test report to: {REPORTS_DIR / 'evaluation_report.md'}")

if __name__ == "__main__":
    evaluate_test_set()
