"""
evaluate_multilingual_model.py
Evaluates the frozen multilingual model on the held-out test set
(17 unseen speakers across English and Italian, 9,685 total recordings).
Reports metrics overall and broken down by language (English vs Italian).
"""
import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "selected_model"
REPORTS_DIR = ROOT_DIR / "reports"

def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "specificity": round(float(spec), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

def evaluate_multilingual_test():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading selected multilingual model and scaler...")
    model = joblib.load(MODELS_DIR / "model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "model_config.json") as f:
        cfg = json.load(f)
        
    print(f"Model: {cfg['model_name']}")
    
    print("Loading multilingual test set (17 unseen speakers)...")
    X_test = np.load(PROCESSED_DIR / "multi_test_features.npy")
    y_test = np.load(PROCESSED_DIR / "multi_test_labels.npy")
    spk_test = np.load(PROCESSED_DIR / "multi_test_speakers.npy", allow_pickle=True)
    lang_test = np.load(PROCESSED_DIR / "multi_test_langs.npy", allow_pickle=True)
    
    print(f"Total test samples: {len(X_test)} (Healthy={np.sum(y_test==0)}, Dysarthric={np.sum(y_test==1)})")
    print(f"Languages: English={np.sum(lang_test=='en')}, Italian={np.sum(lang_test=='it')}")
    print(f"Total unseen test speakers: {len(np.unique(spk_test))}")
    
    # Scale features
    X_test_s = scaler.transform(X_test)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else None
    
    overall_m = compute_metrics(y_test, y_pred, y_prob)
    
    # English subset
    en_idx = np.where(lang_test == "en")[0]
    en_m = compute_metrics(y_test[en_idx], y_pred[en_idx], y_prob[en_idx] if y_prob is not None else None)
    
    # Italian subset
    it_idx = np.where(lang_test == "it")[0]
    it_m = compute_metrics(y_test[it_idx], y_pred[it_idx], y_prob[it_idx] if y_prob is not None else None)
    
    print("\n=======================================================")
    print("MULTILINGUAL HELD-OUT TEST EVALUATION RESULTS")
    print("=======================================================")
    print(f"OVERALL ACCURACY:       {overall_m['accuracy']*100:.2f}%")
    print(f"OVERALL F1-SCORE:       {overall_m['f1_score']*100:.2f}%")
    print(f"OVERALL DYSARTHRIA REC: {overall_m['recall']*100:.2f}%")
    print(f"OVERALL SPECIFICITY:    {overall_m['specificity']*100:.2f}%")
    print(f"OVERALL ROC-AUC:        {overall_m['roc_auc']:.4f}")
    
    print("\n--- By Language Breakdown ---")
    print(f"ENGLISH (TORGO) Accuracy:  {en_m['accuracy']*100:.2f}% | F1: {en_m['f1_score']*100:.2f}% | Recall: {en_m['recall']*100:.2f}% | AUC: {en_m['roc_auc']:.4f}")
    print(f"ITALIAN (EasyCall) Acc:    {it_m['accuracy']*100:.2f}% | F1: {it_m['f1_score']*100:.2f}% | Recall: {it_m['recall']*100:.2f}% | AUC: {it_m['roc_auc']:.4f}")
    
    # Per-speaker analysis
    df_eval = pd.DataFrame({"speaker": spk_test, "lang": lang_test, "y_true": y_test, "y_pred": y_pred})
    spk_breakdown = []
    for spk, g in df_eval.groupby("speaker"):
        spk_acc = (g["y_true"] == g["y_pred"]).mean()
        spk_class = "dysarthric" if g["y_true"].iloc[0] == 1 else "healthy"
        lang = g["lang"].iloc[0]
        spk_breakdown.append({
            "speaker": spk,
            "language": "English" if lang == "en" else "Italian",
            "class": spk_class,
            "total_samples": len(g),
            "accuracy": f"{spk_acc*100:.1f}%",
            "raw_acc": spk_acc
        })
        
    df_spk = pd.DataFrame(spk_breakdown).sort_values(by="raw_acc", ascending=False)
    print("\n--- Per-Speaker Test Set Performance (17 Unseen Speakers) ---")
    print(df_spk[["speaker", "language", "class", "total_samples", "accuracy"]].to_string(index=False))
    
    test_results_dict = {
        "overall": overall_m,
        "english": en_m,
        "italian": it_m,
        "per_speaker": spk_breakdown
    }
    with open(REPORTS_DIR / "multilingual_test_metrics.json", "w") as f:
        json.dump(test_results_dict, f, indent=2)
        
    print(f"\nSaved multilingual test metrics to {REPORTS_DIR / 'multilingual_test_metrics.json'}")

if __name__ == "__main__":
    evaluate_multilingual_test()
