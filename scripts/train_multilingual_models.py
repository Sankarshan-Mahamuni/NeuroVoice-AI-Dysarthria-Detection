"""
train_multilingual_models.py
Trains baseline and regularized classifiers on the combined multilingual training set
(40 independent human speakers across English and Italian).
Evaluates and benchmarks on the multilingual validation set (13 unseen speakers).
Saves the best-performing model, scaler, and multilingual configuration to models/selected_model/.
"""
import json
import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

def calculate_metrics(y_true, y_pred, y_prob=None):
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

def evaluate_subsets(model, scaler, X, y, langs, speakers):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    
    overall = calculate_metrics(y, y_pred, y_prob)
    
    # By Language
    by_lang = {}
    for lang in ["en", "it"]:
        idx = np.where(langs == lang)[0]
        if len(idx) > 0:
            prob_sub = y_prob[idx] if y_prob is not None else None
            by_lang[lang] = calculate_metrics(y[idx], y_pred[idx], prob_sub)
            
    # By Speaker
    by_spk = {}
    df = pd.DataFrame({"speaker": speakers, "y_true": y, "y_pred": y_pred})
    for spk, g in df.groupby("speaker"):
        by_spk[spk] = {
            "total": len(g),
            "accuracy": round(float((g["y_true"] == g["y_pred"]).mean()), 4),
            "class": "dysarthric" if g["y_true"].iloc[0] == 1 else "healthy"
        }
    return overall, by_lang, by_spk

def run_multilingual_training():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading multilingual features (English + Italian)...")
    X_train = np.load(PROCESSED_DIR / "multi_train_features.npy")
    y_train = np.load(PROCESSED_DIR / "multi_train_labels.npy")
    spk_train = np.load(PROCESSED_DIR / "multi_train_speakers.npy", allow_pickle=True)
    lang_train = np.load(PROCESSED_DIR / "multi_train_langs.npy", allow_pickle=True)
    
    X_val = np.load(PROCESSED_DIR / "multi_validation_features.npy")
    y_val = np.load(PROCESSED_DIR / "multi_validation_labels.npy")
    spk_val = np.load(PROCESSED_DIR / "multi_validation_speakers.npy", allow_pickle=True)
    lang_val = np.load(PROCESSED_DIR / "multi_validation_langs.npy", allow_pickle=True)
    
    print(f"Train: {len(X_train)} samples across {len(np.unique(spk_train))} speakers (EN: {np.sum(lang_train=='en')}, IT: {np.sum(lang_train=='it')})")
    print(f"Val:   {len(X_val)} samples across {len(np.unique(spk_val))} speakers (EN: {np.sum(lang_val=='en')}, IT: {np.sum(lang_val=='it')})")
    
    # Fit scaler strictly on training set
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    results = {}
    models = {}
    
    # -------------------------------------------------------------------------
    # 1. Logistic Regression (L2)
    # -------------------------------------------------------------------------
    print("\n[1/4] Training Logistic Regression...")
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    ov, bl, bs = evaluate_subsets(lr, scaler, X_val, y_val, lang_val, spk_val)
    results["Logistic Regression"] = {"overall": ov, "by_language": bl}
    models["Logistic Regression"] = lr
    print("  Val Overall:", ov)
    print("  Val English:", bl.get("en"))
    print("  Val Italian:", bl.get("it"))
    
    # -------------------------------------------------------------------------
    # 2. Random Forest
    # -------------------------------------------------------------------------
    print("\n[2/4] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    ov, bl, bs = evaluate_subsets(rf, scaler, X_val, y_val, lang_val, spk_val)
    results["Random Forest"] = {"overall": ov, "by_language": bl}
    models["Random Forest"] = rf
    print("  Val Overall:", ov)
    print("  Val English:", bl.get("en"))
    print("  Val Italian:", bl.get("it"))
    
    # -------------------------------------------------------------------------
    # 3. Histogram Gradient Boosting (GBDT)
    # -------------------------------------------------------------------------
    print("\n[3/4] Training Histogram Gradient Boosting (GBDT)...")
    hgb = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=0.5, early_stopping=True, random_state=42
    )
    hgb.fit(X_train_s, y_train)
    ov, bl, bs = evaluate_subsets(hgb, scaler, X_val, y_val, lang_val, spk_val)
    results["HistGradientBoosting"] = {"overall": ov, "by_language": bl}
    models["HistGradientBoosting"] = hgb
    print("  Val Overall:", ov)
    print("  Val English:", bl.get("en"))
    print("  Val Italian:", bl.get("it"))
    
    # -------------------------------------------------------------------------
    # 4. Multi-Layer Perceptron (Neural Network)
    # -------------------------------------------------------------------------
    print("\n[4/4] Training Regularized Multi-Layer Perceptron (MLP)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu", alpha=0.05,
        learning_rate_init=0.001, early_stopping=True, validation_fraction=0.15,
        max_iter=150, random_state=42
    )
    mlp.fit(X_train_s, y_train)
    ov, bl, bs = evaluate_subsets(mlp, scaler, X_val, y_val, lang_val, spk_val)
    results["MLP (Neural Network)"] = {"overall": ov, "by_language": bl}
    models["MLP (Neural Network)"] = mlp
    print("  Val Overall:", ov)
    print("  Val English:", bl.get("en"))
    print("  Val Italian:", bl.get("it"))

    # -------------------------------------------------------------------------
    # Model Comparison & Best Selection
    # -------------------------------------------------------------------------
    print("\n=======================================================")
    print("MULTILINGUAL MODEL COMPARISON (VALIDATION SPEAKERS)")
    print("=======================================================")
    rows = []
    for mname, mdata in results.items():
        ov = mdata["overall"]
        en = mdata["by_language"].get("en", {})
        it = mdata["by_language"].get("it", {})
        rows.append({
            "Model": mname,
            "Overall Acc": f"{ov['accuracy']*100:.2f}%",
            "Overall F1": f"{ov['f1_score']*100:.2f}%",
            "English Acc": f"{en.get('accuracy', 0)*100:.2f}%",
            "Italian Acc": f"{it.get('accuracy', 0)*100:.2f}%",
            "ROC-AUC": f"{ov['roc_auc']:.4f}",
            "raw_f1": ov["f1_score"]
        })
    df_cmp = pd.DataFrame(rows).sort_values(by="raw_f1", ascending=False)
    print(df_cmp[["Model", "Overall Acc", "Overall F1", "English Acc", "Italian Acc", "ROC-AUC"]].to_string(index=False))

    best_name = df_cmp.iloc[0]["Model"]
    best_model = models[best_name]
    print(f"\nSELECTED BEST MULTILINGUAL MODEL: {best_name}")
    
    # Save selected model and scaler
    selected_dir = MODELS_DIR / "selected_model"
    selected_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, selected_dir / "model.joblib")
    joblib.dump(scaler, selected_dir / "scaler.joblib")
    
    cfg = {
        "model_name": f"{best_name} (Multilingual English + Italian)",
        "feature_dim": 144,
        "sample_rate": 16000,
        "languages": ["en", "it"],
        "corpora": ["torgo", "easycall"],
        "total_speakers": 70,
        "classes": ["healthy", "dysarthric"],
        "class_mapping": {"0": "healthy", "1": "dysarthric"},
        "validation_metrics": results[best_name]["overall"],
        "validation_by_language": results[best_name]["by_language"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(selected_dir / "model_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
        
    with open(REPORTS_DIR / "multilingual_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved multilingual model artifacts to {selected_dir}")

if __name__ == "__main__":
    run_multilingual_training()
