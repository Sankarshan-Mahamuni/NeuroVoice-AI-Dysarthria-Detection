"""
train_and_compare.py
Trains baseline and advanced classifiers on the training set (31 speakers),
evaluates and tunes on the held-out validation set (11 unseen speakers),
and conducts a controlled synthetic-data augmentation experiment.
Strictly prevents any speaker or test-set leakage.
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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

def calculate_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Specificity = TN / (TN + FP)
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
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
    }

def evaluate_per_speaker(model, scaler, X_val, y_val, speakers_val):
    X_val_scaled = scaler.transform(X_val) if scaler else X_val
    y_pred = model.predict(X_val_scaled)
    
    df = pd.DataFrame({
        "speaker": speakers_val,
        "y_true": y_val,
        "y_pred": y_pred
    })
    
    spk_metrics = {}
    for spk, g in df.groupby("speaker"):
        spk_acc = (g["y_true"] == g["y_pred"]).mean()
        spk_class = "dysarthric" if g["y_true"].iloc[0] == 1 else "healthy"
        spk_metrics[spk] = {
            "class": spk_class,
            "total_samples": len(g),
            "accuracy": round(float(spk_acc), 4)
        }
    return spk_metrics

def run_experiments():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print("Loading processed features and speaker labels...")
    X_train = np.load(PROCESSED_DIR / "train_features.npy")
    y_train = np.load(PROCESSED_DIR / "train_labels.npy")
    spk_train = np.load(PROCESSED_DIR / "train_speakers.npy", allow_pickle=True)
    
    X_val = np.load(PROCESSED_DIR / "validation_features.npy")
    y_val = np.load(PROCESSED_DIR / "validation_labels.npy")
    spk_val = np.load(PROCESSED_DIR / "validation_speakers.npy", allow_pickle=True)
    
    print(f"Train samples: {len(X_train)} ({len(np.unique(spk_train))} speakers)")
    print(f"Val samples:   {len(X_val)} ({len(np.unique(spk_val))} speakers)")
    
    # Fit StandardScaler strictly on training set only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    results = {}
    models_dict = {}
    
    # -------------------------------------------------------------------------
    # Experiment 1: Baseline Logistic Regression (L2 regularized)
    # -------------------------------------------------------------------------
    print("\n[1/5] Training Baseline: Logistic Regression (L2)...")
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_val_scaled)
    y_prob_lr = lr.predict_proba(X_val_scaled)[:, 1]
    metrics_lr = calculate_metrics(y_val, y_pred_lr, y_prob_lr)
    results["Logistic Regression (Baseline)"] = metrics_lr
    models_dict["Logistic Regression (Baseline)"] = (lr, scaler)
    print("  Validation Results:", metrics_lr)
    
    # -------------------------------------------------------------------------
    # Experiment 2: Random Forest Classifier
    # -------------------------------------------------------------------------
    print("\n[2/5] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_val_scaled)
    y_prob_rf = rf.predict_proba(X_val_scaled)[:, 1]
    metrics_rf = calculate_metrics(y_val, y_pred_rf, y_prob_rf)
    results["Random Forest"] = metrics_rf
    models_dict["Random Forest"] = (rf, scaler)
    print("  Validation Results:", metrics_rf)
    
    # -------------------------------------------------------------------------
    # Experiment 3: Histogram Gradient Boosting (GBDT)
    # -------------------------------------------------------------------------
    print("\n[3/5] Training Histogram Gradient Boosting (GBDT)...")
    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.5,
        early_stopping=True,
        random_state=42
    )
    hgb.fit(X_train_scaled, y_train)
    y_pred_hgb = hgb.predict(X_val_scaled)
    y_prob_hgb = hgb.predict_proba(X_val_scaled)[:, 1]
    metrics_hgb = calculate_metrics(y_val, y_pred_hgb, y_prob_hgb)
    results["HistGradientBoosting (GBDT)"] = metrics_hgb
    models_dict["HistGradientBoosting (GBDT)"] = (hgb, scaler)
    print("  Validation Results:", metrics_hgb)
    
    # -------------------------------------------------------------------------
    # Experiment 4: Regularized Multi-Layer Perceptron (Neural Network)
    # -------------------------------------------------------------------------
    print("\n[4/5] Training Regularized Multi-Layer Perceptron (MLP)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=0.05,  # Stronger L2 regularization to prevent overfitting
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.15,
        max_iter=150,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp.predict(X_val_scaled)
    y_prob_mlp = mlp.predict_proba(X_val_scaled)[:, 1]
    metrics_mlp = calculate_metrics(y_val, y_pred_mlp, y_prob_mlp)
    results["MLP (Neural Network)"] = metrics_mlp
    models_dict["MLP (Neural Network)"] = (mlp, scaler)
    print("  Validation Results:", metrics_mlp)

    # -------------------------------------------------------------------------
    # Experiment 5: Controlled Synthetic Augmentation Experiment
    # -------------------------------------------------------------------------
    print("\n[5/5] Investigating Controlled Synthetic Feature Augmentation...")
    # Synthetic Mixup on training data only: convex combination of within-class feature vectors
    # to simulate inter-speaker acoustic continuum without cross-class label corruption
    rng = np.random.default_rng(42)
    synthetic_X = []
    synthetic_y = []
    
    # Generate 2,000 synthetic samples within training set (1000 healthy, 1000 dysarthric)
    for target_class in [0, 1]:
        idx_class = np.where(y_train == target_class)[0]
        for _ in range(1000):
            i1, i2 = rng.choice(idx_class, size=2, replace=False)
            lam = rng.beta(0.4, 0.4)
            syn_feat = lam * X_train[i1] + (1 - lam) * X_train[i2]
            synthetic_X.append(syn_feat)
            synthetic_y.append(target_class)
            
    X_train_aug = np.vstack([X_train, np.array(synthetic_X, dtype=np.float32)])
    y_train_aug = np.concatenate([y_train, np.array(synthetic_y, dtype=np.int64)])
    
    # Scale augmented data using the same training-fitted scaler
    X_train_aug_scaled = scaler.transform(X_train_aug)
    
    print(f"  Training HistGradientBoosting on Augmented Set ({len(X_train_aug)} samples)...")
    hgb_aug = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.5,
        early_stopping=True,
        random_state=42
    )
    hgb_aug.fit(X_train_aug_scaled, y_train_aug)
    y_pred_aug = hgb_aug.predict(X_val_scaled)
    y_prob_aug = hgb_aug.predict_proba(X_val_scaled)[:, 1]
    metrics_aug = calculate_metrics(y_val, y_pred_aug, y_prob_aug)
    results["HistGradientBoosting + Synthetic Augmentation"] = metrics_aug
    models_dict["HistGradientBoosting + Synthetic Augmentation"] = (hgb_aug, scaler)
    print("  Validation Results with Synthetic Data:", metrics_aug)
    
    # -------------------------------------------------------------------------
    # Compare Models and Select the Best Candidate
    # -------------------------------------------------------------------------
    print("\n=======================================================")
    print("MODEL COMPARISON SUMMARY (HELD-OUT VALIDATION SPEAKERS)")
    print("=======================================================")
    summary_rows = []
    for model_name, m in results.items():
        summary_rows.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "F1-Score": f"{m['f1_score']*100:.2f}%",
            "Recall (Dysarthria)": f"{m['recall']*100:.2f}%",
            "Specificity": f"{m['specificity']*100:.2f}%",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "raw_f1": m['f1_score'],
            "raw_acc": m['accuracy']
        })
    df_summary = pd.DataFrame(summary_rows).sort_values(by="raw_f1", ascending=False)
    print(df_summary[["Model", "Accuracy", "F1-Score", "Recall (Dysarthria)", "Specificity", "ROC-AUC"]].to_string(index=False))
    
    best_model_name = df_summary.iloc[0]["Model"]
    best_model_obj, best_scaler = models_dict[best_model_name]
    print(f"\nSELECTED BEST MODEL: {best_model_name}")
    
    # Evaluate per-speaker performance for the best model on validation speakers
    spk_val_eval = evaluate_per_speaker(best_model_obj, best_scaler, X_val, y_val, spk_val)
    print("\nPer-Speaker Validation Performance for Best Model:")
    for spk, sinfo in spk_val_eval.items():
        print(f"  Speaker {spk} ({sinfo['class']}): {sinfo['accuracy']*100:.1f}% accuracy ({sinfo['total_samples']} samples)")
        
    # -------------------------------------------------------------------------
    # Save the Selected Model, Scaler, and Configuration
    # -------------------------------------------------------------------------
    selected_dir = MODELS_DIR / "selected_model"
    selected_dir.mkdir(parents=True, exist_ok=True)
    
    model_save_path = selected_dir / "model.joblib"
    scaler_save_path = selected_dir / "scaler.joblib"
    
    joblib.dump(best_model_obj, model_save_path)
    joblib.dump(best_scaler, scaler_save_path)
    
    config = {
        "model_name": best_model_name,
        "feature_dim": 144,
        "sample_rate": 16000,
        "classes": ["healthy", "dysarthric"],
        "class_mapping": {"0": "healthy", "1": "dysarthric"},
        "threshold": 0.50,
        "validation_metrics": results[best_model_name],
        "training_speaker_count": int(len(np.unique(spk_train))),
        "validation_speaker_count": int(len(np.unique(spk_val))),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(selected_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    # Save comparison report
    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved best model to: {model_save_path}")
    print(f"Saved scaler to: {scaler_save_path}")
    print(f"Saved configuration to: {selected_dir / 'model_config.json'}")

if __name__ == "__main__":
    run_experiments()
