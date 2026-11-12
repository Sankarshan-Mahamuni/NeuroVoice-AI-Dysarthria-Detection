"""
fast_model_experiments.py
Trains and benchmarks fast classifiers and synthetic feature augmentations
with unbuffered output to systematically verify performance.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

def log(msg):
    print(msg, flush=True)

def calc(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0
    return {
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "recall": round(float(rec), 4),
        "precision": round(float(prec), 4),
        "specificity": round(float(spec), 4),
        "roc_auc": round(float(auc), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }

log("Loading data...")
X_train = np.load("data/processed/train_features.npy")
y_train = np.load("data/processed/train_labels.npy")
spk_train = np.load("data/processed/train_speakers.npy", allow_pickle=True)

X_val = np.load("data/processed/validation_features.npy")
y_val = np.load("data/processed/validation_labels.npy")
spk_val = np.load("data/processed/validation_speakers.npy", allow_pickle=True)

X_test = np.load("data/processed/test_features.npy")
y_test = np.load("data/processed/test_labels.npy")
spk_test = np.load("data/processed/test_speakers.npy", allow_pickle=True)

log(f"Train: {X_train.shape[0]} samples, {len(np.unique(spk_train))} speakers")
log(f"Val:   {X_val.shape[0]} samples, {len(np.unique(spk_val))} speakers")
log(f"Test:  {X_test.shape[0]} samples, {len(np.unique(spk_test))} speakers")

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_te_s = scaler.transform(X_test)

results = {}

# 1. Logistic Regression C=0.1
log("\n[1] Logistic Regression (L2, C=0.1)...")
lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr.fit(X_tr_s, y_train)
v = calc(y_val, lr.predict(X_val_s), lr.predict_proba(X_val_s)[:, 1])
t = calc(y_test, lr.predict(X_te_s), lr.predict_proba(X_te_s)[:, 1])
log(f"  Val  -> Acc: {v['accuracy']*100:.2f}% | F1: {v['f1']*100:.2f}% | Rec: {v['recall']*100:.2f}% | Spec: {v['specificity']*100:.2f}% | AUC: {v['roc_auc']:.4f}")
log(f"  Test -> Acc: {t['accuracy']*100:.2f}% | F1: {t['f1']*100:.2f}% | Rec: {t['recall']*100:.2f}% | Spec: {t['specificity']*100:.2f}% | AUC: {t['roc_auc']:.4f}")
results["Logistic Regression (C=0.1)"] = {"val": v, "test": t}

# 2. Logistic Regression (C=0.02, balanced)
log("\n[2] Logistic Regression (C=0.02, balanced)...")
lr_bal = LogisticRegression(C=0.02, class_weight="balanced", max_iter=1000, random_state=42)
lr_bal.fit(X_tr_s, y_train)
v = calc(y_val, lr_bal.predict(X_val_s), lr_bal.predict_proba(X_val_s)[:, 1])
t = calc(y_test, lr_bal.predict(X_te_s), lr_bal.predict_proba(X_te_s)[:, 1])
log(f"  Val  -> Acc: {v['accuracy']*100:.2f}% | F1: {v['f1']*100:.2f}% | Rec: {v['recall']*100:.2f}% | Spec: {v['specificity']*100:.2f}% | AUC: {v['roc_auc']:.4f}")
log(f"  Test -> Acc: {t['accuracy']*100:.2f}% | F1: {t['f1']*100:.2f}% | Rec: {t['recall']*100:.2f}% | Spec: {t['specificity']*100:.2f}% | AUC: {t['roc_auc']:.4f}")
results["Logistic Regression (C=0.02, balanced)"] = {"val": v, "test": t}

# 3. Fast HistGradientBoosting
log("\n[3] HistGradientBoosting (GBDT)...")
hgb = HistGradientBoostingClassifier(
    max_iter=150, learning_rate=0.04, max_leaf_nodes=31,
    min_samples_leaf=20, l2_regularization=0.5, early_stopping=True, random_state=42
)
hgb.fit(X_tr_s, y_train)
v = calc(y_val, hgb.predict(X_val_s), hgb.predict_proba(X_val_s)[:, 1])
t = calc(y_test, hgb.predict(X_te_s), hgb.predict_proba(X_te_s)[:, 1])
log(f"  Val  -> Acc: {v['accuracy']*100:.2f}% | F1: {v['f1']*100:.2f}% | Rec: {v['recall']*100:.2f}% | Spec: {v['specificity']*100:.2f}% | AUC: {v['roc_auc']:.4f}")
log(f"  Test -> Acc: {t['accuracy']*100:.2f}% | F1: {t['f1']*100:.2f}% | Rec: {t['recall']*100:.2f}% | Spec: {t['specificity']*100:.2f}% | AUC: {t['roc_auc']:.4f}")
results["HistGradientBoosting"] = {"val": v, "test": t}

# 4. Multi-Layer Perceptron (64, 32)
log("\n[4] Regularized MLP Neural Network (64, 32)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), activation="relu", alpha=0.1,
    learning_rate_init=0.001, early_stopping=True, validation_fraction=0.15,
    max_iter=120, random_state=42
)
mlp.fit(X_tr_s, y_train)
v = calc(y_val, mlp.predict(X_val_s), mlp.predict_proba(X_val_s)[:, 1])
t = calc(y_test, mlp.predict(X_te_s), mlp.predict_proba(X_te_s)[:, 1])
log(f"  Val  -> Acc: {v['accuracy']*100:.2f}% | F1: {v['f1']*100:.2f}% | Rec: {v['recall']*100:.2f}% | Spec: {v['specificity']*100:.2f}% | AUC: {v['roc_auc']:.4f}")
log(f"  Test -> Acc: {t['accuracy']*100:.2f}% | F1: {t['f1']*100:.2f}% | Rec: {t['recall']*100:.2f}% | Spec: {t['specificity']*100:.2f}% | AUC: {t['roc_auc']:.4f}")
results["MLP (64, 32)"] = {"val": v, "test": t}

# 5. Synthetic Feature Augmentation (Mixup + Jitter on Training Data)
log("\n[5] Synthetic Augmentation (Within-Class Mixup + Acoustic Jitter)...")
rng = np.random.default_rng(42)
synth_X = []
synth_y = []
for c in [0, 1]:
    idx = np.where(y_train == c)[0]
    for _ in range(1200):
        i1, i2 = rng.choice(idx, size=2, replace=False)
        lam = rng.beta(0.5, 0.5)
        syn = lam * X_train[i1] + (1 - lam) * X_train[i2]
        jitter = rng.normal(0, 0.008 * np.std(X_train[idx], axis=0), size=syn.shape)
        synth_X.append(syn + jitter)
        synth_y.append(c)

X_tr_aug = np.vstack([X_train, np.array(synth_X, dtype=np.float32)])
y_tr_aug = np.concatenate([y_train, np.array(synth_y, dtype=np.int64)])
X_tr_aug_s = scaler.transform(X_tr_aug)

lr_aug = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr_aug.fit(X_tr_aug_s, y_tr_aug)
v = calc(y_val, lr_aug.predict(X_val_s), lr_aug.predict_proba(X_val_s)[:, 1])
t = calc(y_test, lr_aug.predict(X_te_s), lr_aug.predict_proba(X_te_s)[:, 1])
log(f"  Val  -> Acc: {v['accuracy']*100:.2f}% | F1: {v['f1']*100:.2f}% | Rec: {v['recall']*100:.2f}% | Spec: {v['specificity']*100:.2f}% | AUC: {v['roc_auc']:.4f}")
log(f"  Test -> Acc: {t['accuracy']*100:.2f}% | F1: {t['f1']*100:.2f}% | Rec: {t['recall']*100:.2f}% | Spec: {t['specificity']*100:.2f}% | AUC: {t['roc_auc']:.4f}")
results["Logistic Regression + Synthetic Augmentation"] = {"val": v, "test": t}

# 6. Clinical Soft-Voting Ensemble (LR + GBDT + MLP)
log("\n[6] Clinical Soft-Voting Ensemble (LR + GBDT + MLP)...")
ens = VotingClassifier(
    estimators=[("lr", lr), ("hgb", hgb), ("mlp", mlp)],
    voting="soft",
    weights=[2, 1, 1]
)
ens.fit(X_tr_s, y_train)
v = calc(y_val, ens.predict(X_val_s), ens.predict_proba(X_val_s)[:, 1])
t = calc(y_test, ens.predict(X_te_s), ens.predict_proba(X_te_s)[:, 1])
log(f"  Val  -> Acc: {v['accuracy']*100:.2f}% | F1: {v['f1']*100:.2f}% | Rec: {v['recall']*100:.2f}% | Spec: {v['specificity']*100:.2f}% | AUC: {v['roc_auc']:.4f}")
log(f"  Test -> Acc: {t['accuracy']*100:.2f}% | F1: {t['f1']*100:.2f}% | Rec: {t['recall']*100:.2f}% | Spec: {t['specificity']*100:.2f}% | AUC: {t['roc_auc']:.4f}")
results["Clinical Soft-Voting Ensemble"] = {"val": v, "test": t}

# 7. Check Per-Speaker Breakdown for Best Model
log("\n--- Per-Speaker Breakdown for Best Model (LR C=0.1) on Test Speakers ---")
df_te = pd.DataFrame({"speaker": spk_test, "y_true": y_test, "y_pred": lr.predict(X_te_s)})
for s, g in df_te.groupby("speaker"):
    s_acc = (g["y_true"] == g["y_pred"]).mean()
    s_cls = "Dysarthric" if g["y_true"].iloc[0] == 1 else "Healthy"
    log(f"  Speaker {s} ({s_cls}): {s_acc*100:.1f}% accuracy ({len(g)} recordings)")

import json
with open("reports/comprehensive_benchmarks.json", "w") as f:
    json.dump(results, f, indent=2)
log("\nSaved benchmarks to reports/comprehensive_benchmarks.json")
