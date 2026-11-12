"""
advanced_model_exploration.py
Rigorous exploration of advanced classifiers, synthetic feature augmentation,
and ensemble methods on strictly speaker-independent validation set.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    ExtraTreesClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

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

print("Loading data...")
X_train = np.load("data/processed/train_features.npy")
y_train = np.load("data/processed/train_labels.npy")
X_val = np.load("data/processed/validation_features.npy")
y_val = np.load("data/processed/validation_labels.npy")
X_test = np.load("data/processed/test_features.npy")
y_test = np.load("data/processed/test_labels.npy")

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_te_s = scaler.transform(X_test)

models = {}

# 1. Baseline Logistic Regression (C=0.1)
print("\n--- 1. Logistic Regression C=0.1 ---")
lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr.fit(X_tr_s, y_train)
m_val = calc(y_val, lr.predict(X_val_s), lr.predict_proba(X_val_s)[:, 1])
m_te = calc(y_test, lr.predict(X_te_s), lr.predict_proba(X_te_s)[:, 1])
print(f"Val Acc: {m_val['accuracy']*100:.2f}% | F1: {m_val['f1']*100:.2f}% | Test Acc: {m_te['accuracy']*100:.2f}% | Test F1: {m_te['f1']*100:.2f}%")
models["Logistic Regression"] = (lr, m_val, m_te)

# 2. Tuned Logistic Regression with higher regularization / elasticnet / balanced
print("\n--- 2. Logistic Regression (C=0.01, balanced) ---")
lr_bal = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=42)
lr_bal.fit(X_tr_s, y_train)
m_val = calc(y_val, lr_bal.predict(X_val_s), lr_bal.predict_proba(X_val_s)[:, 1])
m_te = calc(y_test, lr_bal.predict(X_te_s), lr_bal.predict_proba(X_te_s)[:, 1])
print(f"Val Acc: {m_val['accuracy']*100:.2f}% | F1: {m_val['f1']*100:.2f}% | Test Acc: {m_te['accuracy']*100:.2f}% | Test F1: {m_te['f1']*100:.2f}%")
models["Logistic Regression (C=0.01, balanced)"] = (lr_bal, m_val, m_te)

# 3. Support Vector Machine (RBF kernel, probability=True)
print("\n--- 3. SVM (RBF kernel, C=1.0) ---")
svm = SVC(C=1.0, kernel="rbf", probability=True, random_state=42)
svm.fit(X_tr_s, y_train)
m_val = calc(y_val, svm.predict(X_val_s), svm.predict_proba(X_val_s)[:, 1])
m_te = calc(y_test, svm.predict(X_te_s), svm.predict_proba(X_te_s)[:, 1])
print(f"Val Acc: {m_val['accuracy']*100:.2f}% | F1: {m_val['f1']*100:.2f}% | Test Acc: {m_te['accuracy']*100:.2f}% | Test F1: {m_te['f1']*100:.2f}%")
models["SVM (RBF C=1.0)"] = (svm, m_val, m_te)

# 4. Deep MLP Neural Network
print("\n--- 4. Tuned Multi-Layer Perceptron (128, 64) ---")
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    alpha=0.1,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.15,
    max_iter=200,
    random_state=42
)
mlp.fit(X_tr_s, y_train)
m_val = calc(y_val, mlp.predict(X_val_s), mlp.predict_proba(X_val_s)[:, 1])
m_te = calc(y_test, mlp.predict(X_te_s), mlp.predict_proba(X_te_s)[:, 1])
print(f"Val Acc: {m_val['accuracy']*100:.2f}% | F1: {m_val['f1']*100:.2f}% | Test Acc: {m_te['accuracy']*100:.2f}% | Test F1: {m_te['f1']*100:.2f}%")
models["MLP (128, 64)"] = (mlp, m_val, m_te)

# 5. Tuned ExtraTrees
print("\n--- 5. Extra Trees Classifier ---")
et = ExtraTreesClassifier(n_estimators=150, max_depth=14, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1)
et.fit(X_tr_s, y_train)
m_val = calc(y_val, et.predict(X_val_s), et.predict_proba(X_val_s)[:, 1])
m_te = calc(y_test, et.predict(X_te_s), et.predict_proba(X_te_s)[:, 1])
print(f"Val Acc: {m_val['accuracy']*100:.2f}% | F1: {m_val['f1']*100:.2f}% | Test Acc: {m_te['accuracy']*100:.2f}% | Test F1: {m_te['f1']*100:.2f}%")
models["ExtraTrees"] = (et, m_val, m_te)

# 6. Synthetic Manifold Augmentation (Mixup + Acoustic Jitter)
print("\n--- 6. Synthetic Manifold Augmentation (Mixup + Jitter) ---")
rng = np.random.default_rng(42)
synth_feats = []
synth_labels = []

# Generate 3,000 within-class synthetic combinations
for cls in [0, 1]:
    idx = np.where(y_train == cls)[0]
    for _ in range(1500):
        i1, i2 = rng.choice(idx, size=2, replace=False)
        lam = rng.beta(0.5, 0.5)
        comb = lam * X_train[i1] + (1 - lam) * X_train[i2]
        # Add 1% acoustic variance jitter
        jitter = rng.normal(0, 0.01 * np.std(X_train[idx], axis=0), size=comb.shape)
        synth_feats.append(comb + jitter)
        synth_labels.append(cls)

X_tr_aug = np.vstack([X_train, np.array(synth_feats, dtype=np.float32)])
y_tr_aug = np.concatenate([y_train, np.array(synth_labels, dtype=np.int64)])
X_tr_aug_s = scaler.transform(X_tr_aug)

lr_aug = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr_aug.fit(X_tr_aug_s, y_tr_aug)
m_val = calc(y_val, lr_aug.predict(X_val_s), lr_aug.predict_proba(X_val_s)[:, 1])
m_te = calc(y_test, lr_aug.predict(X_te_s), lr_aug.predict_proba(X_te_s)[:, 1])
print(f"Logistic Regression + Synth: Val Acc: {m_val['accuracy']*100:.2f}% | F1: {m_val['f1']*100:.2f}% | Test Acc: {m_te['accuracy']*100:.2f}% | Test F1: {m_te['f1']*100:.2f}%")
models["Logistic Regression + Synthetic Augmentation"] = (lr_aug, m_val, m_te)

# 7. Soft-Voting Clinical Ensemble (LR + SVM + MLP)
print("\n--- 7. Soft-Voting Clinical Ensemble ---")
ensemble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("svm", svm),
        ("mlp", mlp)
    ],
    voting="soft"
)
ensemble.fit(X_tr_s, y_train)
m_val = calc(y_val, ensemble.predict(X_val_s), ensemble.predict_proba(X_val_s)[:, 1])
m_te = calc(y_test, ensemble.predict(X_te_s), ensemble.predict_proba(X_te_s)[:, 1])
print(f"Clinical Ensemble: Val Acc: {m_val['accuracy']*100:.2f}% | F1: {m_val['f1']*100:.2f}% | Test Acc: {m_te['accuracy']*100:.2f}% | Test F1: {m_te['f1']*100:.2f}%")
models["Clinical Soft-Voting Ensemble"] = (ensemble, m_val, m_te)
