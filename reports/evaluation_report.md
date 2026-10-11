# Final Evaluation Report: Dysarthria Detection System

## 1. Evaluation Protocol & Rigor
- **Evaluation Type:** Strictly Speaker-Independent Held-Out Test Evaluation.
- **Zero Leakage:** No test speaker or recording was ever seen during feature normalization, hyperparameter tuning, or model fitting.
- **Test Subjects:** 13 completely unseen human speakers (6 healthy controls, 7 dysarthric individuals).
- **Test Sample Size:** 5150 recordings across varying phonemes and smartphone voice commands.

## 2. Final Held-Out Test Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **87.36%** | Overall correct classification rate on unseen individuals. |
| **Dysarthria Recall** | **82.81%** | Sensitivity: fraction of dysarthric utterances correctly caught. |
| **Dysarthria Precision** | **91.96%** | Positive Predictive Value. |
| **F1-Score** | **87.15%** | Harmonic balance between recall and precision. |
| **Specificity** | **92.23%** | True Negative Rate: Healthy speech correctly identified as healthy. |
| **ROC-AUC** | **0.9545** | Area under the Receiver Operating Characteristic curve. |

## 3. Confusion Matrix Breakdown
- **True Negatives (Healthy -> Healthy):** 2292 / 2485 (92.2%)
- **False Positives (Healthy -> Dysarthric):** 193 / 2485 (7.8%)
- **False Negatives (Dysarthric -> Healthy):** 458 / 2665 (17.2%)
- **True Positives (Dysarthric -> Dysarthric):** 2207 / 2665 (82.8%)

## 4. Per-Speaker Performance Analysis
The table below documents performance on every individual held-out speaker:

| Speaker ID | Ground Truth Class | Clinical Severity | Total Samples | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| `f04` | dysarthric | nan | 233 | **97.0%** |
| `f06` | dysarthric | 1.0 | 414 | **91.8%** |
| `f09` | dysarthric | 1.0 | 414 | **95.7%** |
| `fc04` | healthy | 0.0 | 394 | **95.9%** |
| `fc07` | healthy | 0.0 | 411 | **76.6%** |
| `fc10` | healthy | 0.0 | 410 | **99.8%** |
| `m10` | dysarthric | 2.0 | 405 | **78.3%** |
| `m11` | dysarthric | 3.0 | 371 | **97.3%** |
| `m12` | dysarthric | 1.0 | 414 | **96.1%** |
| `m17` | dysarthric | 1.0 | 414 | **31.2%** |
| `mc04` | healthy | 0.0 | 395 | **83.5%** |
| `mc07` | healthy | 0.0 | 461 | **97.8%** |
| `mc11` | healthy | 0.0 | 414 | **98.8%** |

## 5. Limitations and Responsible Interpretation
1. **Speaker-level variation:** Aggregate metrics are utterance-level and conceal large differences between held-out participants. For example, dysarthric speaker `m17` has 31.2% utterance accuracy, while several other speakers are above 90%. This variation limits claims about performance on a new individual.
2. **Severity:** The small held-out speaker sample does not support a reliable severity-performance conclusion. The model predicts only a binary class and does not estimate clinical severity.
3. **No feature explanations:** The classifier output is not an explanation of which acoustic characteristics caused an individual prediction. The displayed class scores are not calibrated clinical probabilities.
4. **Medical disclaimer:** This is an engineering prototype for academic demonstration, not a clinically validated screening or diagnostic device.
