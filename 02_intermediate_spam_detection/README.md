# Intermediate Project: Spam Message Detection (Model Comparison)

## Goal
Classify SMS text messages as spam or legitimate ("ham"), comparing three different classification algorithms: Naive Bayes, Logistic Regression, and Linear SVM. This is intermediate-level because it involves text preprocessing, multiple models compared side by side, and multiple evaluation metrics beyond accuracy.

## Dataset
`spam.csv` — 5,572 real SMS messages labeled ham (4,825) or spam (747).
Source: SMS Spam Collection Dataset (publicly available, widely used benchmark for spam classification).

## Method
- 70/30 stratified train-test split (random_state=42)
- Text converted to numeric features using TF-IDF vectorization (3,000 features, English stop words removed)
- Three models trained and compared: Multinomial Naive Bayes, Logistic Regression, Linear SVM

## Results
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Naive Bayes | 97.6% | 0.989 | 0.830 | 0.903 |
| Logistic Regression | 96.7% | 0.994 | 0.759 | 0.861 |
| **SVM (Linear)** | **98.3%** | **0.985** | **0.888** | **0.934** |

SVM achieved the best overall performance (highest F1-score), balancing precision and recall better than the alternatives.

## Files
- `spam_detection_intermediate.py` — full pipeline (load → TF-IDF → train 3 models → compare)
- `spam.csv` — dataset
- `intermediate_fig1_model_comparison.png` — bar chart comparing all models across metrics
- `intermediate_fig2_confusion_matrices.png` — confusion matrices for all three models

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python spam_detection_intermediate.py
```
