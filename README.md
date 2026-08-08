# DATA110 Capstone Project — JAIS SAJI

Golden Gate University | DATA110: Introduction to Python using Machine Learning
Instructor: Dr. Durga Sharma

This repository contains three machine learning projects of increasing complexity, as required by the course capstone structure: a basic-level project, an intermediate-level project, and an advanced-level research project.

## Repository Structure

```
├── 01_basic_house_price/
│   ├── house_price_basic.py
│   ├── USA_Housing.csv
│   └── README.md
├── 02_intermediate_spam_detection/
│   ├── spam_detection_intermediate.py
│   ├── spam.csv
│   └── README.md
├── 03_advanced_football_prediction/
│   ├── football_pipeline.py
│   ├── results.csv
│   ├── football_prediction_paper_GGU.docx   <- Thesis report
│   └── README.md
└── README.md   <- this file
```

## Project Summaries

### 1. Basic: House Price Prediction (Linear Regression)
Predicts house prices from numeric features (income, house age, number of rooms, population) using a single Linear Regression model. Achieves an R² of 0.91 on held-out test data. Demonstrates fundamental regression workflow: load → clean → train → evaluate.

### 2. Intermediate: Spam Message Detection (Model Comparison)
Classifies SMS messages as spam or legitimate ("ham") using TF-IDF text vectorization and compares three classifiers: Naive Bayes, Logistic Regression, and Linear SVM. Best model (SVM) achieves 98.3% accuracy and 0.93 F1-score. Demonstrates text preprocessing, multi-model comparison, and precision/recall/F1 evaluation.

### 3. Advanced: Football Match Outcome Prediction (Primary Research Project)
A controlled empirical comparison of three fundamentally different modeling approaches — Elo rating (logistic regression), Poisson goals model, and XGBoost gradient boosting — for predicting international football match outcomes, using 30,250 real matches (1990–2024) with a leakage-safe, time-based train/test split.

**Key finding:** All three models converge to ~60.8% accuracy, and — more importantly — all three completely fail to predict draws as a top outcome (draws are predicted in under 0.1% of matches despite being 22.7% of real results). This finding is shown to be a structural property of the prediction task rather than a weakness of any single model, addressing a gap explicitly noted but not directly investigated in the existing sports-analytics literature (Bunker, Yeung, & Fujii, 2024).

Full write-up: `03_advanced_football_prediction/football_prediction_paper_GGU.docx`

## How to Run Any Project

Each subfolder is self-contained. Navigate into it and run:
```bash
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn
python <script_name>.py
```
See each subfolder's own README.md for exact filenames and details.

## Academic Integrity

All three projects were developed individually. Datasets are drawn from publicly available sources (cited in each project's README/report). All code is original implementation; no Kaggle notebook or existing solution was copied or replicated without substantial modification.
