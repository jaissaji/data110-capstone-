# Advanced Project: Football Match Outcome Prediction (Primary Research Project)

## Goal
A controlled empirical comparison of three fundamentally different machine learning / statistical approaches for predicting international football match outcomes: Elo rating (logistic regression), a Poisson goals model, and XGBoost gradient boosting. This is the primary/advanced capstone project.

## Research Gap
Prior literature (e.g., Bunker, Yeung, & Fujii, 2024) notes that draw prediction is difficult but is typically sidestepped — excluded from the dataset or merged into a binary win/not-win problem — rather than directly investigated. This project treats draw misprediction as the central research question: is it a fixable weakness of one specific model, or a structural property shared across model families?

## Dataset
`results.csv` — 30,250 real international football matches (1990–2024), sourced from a public GitHub mirror of the standard international-results dataset.

## Method
- Elo rating system implemented from scratch (standard football Elo update rule, home-advantage adjustment, goal-difference-scaled K-factor)
- Feature engineering: recent form, rest days, tournament tier, neutral venue
- Three models trained: Elo-logistic regression, Poisson goals model, XGBoost (random_state=42)
- Leakage-safe time-based split: train on 1990–2018, test on 2019–2024

## Key Results
| Model | Accuracy | Log-Loss | Brier Score |
|---|---|---|---|
| Elo (logistic) | 60.87% | 0.8663 | 0.5084 |
| Poisson (goals) | 60.74% | 0.8681 | 0.5092 |
| XGBoost | 60.81% | 0.8687 | 0.5091 |
| Majority baseline | 48.02% | 1.0491 | 0.6325 |

**Central finding:** All three models predict "draw" as the top outcome in under 0.1% of matches, despite draws being 22.7% of real results — identical across all three model families, indicating this is a structural limitation of the prediction task, not a model-specific weakness.

## Files
- `football_pipeline.py` — full single-file pipeline (data cleaning → Elo → features → 3 models → error analysis → figures)
- `results.csv` — raw match dataset
- `football_prediction_paper_GGU.docx` — full thesis report (Golden Gate University DATA110 format, APA references)
- `fig1_model_comparison.png`, `fig2_confusion_matrices.png`, `fig3_feature_importance.png` — figures used in the report
- `results_summary.json`, `feature_importance.json` — raw numeric results

## How to Run
```bash
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn
python football_pipeline.py
```
