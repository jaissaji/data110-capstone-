# Basic Project: House Price Prediction (Linear Regression)

## Goal
Predict house prices from simple numeric features using a single Linear Regression model. This is the simplest of the three capstone projects — one dataset, one algorithm, minimal preprocessing — intended to demonstrate the fundamental regression workflow.

## Dataset
`USA_Housing.csv` — 5,000 house records with features: average area income, house age, number of rooms, number of bedrooms, and area population. Target variable: Price.
Source: publicly available beginner regression dataset (commonly used in introductory ML tutorials).

## Method
- 70/30 train-test split (random_state=42 for reproducibility)
- Single scikit-learn `LinearRegression` model, no regularization
- No feature engineering beyond dropping the non-numeric Address column

## Results
- R² score: 0.91 (the model explains ~91% of the variation in house prices)
- MAE: ~$81,000, RMSE: ~$100,000

## Files
- `house_price_basic.py` — full pipeline (load → EDA → train → evaluate)
- `USA_Housing.csv` — dataset
- `basic_fig1_correlation_heatmap.png` — feature correlation heatmap
- `basic_fig2_actual_vs_predicted.png` — scatter plot of actual vs. predicted prices

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python house_price_basic.py
```
