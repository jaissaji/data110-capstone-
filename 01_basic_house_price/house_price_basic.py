"""
============================================================
BASIC PROJECT: House Price Prediction using Linear Regression
============================================================
Goal: Predict house price from simple numeric features
      (income, house age, number of rooms, bedrooms, population)
      using a single Linear Regression model.

This is intentionally the SIMPLEST of the three projects:
one dataset, one model, minimal preprocessing.

HOW TO RUN:
    python house_price_basic.py

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn

Make sure "USA_Housing.csv" is in the same folder as this script.
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style('whitegrid')

# ------------------------------------------------------------
# STEP 1: Load and explore the data
# ------------------------------------------------------------
print("="*60)
print("STEP 1: Loading and exploring the data")
print("="*60)

df = pd.read_csv('USA_Housing.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop the Address column - it's just text and not useful for prediction
df = df.drop(columns=['Address'])

# ------------------------------------------------------------
# STEP 2: Exploratory Data Analysis (EDA)
# ------------------------------------------------------------
print("\n" + "="*60)
print("STEP 2: Exploratory Data Analysis")
print("="*60)

print(df.describe())

# Correlation heatmap - shows which features relate most to Price
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Housing Features')
plt.tight_layout()
plt.savefig('basic_fig1_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved basic_fig1_correlation_heatmap.png")

# ------------------------------------------------------------
# STEP 3: Prepare data and train the model
# ------------------------------------------------------------
print("\n" + "="*60)
print("STEP 3: Training the Linear Regression model")
print("="*60)

X = df.drop(columns=['Price'])
y = df['Price']

# 70-30 train-test split, random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained.")
print("\nModel coefficients (how much each feature affects price):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:<35} {coef:>15.2f}")
print(f"  {'Intercept':<35} {model.intercept_:>15.2f}")

# ------------------------------------------------------------
# STEP 4: Evaluate the model
# ------------------------------------------------------------
print("\n" + "="*60)
print("STEP 4: Evaluating the model")
print("="*60)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE):  {mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")
print(f"(R2 of {r2:.2f} means the model explains about {r2*100:.0f}% of the variation in house prices)")

# Scatter plot: actual vs predicted prices
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.4, color='#4C72B0')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.savefig('basic_fig2_actual_vs_predicted.png', bbox_inches='tight')
plt.close()
print("Saved basic_fig2_actual_vs_predicted.png")

print("\n" + "="*60)
print("ALL DONE. Check the folder for the .png figures.")
print("="*60)
