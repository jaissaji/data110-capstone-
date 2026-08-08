"""
============================================================
FOOTBALL MATCH OUTCOME PREDICTION - COMPLETE PIPELINE
============================================================
This single file does everything, in order:
  STEP 1: Clean the data + compute Elo ratings
  STEP 2: Build extra features (form, rest days, etc.)
  STEP 3: Train & evaluate 3 models (Elo, Poisson, XGBoost)
  STEP 4: Analyze errors + create the figures for the paper

HOW TO RUN THIS FILE (see full instructions from Claude below):
    python football_pipeline.py

Requirements (install once):
    pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn

Make sure "results.csv" is in the SAME FOLDER as this script.
============================================================
"""

import pandas as pd
import numpy as np

print("="*60)
print("STEP 1: Cleaning data and computing Elo ratings")
print("="*60)

# ---- Load and clean the raw match data ----
df = pd.read_csv('results.csv')
df['date'] = pd.to_datetime(df['date'], format='mixed')
df = df.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score']).copy()
df = df.sort_values('date').reset_index(drop=True)

# Restrict to modern era for data quality + relevance (post-1990)
df = df[df['date'] >= '1990-01-01'].reset_index(drop=True)

df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)

def result_label(row):
    if row['home_score'] > row['away_score']:
        return 'H'
    elif row['home_score'] < row['away_score']:
        return 'A'
    else:
        return 'D'

df['result'] = df.apply(result_label, axis=1)
df['neutral'] = df['neutral'].astype(str).str.upper() == 'TRUE'

print(f"Matches after cleaning (1990+): {len(df)}")
print(df['result'].value_counts(normalize=True))

# ---- Elo Rating System ----
K_BASE = 20
HOME_ADV = 100

elo = {}
DEFAULT_ELO = 1500

def get_elo(team):
    return elo.get(team, DEFAULT_ELO)

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def goal_diff_multiplier(gd):
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return (11 + gd) / 8

elo_home_pre, elo_away_pre = [], []

for idx, row in df.iterrows():
    home, away = row['home_team'], row['away_team']
    r_home, r_away = get_elo(home), get_elo(away)
    elo_home_pre.append(r_home)
    elo_away_pre.append(r_away)

    r_home_eff = r_home + (0 if row['neutral'] else HOME_ADV)

    exp_home = expected_score(r_home_eff, r_away)
    exp_away = 1 - exp_home

    if row['result'] == 'H':
        actual_home, actual_away = 1, 0
    elif row['result'] == 'A':
        actual_home, actual_away = 0, 1
    else:
        actual_home, actual_away = 0.5, 0.5

    gd = abs(row['home_score'] - row['away_score'])
    mult = goal_diff_multiplier(gd)

    new_r_home = r_home + K_BASE * mult * (actual_home - exp_home)
    new_r_away = r_away + K_BASE * mult * (actual_away - exp_away)

    elo[home] = new_r_home
    elo[away] = new_r_away

df['home_elo_pre'] = elo_home_pre
df['away_elo_pre'] = elo_away_pre
df['elo_diff'] = df['home_elo_pre'] - df['away_elo_pre']

print("Elo ratings computed.\n")


print("="*60)
print("STEP 2: Building extra features (form, rest days, etc.)")
print("="*60)

df = df.sort_values('date').reset_index(drop=True)

team_history = {}
last_played = {}

def get_form(team, window=5):
    hist = team_history.get(team, [])
    if not hist:
        return 1.0
    recent = hist[-window:]
    return np.mean([p for _, p in recent])

def get_rest_days(team, current_date):
    if team not in last_played:
        return 30
    return (current_date - last_played[team]).days

home_form, away_form = [], []
home_rest, away_rest = [], []

for idx, row in df.iterrows():
    home, away, date = row['home_team'], row['away_team'], row['date']

    home_form.append(get_form(home))
    away_form.append(get_form(away))
    home_rest.append(min(get_rest_days(home, date), 365))
    away_rest.append(min(get_rest_days(away, date), 365))

    if row['result'] == 'H':
        h_pts, a_pts = 3, 0
    elif row['result'] == 'A':
        h_pts, a_pts = 0, 3
    else:
        h_pts, a_pts = 1, 1

    team_history.setdefault(home, []).append((date, h_pts))
    team_history.setdefault(away, []).append((date, a_pts))
    last_played[home] = date
    last_played[away] = date

df['home_form'] = home_form
df['away_form'] = away_form
df['home_rest_days'] = home_rest
df['away_rest_days'] = away_rest
df['form_diff'] = df['home_form'] - df['away_form']
df['rest_diff'] = df['home_rest_days'] - df['away_rest_days']

major = ['FIFA World Cup', 'UEFA Euro', 'Copa América', 'African Cup of Nations', 'CONCACAF Championship']
qual = df['tournament'].str.contains('qualification', case=False, na=False)
df['tournament_tier'] = np.where(df['tournament'].isin(major), 2,
                          np.where(qual, 1, 0))

df['neutral_venue'] = df['neutral'].astype(int)

print("Features built.\n")


print("="*60)
print("STEP 3: Training and evaluating the 3 models")
print("="*60)

from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb
from scipy.stats import poisson as poisson_dist
import json

SPLIT_DATE = '2019-01-01'
train = df[df['date'] < SPLIT_DATE].copy()
test = df[df['date'] >= SPLIT_DATE].copy()
print(f"Train: {len(train)} matches, Test: {len(test)} matches")

label_map = {'H': 0, 'D': 1, 'A': 2}
train['y'] = train['result'].map(label_map)
test['y'] = test['result'].map(label_map)

results = {}

def brier_multiclass(y_true, probs, n_classes=3):
    onehot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((probs - onehot) ** 2, axis=1))

# --- Model 1: Elo (logistic regression) ---
X_train_elo = train[['elo_diff', 'neutral_venue']].values
X_test_elo = test[['elo_diff', 'neutral_venue']].values

elo_clf = LogisticRegression(max_iter=1000)
elo_clf.fit(X_train_elo, train['y'])
probs_elo = elo_clf.predict_proba(X_test_elo)
preds_elo = elo_clf.predict(X_test_elo)

results['Elo (logistic)'] = {
    'accuracy': accuracy_score(test['y'], preds_elo),
    'log_loss': log_loss(test['y'], probs_elo, labels=[0,1,2]),
    'brier': brier_multiclass(test['y'].values, probs_elo)
}

# --- Model 2: Poisson goals model ---
X_train_home = train[['home_elo_pre','away_elo_pre','home_form','neutral_venue']].values
X_train_away = train[['away_elo_pre','home_elo_pre','away_form','neutral_venue']].values
X_test_home = test[['home_elo_pre','away_elo_pre','home_form','neutral_venue']].values
X_test_away = test[['away_elo_pre','home_elo_pre','away_form','neutral_venue']].values

pois_home = PoissonRegressor(max_iter=500, alpha=0.1)
pois_away = PoissonRegressor(max_iter=500, alpha=0.1)
pois_home.fit(X_train_home, train['home_score'])
pois_away.fit(X_train_away, train['away_score'])

lambda_home = pois_home.predict(X_test_home)
lambda_away = pois_away.predict(X_test_away)

def match_outcome_probs(lam_h, lam_a, max_goals=10):
    probs = np.zeros(3)
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson_dist.pmf(i, lam_h) * poisson_dist.pmf(j, lam_a)
            if i > j:
                probs[0] += p
            elif i == j:
                probs[1] += p
            else:
                probs[2] += p
    return probs / probs.sum()

probs_poisson = np.array([match_outcome_probs(lh, la) for lh, la in zip(lambda_home, lambda_away)])
preds_poisson = probs_poisson.argmax(axis=1)

results['Poisson (goals model)'] = {
    'accuracy': accuracy_score(test['y'], preds_poisson),
    'log_loss': log_loss(test['y'], probs_poisson, labels=[0,1,2]),
    'brier': brier_multiclass(test['y'].values, probs_poisson)
}

# --- Model 3: XGBoost ---
feature_cols_xgb = ['elo_diff', 'form_diff', 'rest_diff', 'tournament_tier',
                     'neutral_venue', 'home_elo_pre', 'away_elo_pre',
                     'home_form', 'away_form']

X_train_xgb = train[feature_cols_xgb].values
X_test_xgb = test[feature_cols_xgb].values

xgb_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=3,
    eval_metric='mlogloss', random_state=42
)
xgb_clf.fit(X_train_xgb, train['y'])
probs_xgb = xgb_clf.predict_proba(X_test_xgb)
preds_xgb = xgb_clf.predict(X_test_xgb)

results['XGBoost'] = {
    'accuracy': accuracy_score(test['y'], preds_xgb),
    'log_loss': log_loss(test['y'], probs_xgb, labels=[0,1,2]),
    'brier': brier_multiclass(test['y'].values, probs_xgb)
}

# --- Baseline: always predict majority class ---
majority_class = train['y'].mode()[0]
maj_probs = np.tile(train['y'].value_counts(normalize=True).sort_index().values, (len(test), 1))
results['Majority-class baseline'] = {
    'accuracy': accuracy_score(test['y'], np.full(len(test), majority_class)),
    'log_loss': log_loss(test['y'], maj_probs, labels=[0,1,2]),
    'brier': brier_multiclass(test['y'].values, maj_probs)
}

print("\n" + "="*65)
print(f"{'Model':<25}{'Accuracy':>12}{'Log-Loss':>14}{'Brier':>14}")
print("="*65)
for name, m in results.items():
    print(f"{name:<25}{m['accuracy']:>12.4f}{m['log_loss']:>14.4f}{m['brier']:>14.4f}")

importances = dict(zip(feature_cols_xgb, [float(x) for x in xgb_clf.feature_importances_]))
importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
print("\nXGBoost feature importances:")
for k, v in importances.items():
    print(f"  {k:<20} {v:.4f}")

with open('results_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
with open('feature_importance.json', 'w') as f:
    json.dump(importances, f, indent=2)

test_out = test[['date','home_team','away_team','home_score','away_score','result','tournament']].copy()
test_out['y_true'] = test['y'].values
test_out['pred_elo'] = preds_elo
test_out['pred_poisson'] = preds_poisson
test_out['pred_xgb'] = preds_xgb
test_out['prob_xgb_H'] = probs_xgb[:,0]
test_out['prob_xgb_D'] = probs_xgb[:,1]
test_out['prob_xgb_A'] = probs_xgb[:,2]
test_out.to_csv('test_predictions.csv', index=False)
print("\nSaved results_summary.json, feature_importance.json, test_predictions.csv\n")


print("="*60)
print("STEP 4: Error analysis + generating figures")
print("="*60)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# --- Figure 1: Model comparison bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
metrics = ['accuracy', 'log_loss', 'brier']
titles = ['Accuracy (higher better)', 'Log-Loss (lower better)', 'Brier Score (lower better)']
models = list(results.keys())
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

for ax, metric, title in zip(axes, metrics, titles):
    vals = [results[m][metric] for m in models]
    bars = ax.bar(models, vals, color=colors)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis='x', rotation=25)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('fig1_model_comparison.png', bbox_inches='tight')
plt.close()
print("Saved fig1_model_comparison.png")

# --- Figure 2: Confusion matrices ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
pred_cols = ['pred_elo', 'pred_poisson', 'pred_xgb']
titles2 = ['Elo (Logistic)', 'Poisson (Goals)', 'XGBoost']

for ax, col, title in zip(axes, pred_cols, titles2):
    cm = confusion_matrix(test_out['y_true'], test_out[col], labels=[0,1,2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=['H','D','A'], yticklabels=['H','D','A'], cbar=False)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('fig2_confusion_matrices.png', bbox_inches='tight')
plt.close()
print("Saved fig2_confusion_matrices.png")

# --- Figure 3: Feature importance ---
fig, ax = plt.subplots(figsize=(7, 5))
feats = list(importances.keys())
vals = list(importances.values())
ax.barh(feats[::-1], vals[::-1], color='#4C72B0')
ax.set_xlabel('XGBoost Feature Importance (gain-based)')
ax.set_title('Feature Importance for Match Outcome Prediction')
plt.tight_layout()
plt.savefig('fig3_feature_importance.png', bbox_inches='tight')
plt.close()
print("Saved fig3_feature_importance.png")

# --- Draw-prediction error analysis ---
test_out['correct_xgb'] = (test_out['pred_xgb'] == test_out['y_true']).astype(int)
draw_mask = test_out['y_true'] == 1
print(f"\nDraw prediction: XGBoost predicted draw correctly {test_out.loc[draw_mask,'correct_xgb'].mean():.1%} of the time")
print(f"XGBoost predicted 'Draw' as its top pick in {(test_out['pred_xgb']==1).mean():.1%} of all matches")
print(f"Actual draw rate in test set: {draw_mask.mean():.1%}")

print("\n" + "="*60)
print("ALL DONE. Check the folder for the .png figures and .json/.csv results.")
print("="*60)
