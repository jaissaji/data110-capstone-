"""
============================================================
INTERMEDIATE PROJECT: Spam Email/SMS Detection
============================================================
Goal: Classify text messages as "spam" or "ham" (not spam),
      comparing THREE different classification models:
      Naive Bayes, Logistic Regression, and Support Vector Machine (SVM).

This is INTERMEDIATE level because it involves:
  - Text preprocessing (turning words into numeric features)
  - Comparing multiple models instead of just one
  - Multiple evaluation metrics (accuracy, precision, recall, F1)

HOW TO RUN:
    python spam_detection_intermediate.py

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn

Make sure "spam.csv" is in the same folder as this script.
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

sns.set_style('whitegrid')

# ------------------------------------------------------------
# STEP 1: Load the data
# ------------------------------------------------------------
print("="*60)
print("STEP 1: Loading the data")
print("="*60)

df = pd.read_csv('spam.csv')
print(f"Dataset shape: {df.shape}")
print(df['label'].value_counts())
print(f"\nSpam makes up {(df['label']=='spam').mean():.1%} of all messages")

# Convert labels to 0/1: ham=0 (not spam), spam=1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# ------------------------------------------------------------
# STEP 2: Preprocessing - turn text into numbers (TF-IDF)
# ------------------------------------------------------------
print("\n" + "="*60)
print("STEP 2: Converting text into numeric features (TF-IDF)")
print("="*60)
print("TF-IDF = Term Frequency-Inverse Document Frequency.")
print("It scores each word by how important it is to a message,")
print("giving less weight to very common words like 'the' or 'a'.\n")

X = df['message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Training messages: {X_train_vec.shape[0]}, Features (words): {X_train_vec.shape[1]}")

# ------------------------------------------------------------
# STEP 3: Train and compare 3 different models
# ------------------------------------------------------------
print("\n" + "="*60)
print("STEP 3: Training and comparing 3 models")
print("="*60)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM (Linear)': LinearSVC(max_iter=2000)
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    predictions[name] = y_pred

    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

# ------------------------------------------------------------
# STEP 4: Show results
# ------------------------------------------------------------
print("\n" + "="*70)
print(f"{'Model':<22}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1-score':>12}")
print("="*70)
for name, m in results.items():
    print(f"{name:<22}{m['accuracy']:>12.4f}{m['precision']:>12.4f}{m['recall']:>12.4f}{m['f1_score']:>12.4f}")

best_model = max(results, key=lambda k: results[k]['f1_score'])
print(f"\nBest model by F1-score: {best_model}")

# Bar chart comparing all models across all metrics
metrics_df = pd.DataFrame(results).T
metrics_df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title('Model Comparison: Spam Detection')
plt.ylabel('Score')
plt.xticks(rotation=15)
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('intermediate_fig1_model_comparison.png', bbox_inches='tight')
plt.close()
print("Saved intermediate_fig1_model_comparison.png")

# Confusion matrices for all 3 models
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], cbar=False)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('intermediate_fig2_confusion_matrices.png', bbox_inches='tight')
plt.close()
print("Saved intermediate_fig2_confusion_matrices.png")

print("\n" + "="*60)
print("ALL DONE. Check the folder for the .png figures.")
print("="*60)
