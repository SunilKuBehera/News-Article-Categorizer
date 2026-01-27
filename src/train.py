import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize

from src.utils import preprocess_text

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "bbc_text.csv")
models_dir = os.path.join(current_dir, "..", "models")

# --- LOAD DATA ---
print(f"Loading data from: {data_path}")
if not os.path.exists(data_path):
    print("❌ Error: File not found. Check your 'data' folder structure.")
    exit(1)

df = pd.read_csv(data_path)

# --- PREPROCESSING ---
print("Preprocessing text...")
df["clean_text"] = df["text"].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["category"],
    test_size=0.2,
    random_state=42
)

# --- BUILD PIPELINES ---
bow_pipeline = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

tfidf_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# --- TRAIN MODELS ---
print("Training models...")
bow_pipeline.fit(X_train, y_train)
tfidf_pipeline.fit(X_train, y_train)

# --- EVALUATION ---
print("\n--- Model Evaluation ---")
bow_acc = bow_pipeline.score(X_test, y_test)
tfidf_acc = tfidf_pipeline.score(X_test, y_test)

print(f"Bag of Words Accuracy: {bow_acc:.4f}")
print(f"TF-IDF Accuracy:      {tfidf_acc:.4f}")

categories = sorted(df["category"].unique())

# --- CONFUSION MATRICES ---
y_pred_bow = bow_pipeline.predict(X_test)
cm_bow = confusion_matrix(y_test, y_pred_bow)

plt.figure(figsize=(10, 7))
sns.heatmap(cm_bow, annot=True, fmt="d",
            xticklabels=categories, yticklabels=categories, cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: News Categorization (Bag of Words)")
os.makedirs(models_dir, exist_ok=True)
plt.savefig(os.path.join(models_dir, "confusion_matrix_bow.png"))
print(f"✅ BoW Confusion Matrix saved to {models_dir}")

y_pred_tfidf = tfidf_pipeline.predict(X_test)
cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)

plt.figure(figsize=(10, 7))
sns.heatmap(cm_tfidf, annot=True, fmt="d",
            xticklabels=categories, yticklabels=categories, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: News Categorization (TF-IDF)")
plt.savefig(os.path.join(models_dir, "confusion_matrix_tfidf.png"))
print(f"✅ TF-IDF Confusion Matrix saved to {models_dir}")

# --- CLASSIFICATION REPORTS ---
print("\nDetailed Classification Report (BoW):")
print(classification_report(y_test, y_pred_bow))

print("\nDetailed Classification Report (TF-IDF):")
print(classification_report(y_test, y_pred_tfidf))

# --- ROC & PRECISION-RECALL CURVES ---
print("\nGenerating ROC and Precision-Recall curves...")

y_test_bin = label_binarize(y_test, classes=categories)
n_classes = len(categories)

y_score_bow = bow_pipeline.predict_proba(X_test)
y_score_tfidf = tfidf_pipeline.predict_proba(X_test)

# ROC curves
for model_name, y_score in [("BoW", y_score_bow), ("TF-IDF", y_score_tfidf)]:
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{categories[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({model_name})")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(models_dir, f"roc_curve_{model_name.lower()}.png"))
    print(f"✅ ROC Curve ({model_name}) saved to {models_dir}")

# Precision-Recall curves
for model_name, y_score in [("BoW", y_score_bow), ("TF-IDF", y_score_tfidf)]:
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f"{categories[i]}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({model_name})")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(models_dir, f"pr_curve_{model_name.lower()}.png"))
    print(f"✅ Precision-Recall Curve ({model_name}) saved to {models_dir}")

# --- SAVE MODELS ---
print(f"Saving models to: {models_dir}")
joblib.dump(bow_pipeline, os.path.join(models_dir, "news_classifier_bow.pkl"))
joblib.dump(tfidf_pipeline, os.path.join(models_dir, "news_classifier_tfidf.pkl"))

# --- SAVE METRICS ---
metrics_file = os.path.join(models_dir, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"BoW Accuracy: {bow_acc:.4f}\n")
    f.write(f"TF-IDF Accuracy: {tfidf_acc:.4f}\n")
print(f"✅ Metrics saved to {metrics_file}")

# --- SAVE METRICS ---
metrics_file = os.path.join(models_dir, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"BoW Accuracy: {bow_acc:.4f}\n")
    f.write(f"TF-IDF Accuracy: {tfidf_acc:.4f}\n")
print(f"✅ Metrics saved to {metrics_file}")

# --- SAVE CLASSIFICATION REPORTS ---
report_file = os.path.join(models_dir, "report.txt")
with open(report_file, "w") as f:
    f.write("=== Classification Report (BoW) ===\n")
    f.write(classification_report(y_test, y_pred_bow))
    f.write("\n\n=== Classification Report (TF-IDF) ===\n")
    f.write(classification_report(y_test, y_pred_tfidf))
print(f"✅ Classification reports saved to {report_file}")

print("✅ Success! Models and evaluation artifacts saved.")