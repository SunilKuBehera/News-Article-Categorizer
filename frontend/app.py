import streamlit as st
import requests
import os
import sys
import pandas as pd

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# --- 1. Page Configuration ---
st.set_page_config(page_title="News Article Categorizer", layout="wide")
st.title("📰 News Article Categorizer")

st.markdown("""
News articles span a vast spectrum — from global economics and politics to sports and entertainment.  
Manually sorting this volume of data is a bottleneck; however, **automated classification** streamlines the process for journalists, readers, and aggregators alike.
""")

# --- Benefits Section ---
col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Media Monitoring** \nQuickly track news on specific topics.")
with col2:
    st.info("**Recommendations** \nSuggest articles based on user interests.")
with col3:
    st.info("**Sentiment Analysis** \nGauge public reaction to events.")

st.write("""
This is achieved using **Natural Language Processing (NLP)** to convert text into numerical vectors.  
Two primary techniques drive this transformation:
""")

tech_col1, tech_col2 = st.columns(2)
with tech_col1:
    st.markdown("### Bag of Words (BoW)")
    st.caption("Counts the frequency of every word in a document, creating a simple 'bag' of counts.")
with tech_col2:
    st.markdown("### TF-IDF")
    st.caption("Weights words by their uniqueness, highlighting the most descriptive terms in an article.")

# --- 2. Backend Setup ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# --- 3. User Input ---
st.divider()
user_input = st.text_area(
    "Paste a news article or headline here:",
    height=100,
    placeholder="Example: The tech giant released a new AI model today..."
)

if st.button("Submit", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing with both models..."):
            try:
                response = requests.post(f"{BACKEND_URL}/predict", json={"text": user_input})
                if response.status_code == 200:
                    data = response.json()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📊 Bag of Words (BoW) Model")
                        prediction_bow = data.get('bow_prediction', 'Unknown')
                        st.metric(label="Predicted Category", value=prediction_bow.upper())
                        st.info("Simple frequency-based classification.")

                    with col2:
                        st.subheader("📈 TF-IDF Model")
                        prediction_tfidf = data.get('tfidf_prediction', 'Unknown')
                        st.metric(label="Predicted Category", value=prediction_tfidf.upper())
                        st.info("Importance-weighted classification.")

                    st.divider()
                    if prediction_bow != prediction_tfidf:
                        st.warning("⚠️ **Note:** The models disagree on this article!")
                    else:
                        st.success("✅ **Result:** Both models agreed on the same category.")
                else:
                    st.error(f"Backend Error: {response.status_code}")
            except Exception as e:
                st.error("Could not connect to the backend. Ensure 'uvicorn backend.main:app' is running.")
                st.debug(f"Error details: {e}")

# --- 4. Model Performance Insights ---
st.divider()
st.subheader("📊 Model Performance Insights")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_dir = os.path.join(BASE_DIR, "models")

# --- Accuracy Metrics ---
metrics_path = os.path.join(models_dir, "metrics.txt")
if os.path.exists(metrics_path):
    st.markdown("### Accuracy Scores")
    with open(metrics_path, "r") as f:
        metrics = f.readlines()
        for line in metrics:
            if ":" in line:
                label, value = line.split(":")
                st.metric(label=label.strip(), value=f"{float(value.strip())*100:.2f}%")
    st.caption("Results based on 20% test split of the BBC News dataset.")
else:
    st.warning("Metrics file not found. Run training script to generate accuracy scores.")

# --- Confusion Matrices ---
st.markdown("### Confusion Matrices")
cm_bow_path = os.path.join(models_dir, "confusion_matrix_bow.png")
cm_tfidf_path = os.path.join(models_dir, "confusion_matrix_tfidf.png")

col1, col2 = st.columns(2)
with col1:
    if os.path.exists(cm_bow_path):
        st.image(cm_bow_path, caption="Confusion Matrix (BoW Model)", use_container_width=True)
    else:
        st.info("BoW confusion matrix not found.")
with col2:
    if os.path.exists(cm_tfidf_path):
        st.image(cm_tfidf_path, caption="Confusion Matrix (TF-IDF Model)", use_container_width=True)
    else:
        st.info("TF-IDF confusion matrix not found.")

# --- ROC Curves ---
st.markdown("### ROC Curves")
roc_bow_path = os.path.join(models_dir, "roc_curve_bow.png")
roc_tfidf_path = os.path.join(models_dir, "roc_curve_tf-idf.png")

col1, col2 = st.columns(2)
with col1:
    if os.path.exists(roc_bow_path):
        st.image(roc_bow_path, caption="ROC Curve (BoW Model)", use_container_width=True)
    else:
        st.info("BoW ROC curve not found.")
with col2:
    if os.path.exists(roc_tfidf_path):
        st.image(roc_tfidf_path, caption="ROC Curve (TF-IDF Model)", use_container_width=True)
    else:
        st.info("TF-IDF ROC curve not found.")

# --- Precision-Recall Curves ---
st.markdown("### Precision-Recall Curves")
pr_bow_path = os.path.join(models_dir, "pr_curve_bow.png")
pr_tfidf_path = os.path.join(models_dir, "pr_curve_tf-idf.png")

col1, col2 = st.columns(2)
with col1:
    if os.path.exists(pr_bow_path):
        st.image(pr_bow_path, caption="Precision-Recall Curve (BoW Model)", use_container_width=True)
    else:
        st.info("BoW Precision-Recall curve not found.")
with col2:
    if os.path.exists(pr_tfidf_path):
        st.image(pr_tfidf_path, caption="Precision-Recall Curve (TF-IDF Model)", use_container_width=True)
    else:
        st.info("TF-IDF Precision-Recall curve not found.")

# --- Classification Reports ---
st.markdown("### Classification Reports")
report_path = os.path.join(models_dir, "report.txt")

def parse_classification_report(report_text):
    """Convert sklearn classification_report text into a DataFrame."""
    lines = [line.strip() for line in report_text.splitlines() if line.strip()]
    data = []
    section = None
    for line in lines:
        if line.startswith("==="):
            section = line.strip("=").strip()
            continue
        parts = line.split()
        if len(parts) >= 5 and parts[0].isalpha() and parts[0].lower() not in ["avg", "macro", "weighted", "accuracy"]:
            try:
                label = parts[0]
                precision, recall, f1, support = parts[1:5]
                data.append([section, label, float(precision), float(recall), float(f1), int(support)])
            except ValueError:
                continue  # Skip rows that can't be parsed
    return pd.DataFrame(data, columns=["Model", "Class", "Precision", "Recall", "F1-Score", "Support"])

if os.path.exists(report_path):
    with open(report_path, "r") as f:
        report_text = f.read()
    df_report = parse_classification_report(report_text)
    if not df_report.empty:
        for model in df_report["Model"].unique():
            st.markdown(f"#### {model}")
            st.dataframe(df_report[df_report["Model"] == model].drop(columns=["Model"]), use_container_width=True)
    else:
        st.text(report_text)
else:
    st.info("Classification report not found. Run training script to generate reports.")