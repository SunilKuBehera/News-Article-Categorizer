import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from src.utils import preprocess_text

# --- PATH CONFIGURATION ---
# Get the directory where THIS script is currently located (src/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the script location
# Go up one level (..), then into 'data'
data_path = os.path.join(current_dir, "..", "data", "bbc_text.csv")
# Go up one level (..), then into 'models'
models_dir = os.path.join(current_dir, "..", "models")

# --- LOAD DATA ---
print(f"Loading data from: {data_path}")
if not os.path.exists(data_path):
    print("❌ Error: File not found. Check your 'data' folder structure.")
    exit(1)

df = pd.read_csv(data_path)

# --- PREPROCESSING ---
print("Preprocessing text...")
df['clean_text'] = df['text'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], 
    df['category'], 
    test_size=0.2, 
    random_state=42
)

# --- BUILD PIPELINES ---
bow_pipe = Pipeline([('v', CountVectorizer()), ('c', MultinomialNB())])
tfidf_pipe = Pipeline([('v', TfidfVectorizer()), ('c', MultinomialNB())])

# --- TRAIN ---
print("Training models...")
bow_pipe.fit(X_train, y_train)
tfidf_pipe.fit(X_train, y_train)

# --- SAVE MODELS ---
# Create the models directory if it doesn't exist yet
os.makedirs(models_dir, exist_ok=True)

print(f"Saving models to: {models_dir}")
joblib.dump(bow_pipe, os.path.join(models_dir, "news_classifier_bow.pkl"))
joblib.dump(tfidf_pipe, os.path.join(models_dir, "news_classifier_tfidf.pkl"))

print("✅ Success! Models saved.")