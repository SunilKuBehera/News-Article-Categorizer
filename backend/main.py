from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import sys
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# In Docker, /app is the root. This helps Python find the 'src' folder.
# We use the absolute path of the current file to stay flexible.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.utils import preprocess_text

app = FastAPI()

# Point directly to the models folder in the root
MODEL_DIR = os.path.join(root_dir, "models")
model_bow_path = os.path.join(MODEL_DIR, "news_classifier_bow.pkl")
model_tfidf_path = os.path.join(MODEL_DIR, "news_classifier_tfidf.pkl")

# Load models with basic error handling
try:
    model_bow = joblib.load(model_bow_path)
    model_tfidf = joblib.load(model_tfidf_path)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # You might want to initialize as None or exit if models are critical
    model_bow = None
    model_tfidf = None

class NewsInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "Backend is running", "models_loaded": model_bow is not None}

@app.post("/predict")
def predict(data: NewsInput):
    if model_bow is None or model_tfidf is None:
        return {"error": "Models not loaded on server"}
        
    clean_text = preprocess_text(data.text)
    
    res_bow = model_bow.predict([clean_text])[0]
    res_tfidf = model_tfidf.predict([clean_text])[0]
    
    return {
        "bow_prediction": str(res_bow),
        "tfidf_prediction": str(res_tfidf)
    }