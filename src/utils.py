import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download resources if not present
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    Cleans text: lowercase, removes punctuation/numbers and stopwords.
    """
    stop_words = set(stopwords.words('english'))
    
    # Lowercase and remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    
    # Remove stopwords
    cleaned = [w for w in tokens if w not in stop_words]
    return " ".join(cleaned)