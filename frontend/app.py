import streamlit as st
import requests
import os

# 1. Page Configuration
st.set_page_config(page_title="News Article Categorizer", layout="wide")

st.title("📰 News Article Categorizer")

st.markdown("""
News articles span a vast spectrum — from global economics and politics to sports and entertainment. 
Manually sorting this volume of data is a bottleneck; however, **automated classification** streamlines the process for journalists, readers, and aggregators alike.
""")

# Using columns to make the benefits more scannable
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

# Creating a clean split for the two methods
tech_col1, tech_col2 = st.columns(2)

with tech_col1:
    st.markdown("### Bag of Words (BoW)")
    st.caption("Counts the frequency of every word in a document, creating a simple 'bag' of counts.")

with tech_col2:
    st.markdown("### TF-IDF")
    st.caption("Weights words by their uniqueness, highlighting the most descriptive terms in an article.")

# 2. Setup Backend URL
# If running locally without Docker, this defaults to http://localhost:8000
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
# 3. User Interface
st.divider()
user_input = st.text_area("Paste a news article or headline here:", height=100, 
                          placeholder="Example: The tech giant released a new AI model today...")

if st.button("Submit", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing with both models..."):
            try:
                # Send request to FastAPI
                response = requests.post(f"{BACKEND_URL}/predict", json={"text": user_input})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Create two columns for visual comparison
                    col1, col2 = st.columns(2)
                    
                    # Column 1: Bag of Words Results
                    # Matches keys from your main.py: "bow_prediction"
                    with col1:
                        st.subheader("📊 Bag of Words (BoW) Model")
                        prediction_bow = data.get('bow_prediction', 'Unknown')
                        st.metric(label="Predicted Category", value=prediction_bow.upper())
                        st.info("Simple frequency-based classification.")
                        
                    # Column 2: TF-IDF Results
                    # Matches keys from your main.py: "tfidf_prediction"
                    with col2:
                        st.subheader("📈 TF-IDF Model")
                        prediction_tfidf = data.get('tfidf_prediction', 'Unknown')
                        st.metric(label="Predicted Category", value=prediction_tfidf.upper())
                        st.info("Importance-weighted classification.")

                    # Highlight differences
                    st.divider()
                    if prediction_bow != prediction_tfidf:
                        st.warning("⚠️ **Note:** The models disagree on this article!")
                    else:
                        st.success("✅ **Result:** Both models agreed on the same category.")

                else:
                    st.error(f"Backend Error: {response.status_code}")
            except Exception as e:
                st.error(f"Could not connect to the backend. Ensure 'uvicorn backend.main:app' is running.")
                st.debug(f"Error details: {e}")