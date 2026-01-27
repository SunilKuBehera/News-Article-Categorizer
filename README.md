# 🗞️ News Article Categorizer

An end-to-end Machine Learning application that automatically classifies news articles into five distinct categories: **Business, Entertainment, Politics, Sport, and Tech**. This project implements a full-stack architecture, comparing **Bag of Words (BoW)** and **TF-IDF** vectorization techniques using a **Multinomial Naive Bayes** classifier.

---

## 🚀 Project Overview

News articles span a vast spectrum of topics. Manually sorting this volume of data is a bottleneck; however, **automated classification** streamlines the process for journalists, readers, and content aggregators.

### Key Benefits

* **Media Monitoring:** Quickly track news on specific topics.
* **Content Recommendations:** Recommend articles based on users' interests.
* **Sentiment Analysis:** Determine public sentiment towards political events or companies.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Community Cloud)
* **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Hosted on Railway)
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **NLP:** NLTK (Tokenization, Stopword removal)
* **Containerization:** Docker & Docker Compose

---

## 🧪 Machine Learning Workflow

### 1. Data Preprocessing

Raw text is cleaned through a custom pipeline to ensure high model accuracy:

* **Tokenization:** Splitting text into individual words using `nltk.word_tokenize`.
* **Normalization:** Converting text to lowercase and removing non-alphabetic characters.
* **Stopword Removal:** Filtering out common English words (e.g., "the", "is") that lack predictive value.

### 2. Feature Extraction

The project compares two essential text representation techniques:

* **Bag of Words (BoW):** Converts text into a matrix of token counts.
* **TF-IDF:** Weights words by their uniqueness across the dataset, highlighting descriptive terms.

### 3. Model Performance

Both models achieved impressive accuracy on the BBC News dataset:

| Model      | Accuracy
|:--------------- |:------------ |
| **BoW**    | **96.63%** | 
| **TF-IDF** | **96.18%** |

---

## 📂 Project Structure

```text
NEWS-ARTICLE-CATEGORIZER/
├── backend/            # FastAPI Source Code
│   ├── main.py         # API Endpoints & Model Loading
│   └── Dockerfile    
├── frontend/           # Streamlit UI
│   ├── app.py        
│   └── Dockerfile    
├── data/               # Dataset (bbc_text.csv)
├── models/             # Trained .pkl files
├── src/                # Shared logic
│   ├── train.py        # Model Training Script
│   └── utils.py        # Preprocessing Functions
├── docker-compose.yml  # Local Orchestration
└── requirements.txt    # Project Dependencies
```
