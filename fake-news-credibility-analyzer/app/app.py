import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fake News Credibility Analyzer",
    layout="centered"
)

st.title("📰 Fake News Credibility Analyzer")
st.write("Paste a news article below to check its credibility with explanation.")

# -----------------------------
# Load NLP Tools (NO DOWNLOAD HERE)
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Load Model & Vectorizer (ABSOLUTE PATHS)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fake_news_model.pkl")
VECT_PATH  = os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECT_PATH, "rb") as f:
    tfidf = pickle.load(f)

# -----------------------------
# Feature Names & Coefficients (GLOBAL)
# -----------------------------
feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# Explain Prediction Function
# -----------------------------
def explain_prediction(cleaned_text, top_n=8):
    vector = tfidf.transform([cleaned_text])
    feature_index = vector.nonzero()[1]

    word_weights = []
    for idx in feature_index:
        word = feature_names[idx]
        weight = coefficients[idx]
        word_weights.append((word, weight))

    word_weights = sorted(word_weights, key=lambda x: abs(x[1]), reverse=True)
    return word_weights[:top_n]

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area(
    "Paste News Article Here",
    height=250,
    placeholder="Breaking news! This shocking revelation will change everything..."
)

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("Analyze Credibility"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0]

        # Result
        if prediction == 1:
            st.success(f"🟢 REAL NEWS (Confidence: {confidence[1]*100:.2f}%)")
        else:
            st.error(f"🔴 FAKE NEWS (Confidence: {confidence[0]*100:.2f}%)")

        # Explanation
        st.subheader("🔍 Why this prediction?")
        explanation = explain_prediction(cleaned)

        for word, weight in explanation:
            if weight > 0:
                st.write(f"🟢 **{word}** → pushes toward REAL ({weight:.2f})")
            else:
                st.write(f"🔴 **{word}** → pushes toward FAKE ({weight:.2f})")
