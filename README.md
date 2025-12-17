# 📰 Fake News Credibility Analyzer

An end-to-end **Explainable NLP-based Machine Learning application** that classifies news articles as **Real** or **Fake**, while also providing **word-level explanations** for each prediction.

This project focuses on **trustworthy and interpretable AI**, combining text classification, model explainability, and a deployed web interface.

---

## 🚀 Features

- 🔍 Fake vs Real news classification
- 🧠 NLP-based text preprocessing
- 📊 TF-IDF feature extraction
- 🤖 Logistic Regression model
- 📈 Model evaluation & error analysis
- 🧩 Explainable AI (word-level contribution)
- 🌐 Interactive Streamlit web application

---

## 🧠 Why This Project?

Fake news spreads rapidly and can significantly influence public opinion.  
This project not only predicts whether a news article is fake or real, but also **explains why**, making the system transparent, interpretable, and trustworthy.

---

## 🗂️ Project Structure
```
fake-news-credibility-analyzer/
│
├── app/
│ └── app.py # Streamlit web application
│
├── models/
│ ├── fake_news_model.pkl # Trained ML model
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_text_preprocessing.ipynb
│ ├── 03_model_training.ipynb
│ ├── 04_model_evaluation.ipynb
│ └── 05_model_explainability.ipynb
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE

```

---

## 📊 Dataset

**Source:** Kaggle – Fake and Real News Dataset  
🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### Dataset Files
- `Fake.csv` – Fake news articles  
- `True.csv` – Real news articles (Reuters)

### Columns
- `title`
- `text`
- `subject`
- `date`

### Labels
- `0` → Fake News  
- `1` → Real News  

⚠️ **Note:**  
Due to GitHub file size limits, raw and processed datasets are **not included** in this repository.

---

## 📦 Dataset Setup (Required)

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

2. Place the files locally as:
    data/raw/Fake.csv
    data/raw/True.csv

3. Run the notebooks in order to regenerate processed datasets.

---

## 🔄 Workflow

1. **Data Exploration**
   - Dataset inspection
   - Class distribution analysis
   - Text length analysis

2. **Text Preprocessing**
   - Lowercasing
   - URL & punctuation removal
   - Stopword removal
   - Lemmatization

3. **Feature Engineering**
   - TF-IDF Vectorization
   - Unigrams & bigrams

4. **Model Training**
   - Logistic Regression
   - Stratified train-test split

5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - Error analysis

6. **Explainable AI**
   - Feature weight analysis
   - Word-level contribution to predictions

7. **Deployment**
   - Streamlit web application

---

## 📈 Model Performance

- **Accuracy:** ~94–96%
- Balanced performance across Fake and Real classes
- Strong generalization on unseen articles

---

## 🔍 Explainability Example

- Words such as **breaking**, **shocking**, **revelation** push predictions toward **Fake**
- Words such as **reuters**, **official**, **statement** push predictions toward **Real**

This avoids black-box predictions and improves user trust.

---

## 🌐 Streamlit Application

**Input:** Paste a news article  
**Output:**
- Credibility label (Real / Fake)
- Confidence score
- Word-level explanation

---

## ▶️ How to Run Locally

### 1️⃣ Clone Repository
```bash
git clone https://github.com/tirthch25/fake-news-credibility-analyzer.git
cd fake-news-credibility-analyzer
```

### 2️⃣ Install Dependencies
```
python -m pip install -r requirements.txt
```
### 3️⃣ Download NLTK Resources (Once)
```
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```
### 4️⃣ Run Streamlit App
```
python -m streamlit run app/app.py
```

### 5️⃣Open in browser:
```
http://localhost:8501
```
---
## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Streamlit
- TF-IDF
- Logistic Regression
---

### 📌 Future Enhancements

- BERT-based text classification
- SHAP-based explainability
- Multilingual fake news detection
- Cloud deployment (Streamlit Cloud / Hugging Face Spaces)
---
### 👨‍💻 Author

- Tirth Chankeshwara
- Engineering Student | Data Analyst | AI/ML Enthusiast
