# 📩 Spam SMS Classifier

A web application that detects whether a message is **Spam** or **Not Spam** using a Voting Classifier trained on SMS data. This project combines **Natural Language Processing (NLP)**, **TF-IDF vectorization**, and **Ensemble Machine Learning**, all wrapped in a user-friendly **Streamlit** interface.

---

## 🚀 Features

- 🔤 **Text Preprocessing** (lowercasing, tokenization, stopword & punctuation removal, stemming)
- ✨ **TF-IDF Vectorization** for feature extraction
- 🧠 **Voting Classifier** using:
  - SVM (`sigmoid` kernel)
  - Multinomial Naive Bayes
  - Extra Trees Classifier
- 🎯 Prediction Confidence (for probabilistic models)
- 📈 **Live Word Frequency Visualization** from user input
- 🧾 Sidebar with project info
- 🖥️ Built with **Streamlit**

---

## 🧠 Model Training Overview

- Dataset: SMS Spam Collection (UCI Machine Learning Repository)
- Labels: `ham` (not spam) and `spam`
- Preprocessing:
  - Tokenization
  - Removal of punctuation and stopwords
  - Stemming using PorterStemmer
- Feature Extraction: TF-IDF
- Classifier: Soft Voting Ensemble with SVM, MultinomialNB, and ExtraTreesClassifier
- Pickled Model + Vectorizer for deployment
- Working video link: [Streamlit working video]https://jklujaipur-my.sharepoint.com/:v:/g/personal/dishaarora_jklu_edu_in/ETRhV_aSTxZEsoBYNYmr4ZEBNQDeRXg-2fV8CL-6iWRogg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=gcdVPq

---

## 🖥️ How to Run the App

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/spam-sms-classifier.git
   cd spam-sms-classifier
   ```
2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   
---

## 📊 Sample Output

✅ "This is to inform you about tomorrow’s class timing." → Not Spam

🚨 "You won a free iPhone! Click here to claim now." → Spam

Includes:
- Word frequency bar chart
- Prediction confidence meter
- Spam/Not Spam label

--- 

## 🛠️ Tools & Libraries

- Python
- scikit-learn
- NLTK
- Streamlit
- Pandas
- Matplotlib
- Seaborn

---

## 📌 Future Improvements

- Add support for multiple language detection
- Deploy to cloud (e.g., Streamlit Cloud or HuggingFace Spaces)
- Extend dataset with more recent spam examples
- Add explainability (SHAP/LIME visualizations)

---
