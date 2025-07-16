import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

# Ensure required NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

# Custom page style
st.markdown("""
    <style>
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        font-size: 40px;
        color: #262730;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.title("📘 About")
st.sidebar.markdown("""
This is a **Spam Detection Web App** built with:
- ✅ Natural Language Processing (NLP)
- ✅ TF-IDF Vectorizer
- ✅ Voting Ensemble ML Model (Soft Voting)
- ✅ Word Frequency Visualization
- ✅ Streamlit for UI/UX

""")

# Initialize stemmer
ps = PorterStemmer()

# Text Preprocessing Function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Input Area
input_sms = st.text_area("📝 Enter the message to classify", help="Paste any email or SMS message here")

if st.button('🔍 Predict') and input_sms.strip() != "":
    # Preprocess
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Confidence (if available)
    try:
        proba = model.predict_proba(vector_input)[0]
        confidence = proba[1] if result == 1 else proba[0]
    except:
        confidence = None

    # Layout using tabs
    tab1, tab2, tab3 = st.tabs(["📊 Result", "📈 Word Frequency", "☁️ Word Cloud"])

    with tab1:
        if result == 1:
            st.markdown("## 🚨 <span style='color:red'>Spam</span>", unsafe_allow_html=True)
        else:
            st.markdown("## ✅ <span style='color:green'>Not Spam</span>", unsafe_allow_html=True)

        if confidence is not None:
            st.write(f"**Prediction Confidence:** {confidence*100:.2f}%")
            st.progress(confidence)

    with tab2:
        word_list = transformed_sms.split()
        if word_list:
            word_freq = Counter(word_list).most_common(30)
            word_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x="Word", y="Frequency", data=word_df, palette="magma", ax=ax)
            plt.xticks(rotation=45)
            plt.title("Top Frequent Words in Your Message")
            st.pyplot(fig)
        else:
            st.info("No valid words found in the message to display word frequency.")

    with tab3:
        if word_list:
            wc = WordCloud(width=600, height=300, background_color='white').generate(" ".join(word_list))
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Not enough words for a word cloud.")

# Footer
st.markdown("""
---
📚 [Streamlit](https://streamlit.io) | 🤖 ML Model | 🧠 NLP
""")
