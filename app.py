import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

#loading saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

#setting up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www.\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)

#streamlit UI
st.title('🕵️NewsInspector')

st.markdown('Welcome to **NewInspector**, a machine learning-powered application designed to assess the credibility of news articles \nThis tool analyzes input text and classifies it as either **Fake** or **Real**, based on the patterns learned from a labeled dataset of news content. \nplease paste a news article below to begin the analysis.')

st.caption('Developed using streamlit, Random Forest Classifier, and TF-IDF Vectorization')


user_input = st.text_area('Paste the news article here:')

if st.button('Predict'):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]

    if result == 1:
        st.success('This News article is **REAL**.')
    else:
        st.error("This News article is **FAKE**.")












