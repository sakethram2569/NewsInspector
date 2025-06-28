import pandas as pd
import streamlit as st
import pickle
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

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

st.markdown("""Welcome to **NewsInspector**, an intelligent, feedback-driven application designed to assess the credibility of news articles.\nThis tool leverages machine learning to classify news content as either **Fake** or **Real**, based on patterns learned from a labeled dataset and ongoing user feedback.
\nPaste a news article below to begin the analysis and help improve the system through your feedback.""")

st.caption('Built with Streamlit, Random Forest Classifier, TF-IDF Vectorization, and a live feedback loop for adaptive learning.')

# session state setup
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
    
#taking input from user
user_input = st.text_area('Paste the 📰 news article here:', value = st.session_state.user_input)

#Prediction Logic
if st.button('🔍 Predict'):
    if not user_input.strip():
        st.session_state.predicted = False
        st.error('⚠️ Please paste a news article before clicking predict.')
    else:
        st.session_state.user_input = user_input
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)[0]
        st.session_state.predicted = True
        st.session_state.prediction_result = int(result)

#showing prediction results
if st.session_state.predicted:
    result = st.session_state.prediction_result
    if result == 1:
        st.success('✅ This News article is **REAL**.')
    else:
        st.error("❌ This News article is **FAKE**.")

    # Feedback System

    #1 Collecting Feedback from user
    st.markdown('#### 📣 Was this prediction correct?')
    feedback = st.radio('Your Feedback: ', ["Yes", 'No'], key = 'feedback_radio')

    #2 Saving wrong predictions
    if st.button('✅ Submit Feedback'):
        if feedback == 'No':
            st.warning("📝We'll learn from this incorrect prediction! ")

            #correcting the class as it is false prediction
            corrected_class = 1 - result

            #creating a row with the user input text and corrected class
            feedback_row = pd.DataFrame({
                'text': [st.session_state.user_input],
                'class': [corrected_class]
            })

            #Appending failed feedback to 'failed_predictions.csv'
            feedback_file = 'failed_predictions.csv'
            if not os.path.exists(feedback_file):
                pd.DataFrame(columns = ['text', 'class']).to_csv(feedback_file, index = False)
                
            existing = pd.read_csv(feedback_file)
            updated = pd.concat([existing, feedback_row], ignore_index = True)
            
            #save the updated file
            updated.to_csv(feedback_file, index = False)
            st.success("📥 Feedback saved successfully for future model improvement!")

            # 📊 Feedback count tracker
            remaining = 10 - len(updated)
            if remaining > 0:
                st.info(f"📊 {len(updated)} out of 10 failed prediction feedbacks collected. Retraining will start after {remaining} more.")
                
            # Retraining the model if 10 or more feedbacks collected
            if len(updated) >= 10:
                st.info("🔁 Retraining model with 10 new feedback corrections...")
                
                #loading the original data
                original_data = pd.read_csv("original_data.csv")

                # copying the 10 feedbacks for retraining
                feedback_batch = updated.iloc[:10].copy()

                #Merge original + feedback
                updated_data = pd.concat([original_data, feedback_batch], ignore_index = True)

                # full sanitation
                updated_data = updated_data.dropna(subset = ['text'])
                updated_data = updated_data[updated_data['text'].apply(lambda x: isinstance(x, str))]
                updated_data = updated_data[updated_data['text'].str.strip().str.lower() != 'nan']
                updated_data = updated_data[updated_data['text'].str.strip() != '']
                
                updated_data.to_csv('original_data.csv', index = False)

                #Retrain the model from scratch
                vectorizer = TfidfVectorizer()
                x = vectorizer.fit_transform(updated_data['text'])
                y = updated_data['class']

                model = RandomForestClassifier() 
                model.fit(x,y)
                #here the reason behind creating a new vectorizer and model everytime we retrain without fitting the existing model is to make sure that the model will not crash after fitting it repeatedly, and avoiding repeted features extractions

                # the new model is not reflected immediately unless the app is rerun, to apply the updated model immediately, you need to reload it after saving or.
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                
                #saving updated model and vectorizer
                pickle.dump(model, open('model.pkl', 'wb'))
                pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

                #removing the used feedback (the 10 rows)
                remaining_feedback = updated.iloc[10:]
                remaining_feedback.to_csv(feedback_file, index = False)

                st.success('✅ Model succesfully retrained using the 10 collected feedbacks. Thank you for helping improve the system!')
            
        else:
            st.success("👍Thanks for confirming the model's prediction.")

st.markdown("---")
st.caption('Developed by **Devanaboyina Saketh Ram**')
st.caption(
    """ 
    B.Tech, IIT Kharagpur  
    🖂 [sakethram2569@gmail.com](mailto:sakethram2569@gmail.com)  
    🔗 [View GitHub Profile](https://github.com/sakethram2569)
    """
)
 







