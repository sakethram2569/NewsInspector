# 📰 NewsInspector - Fake News Detection App (ML + Streamlit)

A self-improving fake news detection web app that classifies news articles as **Real** or **Fake** using **TF-IDF**, **Random Forest**, and a real-time **feedback-driven retraining system**.

---

## 🔍 Features

- 🧠 **Machine Learning Pipeline**: Utilizes **TF-IDF vectorization** and a **Random Forest Classifier**
- 🎯 **High Accuracy**: Achieves **~99.2% test accuracy** and **F1-score > 0.99** on Kaggle dataset
- 🧼 **Robust Preprocessing**: Includes lowercasing, lemmatization, punctuation & stopword removal
- 💡 **Interactive Web Interface**: Real-time prediction powered by **Streamlit**
- ♻️ **Feedback-Based Retraining**: Collects failed predictions and auto-retrains after every 10
- 📊 **Real-World Tested**: Evaluated on real articles for edge case handling
- 🔐 **Input Sanitization**: Cleans user input and prevents noisy/unusable text
- ⚡️ **Fast Predictions**: Live results with average response time under **2 seconds**

---

## 📁 Project Structure

```
NewsInspector/
├── app.py                   # Streamlit application
├── model.pkl                # Trained Random Forest model
├── vectorizer.pkl           # TF-IDF vectorizer
├── original_data.csv        # Main dataset with feedback included
├── failed_predictions.csv   # Stores failed feedbacks for retraining
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```


---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/NewsInspector.git
   cd NewsInspector

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

3. **Run the App**
   ```bash
   streamlit run app.py

---

## 🧠 Model Details

- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)  
- **Classifier**: Random Forest with optimized hyperparameters
- **Retraining Logic**: After 10 failed predictions via user feedback 
- **Evaluation**: Confusion matrix, classification report, accuracy, precision, recall  

---

## 📊 Dataset

- **Source**: [Kaggle - Fake News Detection Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)  
- Includes real and fake news labeled articles for supervised learning.

---


## 📌 Future Enhancements

- 🔬  Add LSTM or BERT-based deep learning models  
- 💬 Save prediction history per user session  
- ✅ Visualize feature importance and model confidence  
- 📈 Add interactive charts and confidence levels

---

## 👨‍💻 Author

**Devanaboyina Saketh Ram**  
_IIT Kharagpur | Electrical Engineering | Aspiring ML Engineer_
📧 [sakethram2569@gmail.com](mailto:sakethram2569@gmail.com)  
🔗 [View GitHub Profile](https://github.com/sakethram2569)
