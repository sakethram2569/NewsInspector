# 📰 NewsInspector - Fake News Detection App (ML + Streamlit)

**NewsInspector** is a machine learning-powered web application that classifies news articles as **Real** or **Fake**. Built with Natural Language Processing (NLP) and a Random Forest model, it enables real-time detection through an interactive **Streamlit** interface.

---

## 🔍 Features

- 🧠 **Machine Learning**: Utilizes TF-IDF vectorization and a Random Forest Classifier  
- 🎯 **High Accuracy**: Achieves **~99.6% test accuracy** on a Kaggle dataset  
- 🧼 **Robust Preprocessing**: Handles noise with lowercasing, lemmatization, punctuation and stopword removal  
- 💡 **Interactive Web Interface**: Streamlit-powered real-time input and predictions  
- 🧪 **Edge Case Handling**: Tested with real-world articles to ensure robustness  

---

## 📁 Project Structure

```
NewsInspector/
│
├── data/             # Dataset files (e.g., train.csv, test.csv)
├── models/           # Trained model pickle files
├── notebooks/        # Jupyter notebooks for EDA and model development
├── app.py            # Streamlit app code
├── utils.py          # Preprocessing and helper functions
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
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
- **Evaluation**: Confusion matrix, classification report, accuracy, precision, recall  

---

## 📊 Dataset

- **Source**: [Kaggle - Fake News Detection Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)  
- Includes real and fake news labeled articles for supervised learning.

---


## 📌 Future Enhancements

- ✅ Add LSTM or BERT-based deep learning models  
- ✅ Store user inputs and feedback for improving predictions  
- ✅ Visualize feature importance and model confidence  
- ✅ Deploy using cloud platforms like Streamlit Cloud or AWS  

---

## 👨‍💻 Author

**Devanaboyina Saketh Ram**  
_IIT Kharagpur | Electrical Engineering | Aspiring ML Engineer_

