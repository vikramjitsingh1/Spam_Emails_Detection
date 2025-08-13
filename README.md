# 📧 Spam/Ham Email Detection with Naive Bayes + TF-IDF

This project is a **Spam vs Ham email classifier** built in **Python** with a **Tkinter GUI**.  
It uses **TF-IDF vectorization** with **Multinomial Naive Bayes** for text classification and supports **loading custom CSV datasets**.  
You can train a model, save it, reload it, and classify new messages in real-time.

---

## 🚀 Features
- **Naive Bayes Classifier** with `TfidfVectorizer` (unigrams + bigrams)
- **Custom CSV file support** (`Message`, `Category` format)
- **GUI built with Tkinter** — No coding required for end-users
- **Data Preprocessing**: Lowercasing, tokenization, stopword removal
- **Data Visualization**: Spam/Ham count bar chart
- **Model Saving/Loading**: Avoids retraining every time
- **Real-time message classification**

---

## 🛠 Tools & Libraries
- Python 3.x
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Joblib
- Tkinter (built-in with Python)

---

## 📂 Folder Structure
Spam-Detection-NaiveBayes/
├── maincode.py # Main GUI-based Python application
├── sample_dataset.csv # Example dataset (Message, Category)
├── spam_model.pkl # Saved trained model (after training)
├── vectorizer.pkl # Saved TF-IDF vectorizer (after training)
├── README.md # Project documentation
---

## 📦 Installation & Running:
### 1️⃣ Clone this repository

git clone https://github.com/your-username/spam-ham-detection.git
cd spam-ham-detection

### 2. Install dependencies and NLTK resources:

pip install pandas scikit-learn nltk matplotlib joblib

nltk.download('stopwords')
nltk.download('punkt')

### 3. Run the app:

python maincode.py

### 4. Load your CSV file and begin classification.

## CSV Format
Your dataset should have two columns:
### Message	Category
“You’ve won a free iPhone!”	              spam

“Let’s meet tomorrow.”	                   ham

## Visualization Features

* Bar Chart: Count of spam vs ham messages in dataset
* Accuracy Report: Accuracy score, confusion matrix, classification report
* Real-time Classification: Enter a message, get spam/ham result instantly

## Use Case

* Learning Text Classification with Naive Bayes
* Academic NLP projects
* Building real-time spam filters
* Understanding TF-IDF vectorization

## Credits

**Developed by Vikramjit Singh**
Based on Naive Bayes classification concepts and improved with TF-IDF + GUI integration.

