import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

# Download NLTK data only if missing
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")

# Global variables
data = None
vectorizer = None
model = None
X_train = X_test = y_train = y_test = None

# Text preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

# Load dataset
def load_dataset():
    global data
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        data = pd.read_csv(file_path)

        # Clean column names
        data.columns = data.columns.str.strip()
        if 'Category' not in data.columns or 'Message' not in data.columns:
            messagebox.showerror("Error", "Dataset must have 'Category' and 'Message' columns!")
            data = None
            return

        messagebox.showinfo("Success", f"Loaded dataset: {os.path.basename(file_path)}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

# Train model
def train_classifier():
    global data, model, vectorizer, X_train, X_test, y_train, y_test
    if data is None:
        messagebox.showerror("Error", "Load dataset first!")
        return
    try:
        data = data.dropna(subset=['Category', 'Message'])
        data['Message'] = data['Message'].astype(str)
        data['Category'] = data['Category'].str.lower().map({'ham': 0, 'spam': 1})
        data = data.dropna(subset=['Category'])

        data['processed_text'] = data['Message'].apply(preprocess_text)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        X = vectorizer.fit_transform(data['processed_text'])
        y = data['Category']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = MultinomialNB()
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        # Save model and vectorizer
        joblib.dump(model, "spam_model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")

        messagebox.showinfo("Success", f"Model trained!\nAccuracy: {acc:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Training failed: {e}")

# Load existing model
def load_model():
    global model, vectorizer
    try:
        model = joblib.load("spam_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        messagebox.showinfo("Success", "Model loaded from disk.")
    except Exception:
        messagebox.showerror("Error", "No saved model found. Train first!")

# Classify a message
def classify_message():
    if model is None or vectorizer is None:
        messagebox.showerror("Error", "Train or load a model first!")
        return
    msg = input_message.get("1.0", tk.END).strip()
    if not msg:
        messagebox.showerror("Error", "Message cannot be empty!")
        return
    processed = preprocess_text(msg)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)
    result = "Spam" if prediction[0] == 1 else "Ham"
    messagebox.showinfo("Result", f"Message classified as: {result}")

# Show evaluation metrics
def show_metrics():
    if model is None or X_test is None:
        messagebox.showerror("Error", "Train the model first!")
        return
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["Ham", "Spam"])

    win = tk.Toplevel(root)
    win.title("Model Metrics")
    st = scrolledtext.ScrolledText(win, wrap=tk.WORD, width=70, height=20)
    st.insert(tk.END, f"Accuracy: {acc:.2f}\n\nConfusion Matrix:\n{cm}\n\nClassification Report:\n{report}")
    st.config(state=tk.DISABLED)
    st.pack(padx=10, pady=10)

# Show spam/ham counts
def show_counts():
    if data is None:
        messagebox.showerror("Error", "Load dataset first!")
        return
    spam_count = sum(data['Category'] == 1)
    ham_count = sum(data['Category'] == 0)

    plt.bar(['Ham', 'Spam'], [ham_count, spam_count], color=['blue', 'red'])
    plt.title("Spam vs Ham Count")
    plt.show()

# Exit
def exit_app():
    root.destroy()

# GUI setup
root = tk.Tk()
root.title("Spam Detection (Naive Bayes + TF-IDF)")
root.geometry("400x600")

btn_load = tk.Button(root, text="Load Dataset", command=load_dataset, width=25)
btn_load.pack(pady=5)

btn_train = tk.Button(root, text="Train Model", command=train_classifier, width=25)
btn_train.pack(pady=5)

btn_load_model = tk.Button(root, text="Load Saved Model", command=load_model, width=25)
btn_load_model.pack(pady=5)

btn_metrics = tk.Button(root, text="Show Metrics", command=show_metrics, width=25)
btn_metrics.pack(pady=5)

btn_counts = tk.Button(root, text="Show Spam/Ham Counts", command=show_counts, width=25)
btn_counts.pack(pady=5)

tk.Label(root, text="Enter message:").pack(pady=5)
input_message = tk.Text(root, height=5, width=40)
input_message.pack(pady=5)

btn_classify = tk.Button(root, text="Classify Message", command=classify_message, width=25)
btn_classify.pack(pady=5)

btn_exit = tk.Button(root, text="Exit", command=exit_app, width=25)
btn_exit.pack(pady=5)

root.mainloop()
