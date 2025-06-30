# 🧠 ChatGPT VibeCheck

**ChatGPT VibeCheck** is a beginner-friendly Machine Learning project that performs **sentiment analysis** on real tweets about ChatGPT. The app is trained to classify sentiments into **Positive**, **Neutral**, or **Negative**, and is deployed using **Gradio** for a clean web interface.

---

## 📌 What It Does

- Cleans and pre-processes real ChatGPT tweets
- Trains a Logistic Regression model with TF-IDF vectorization
- Predicts sentiment for any user input
- Visualizes data with **Matplotlib** and **Seaborn**
- Launches a **Gradio web UI** for real-time sentiment detection

---


---

## 🧪 Features

- ✅ Real-time sentiment predictions using Gradio
- ✅ Pre-trained model saved and loaded via `joblib`
- ✅ Interactive web interface for easy testing
- ✅ Visualization of results with `matplotlib` & `seaborn`
- ✅ Clean code with reusable components

---
🧠 Technologies Used
.Python 🐍

.scikit-learn

.pandas, numpy

.matplotlib, seaborn

.joblib

.Gradio

---

## 🗂 Folder Structure
chatgpt_vibecheck/
│

├── vibecheck_train.py # Training + Plots + Model saving

├── vibecheck_app.py # Gradio app

├── requirements.txt

├── .gitignore

├──file.csv

├── models/

│ ├── vibecheck_model.pkl

│ └── vibecheck_vectorizer.pkl

│

├── sentiment_distribution.png

├── confusion_matrix.png


