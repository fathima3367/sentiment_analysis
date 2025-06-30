import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Load dataset
df = pd.read_csv(r"C:\Users\fma79\Desktop\chatgpt_VibeCheck\file.csv")
print(df.columns)

# Preprocess text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['cleaned_text'] = df['tweets'].apply(clean_text)

# Drop empty rows
df = df[df['cleaned_text'].str.strip() != '']

# Drop empty rows
df = df[df['cleaned_text'].str.strip() != '']

# Show first 10 rows 
print(df[['tweets', 'cleaned_text']].head(10))  

# Plot sentiment distribution
plt.figure(figsize=(6,4))
df['labels'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('sentiment_distribution.png')
plt.show()

# Vectorize text
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['labels']
print('VECTORIZING TEXT COMPLETED')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(solver='lbfgs', max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('TRAINING COMPLETED')

# Evaluate
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.show()

# Save model and vectorizer
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/vibecheck_model.pkl')
joblib.dump(vectorizer, 'models/vibecheck_vectorizer.pkl')