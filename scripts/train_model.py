# train_model.py - Train URL tracking detection model

import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("../data/urls.csv")

# Preprocess URLs
def preprocess_url(url):
    return re.sub(r"https?://(www\.)?", "", url)

df['processed_url'] = df['url'].apply(preprocess_url)

# Feature extraction - TF-IDF on character n-grams
vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
X = vectorizer.fit_transform(df['processed_url'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {accuracy:.2f}")

# Save model & vectorizer
joblib.dump(model, "../models/tracker_detection_model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")
print("✅ Model and vectorizer saved successfully.")
