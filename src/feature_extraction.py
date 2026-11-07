import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load training and testing data
train_df = pd.read_csv(PROJECT_ROOT / "data" / "train.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")

# Separate reviews and labels
X_train = train_df["cleaned_review"]
X_test = test_df["cleaned_review"]

# Initialize TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

# Fit on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data
X_test_tfidf = tfidf.transform(X_test)

# Save TF-IDF vectorizer
joblib.dump(
    tfidf,
    PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
)

print("TF-IDF feature extraction completed!")

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)