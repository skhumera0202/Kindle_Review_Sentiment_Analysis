import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load datasets
train_df = pd.read_csv(PROJECT_ROOT / "data" / "train.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")

# Separate features and labels
X_train = train_df["cleaned_review"]
y_train = train_df["sentiment"]

X_test = test_df["cleaned_review"]
y_test = test_df["sentiment"]

# Load TF-IDF vectorizer
tfidf = joblib.load(
    PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
)

# Convert reviews into numerical features
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize Logistic Regression
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions
y_train_pred = model.predict(X_train_tfidf)
y_test_pred = model.predict(X_test_tfidf)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Save trained model
joblib.dump(
    model,
    PROJECT_ROOT / "models" / "sentiment_model.pkl"
)

print("Model training completed!")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)