import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load test dataset
test_df = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")

X_test = test_df["cleaned_review"]
y_test = test_df["sentiment"]

# Load TF-IDF and trained model
tfidf = joblib.load(PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl")
model = joblib.load(PROJECT_ROOT / "models" / "sentiment_model.pkl")

# Transform test reviews
X_test_tfidf = tfidf.transform(X_test)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"]
    )
)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))