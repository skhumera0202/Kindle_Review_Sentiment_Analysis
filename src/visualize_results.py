import pandas as pd
import joblib

import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load test data
test_df = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")

X_test = test_df["cleaned_review"]
y_test = test_df["sentiment"]

# Load saved models
tfidf = joblib.load(PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl")
model = joblib.load(PROJECT_ROOT / "models" / "sentiment_model.pkl")

# Transform and predict
X_test_tfidf = tfidf.transform(X_test)
y_pred = model.predict(X_test_tfidf)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display chart
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

display.plot(cmap="Blues", values_format="d")

plt.title("Kindle Review Sentiment - Confusion Matrix")
plt.tight_layout()

# Create results folder if needed
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)

# Save chart
plt.savefig(
    results_dir / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("Confusion matrix chart saved successfully!")
print("Saved at:", results_dir / "confusion_matrix.png")

plt.show()