
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load test dataset
test_df = pd.read_csv(
    PROJECT_ROOT / "data" / "test.csv"
)

X_test = test_df["cleaned_review"]
y_test = test_df["sentiment"]

# Load TF-IDF vectorizer
tfidf = joblib.load(
    PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
)

# Transform test reviews
X_test_tfidf = tfidf.transform(X_test)

# Load two Logistic Regression models
models = {
    "Original Logistic Regression": joblib.load(
        PROJECT_ROOT / "models" / "logistic_regression.pkl"
    ),

    "Balanced Logistic Regression": joblib.load(
        PROJECT_ROOT / "models" / "balanced_logistic_regression.pkl"
    )
}

# Create figure with two confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Generate confusion matrices
for ax, (name, model) in zip(axes, models.items()):

    y_pred = model.predict(X_test_tfidf)

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(name)

# Overall title
plt.suptitle(
    "Original vs Balanced Logistic Regression",
    fontsize=14
)

plt.tight_layout()

# Create results folder
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)

# Save chart
chart_path = results_dir / "confusion_matrix_comparison.png"

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

print("Confusion matrix comparison saved successfully!")
print("Saved at:", chart_path)

plt.show()
