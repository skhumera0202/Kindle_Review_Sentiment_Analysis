
import pandas as pd
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

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

# Transform text data
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Balanced Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Multinomial Naive Bayes": MultinomialNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

# Store results
results = []

# Train and evaluate models
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    negative_recall = recall_score(
        y_test, y_pred, pos_label=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Negative Recall": negative_recall
    })

    # Save trained model
    model_filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        PROJECT_ROOT / "models" / model_filename
    )

    print(f"{name} completed!")

# Create comparison table
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Save comparison results
results_df.to_csv(
    PROJECT_ROOT / "results" / "model_comparison.csv",
    index=False
)

print("\nModel comparison completed!")
print("Results saved in results/model_comparison.csv")
print("Trained models saved in models/")
