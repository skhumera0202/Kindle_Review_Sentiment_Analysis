
import gc
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Store comparison results
results = []

# Define vectorizers
vectorizers = {
    "Bag of Words": CountVectorizer(
        max_features=5000,
        dtype=np.float32
    ),
    "TF-IDF": TfidfVectorizer(
        max_features=5000,
        dtype=np.float32
    )
}

# Train and evaluate GaussianNB with each representation
for name, vectorizer in vectorizers.items():

    print(f"\n========== {name} ==========")

    # Convert text into numerical features
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    print("Training matrix:", X_train_features.shape)
    print("Testing matrix:", X_test_features.shape)

    # GaussianNB requires dense matrices
    X_train_dense = X_train_features.toarray()
    X_test_dense = X_test_features.toarray()

    # Train model
    model = GaussianNB()
    model.fit(X_train_dense, y_train)

    # Make predictions
    y_pred = model.predict(X_test_dense)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Store results
    results.append({
        "Technique": name,
        "Model": "GaussianNB",
        "Accuracy": accuracy,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    })

    # Save model and vectorizer
    filename = name.lower().replace(" ", "_")

    joblib.dump(
        vectorizer,
        PROJECT_ROOT / "models" / f"{filename}_vectorizer.pkl"
    )

    joblib.dump(
        model,
        PROJECT_ROOT / "models" / f"gaussian_nb_{filename}.pkl"
    )

    print(f"\n{name} model and vectorizer saved!")

    # Release memory before the next experiment
    del X_train_features, X_test_features
    del X_train_dense, X_test_dense, model
    gc.collect()

# Create comparison table
results_df = pd.DataFrame(results)

print("\n========== BOW vs TF-IDF COMPARISON ==========")
print(results_df.to_string(index=False))

# Save comparison results
results_df.to_csv(
    PROJECT_ROOT / "results" / "gaussian_nb_comparison.csv",
    index=False
)

print("\nComparison completed successfully!")
print("Results saved in results/gaussian_nb_comparison.csv")
