import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load processed dataset
df = pd.read_csv(
    PROJECT_ROOT / "data" / "processed_kindle_reviews.csv"
)

# Separate features and target
X = df["cleaned_review"]
y = df["sentiment"]

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save training data
train_df = pd.DataFrame({
    "cleaned_review": X_train,
    "sentiment": y_train
})

# Save testing data
test_df = pd.DataFrame({
    "cleaned_review": X_test,
    "sentiment": y_test
})

train_df.to_csv(PROJECT_ROOT / "data" / "train.csv", index=False)
test_df.to_csv(PROJECT_ROOT / "data" / "test.csv", index=False)

print("Train-test split completed!")
print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)