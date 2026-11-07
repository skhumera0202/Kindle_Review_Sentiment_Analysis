import pandas as pd

from preprocessing import clean_text


# Load dataset
df = pd.read_csv("../data/all_kindle_review.csv")

# Create sentiment labels
df["sentiment"] = df["rating"].apply(
    lambda x: 0 if x < 3 else 1
)

# Clean review text
df["cleaned_review"] = df["reviewText"].apply(clean_text)

# Keep required columns
processed_df = df[["cleaned_review", "sentiment"]]

# Save processed dataset
processed_df.to_csv(
    "../data/processed_kindle_reviews.csv",
    index=False
)

print("Preprocessing completed!")
print("Dataset shape:", processed_df.shape)
print("\nSentiment distribution:")
print(processed_df["sentiment"].value_counts())

print("\nSample processed reviews:")
print(processed_df.head())