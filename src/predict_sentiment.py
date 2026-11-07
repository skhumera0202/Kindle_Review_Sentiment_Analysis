import joblib
from pathlib import Path
from preprocessing import clean_text

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load saved TF-IDF vectorizer
tfidf = joblib.load(
    PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
)

# Load Logistic Regression model
model = joblib.load(
    PROJECT_ROOT / "models" / "logistic_regression.pkl"
)


def predict_sentiment(review):

    # Clean the input review
    cleaned_review = clean_text(review)

    # Convert review into TF-IDF features
    review_tfidf = tfidf.transform([cleaned_review])

    # Predict sentiment
    prediction = model.predict(review_tfidf)[0]

    if prediction == 1:
        return "Positive"

    return "Negative"


# Run prediction
if __name__ == "__main__":

    print("\n Kindle Review Sentiment Predictor")
    print("----------------------------------")

    review = input("\nEnter your book review: ")

    result = predict_sentiment(review)

    print("\nPredicted Sentiment:", result)