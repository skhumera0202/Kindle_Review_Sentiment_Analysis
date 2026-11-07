# Kindle Review Sentiment Analysis

## Project Overview

This project focuses on classifying Kindle book reviews into two sentiment categories:

- **Positive (1)**
- **Negative (0)**

The project explores different Natural Language Processing (NLP) techniques, text vectorization methods, and machine learning algorithms to compare their performance on Kindle reviews.

## Dataset

- Total reviews: 12,000
- Total original columns: 11
- Negative reviews: 4,000
- Positive reviews: 8,000
- Training samples: 9,600
- Testing samples: 2,400

### Sentiment Labeling

The review ratings were converted into binary sentiment labels:

- Rating below 3 → Negative (0)
- Rating 3 or above → Positive (1)

**Note:** Three-star reviews are classified as positive in this project.

The dataset source and redistribution license have not yet been verified. Therefore, the dataset is excluded from this repository.

## Project Workflow

1. Data loading and exploration
2. Text preprocessing
3. Sentiment label creation
4. Train-test splitting
5. Text feature extraction
6. Machine learning model training
7. Model evaluation
8. Model comparison
9. Confusion matrix visualization
10. Sentiment prediction on new reviews

## Text Preprocessing

The following preprocessing techniques were applied:

- Lowercasing
- HTML tag removal
- URL removal
- Special character removal
- Tokenization
- Stopword removal
- Part-of-Speech (POS) tagging
- Lemmatization

## Feature Extraction Techniques

The project explores three text representation approaches:

- **Bag of Words (BoW):** Represents text using word-frequency counts.
- **TF-IDF:** Assigns importance to words based on their frequency across documents.
- **CountVectorizer:** Converts text into a matrix of token counts.

TF-IDF and Bag of Words were configured with a maximum of 5,000 features.

## Machine Learning Models

The following algorithms were evaluated:

### TF-IDF-based models

- Logistic Regression
- Balanced Logistic Regression
- Multinomial Naive Bayes
- Random Forest

### Gaussian Naive Bayes experiments

- Gaussian Naive Bayes with Bag of Words
- Gaussian Naive Bayes with TF-IDF

The Gaussian Naive Bayes experiments were conducted separately for the two feature extraction methods.

## Model Results

### TF-IDF Model Comparison

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 84.50% | 88.98% |
| Balanced Logistic Regression | 83.92% | 87.60% |
| Multinomial Naive Bayes | 81.04% | 87.14% |
| Random Forest | 79.88% | 86.27% |

### Gaussian Naive Bayes Comparison

| Feature Extraction | Accuracy |
|---|---:|
| Bag of Words | 52.00% |
| TF-IDF | 58.42% |

### Key Findings

- Logistic Regression achieved the highest accuracy among the evaluated TF-IDF models.
- Balanced Logistic Regression improved negative-class recall compared with standard Logistic Regression.
- TF-IDF performed better than Bag of Words in the Gaussian Naive Bayes experiment.
- Different feature extraction techniques and algorithms produced different classification results.

## Visualizations

The following visualizations were generated:

- `confusion_matrix.png` — Logistic Regression confusion matrix.
- `confusion_matrix_comparison.png` — Comparison of TF-IDF model confusion matrices.
- `gaussian_nb_confusion_matrices.png` — Gaussian Naive Bayes confusion matrix comparison.
- `model_comparison.png` — Comparison of TF-IDF-based machine learning models.

## Project Structure

```text
Kindle_Review_Sentiment_Analysis/
│
├── data/
│   ├── all_kindle_review.csv
│   ├── processed_kindle_reviews.csv
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── Saved vectorizers
│   └── Trained machine learning models
│
├── results/
│   ├── confusion_matrix.png
│   ├── confusion_matrix_comparison.png
│   ├── gaussian_nb_comparison.csv
│   ├── gaussian_nb_confusion_matrices.png
│   ├── model_comparison.csv
│   └── model_comparison.png
│
├── src/
│   ├── compare_bow_tfidf_gaussian_nb.py
│   ├── compare_confusion_matrices.py
│   ├── compare_models.py
│   ├── evaluate_model.py
│   ├── feature_extraction.py
│   ├── predict_sentiment.py
│   ├── preprocessing.py
│   ├── preprocess_data.py
│   ├── split_data.py
│   ├── train_model.py
│   ├── visualize_gaussian_nb_results.py
│   ├── visualize_model_comparison.py
│   └── visualize_results.py
│
├── .gitignore
├── requirements.txt
├── test_preprocessing.py
└── README.md
```

**Note:** Dataset CSV files and generated model `.pkl` files are excluded from version control. The folder structure above describes the local project; those excluded files will not appear in the public repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/skhumera0202/Kindle_Review_Sentiment_Analysis.git
cd Kindle_Review_Sentiment_Analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Obtain the original Kindle review dataset from a source whose usage and redistribution license permits your intended use.

Place the CSV file here:

```text
data/all_kindle_review.csv
```

### 5. Download required NLTK resources

```python
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
```

## Running the Project

Run the following commands from the project root directory in the listed order.

### Step 1: Test preprocessing

```powershell
python .\test_preprocessing.py
```

### Step 2: Preprocess the dataset

```powershell
python .\src\preprocess_data.py
```

### Step 3: Split the dataset

```powershell
python .\src\split_data.py
```

### Step 4: Extract TF-IDF features

```powershell
python .\src\feature_extraction.py
```

### Step 5: Train the initial Logistic Regression model

```powershell
python .\src\train_model.py
```

### Step 6: Evaluate the model

```powershell
python .\src\evaluate_model.py
```

### Step 7: Compare machine learning models

```powershell
python .\src\compare_models.py
```

### Step 8: Run Gaussian Naive Bayes experiments

```powershell
python .\src\compare_bow_tfidf_gaussian_nb.py
```

### Step 9: Generate visualizations

```powershell
python .\src\visualize_results.py
python .\src\visualize_model_comparison.py
python .\src\compare_confusion_matrices.py
python .\src\visualize_gaussian_nb_results.py
```

### Step 10: Predict sentiment

After generating the required TF-IDF vectorizer and Logistic Regression model:

```powershell
python .\src\predict_sentiment.py
```

## Example Predictions

| Review | Predicted Sentiment |
|---|---|
| This book is amazing and I really enjoyed reading it. | Positive |
| This book is boring, disappointing, and a waste of money. | Negative |

These examples are functional prediction checks, not additional test-set accuracy measurements.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- BeautifulSoup
- Matplotlib
- Joblib

## Limitations and Future Improvements

- Verify and document the original dataset source and license.
- Experiment with additional NLP feature extraction techniques.
- Explore advanced machine learning and deep learning models.
- Improve negative sentiment classification.
- Experiment with hyperparameter tuning and cross-validation.
- Explore deployment as an API.

## Author

**Humera Shaikh**

Aspiring Machine Learning Engineer

GitHub: [skhumera0202](https://github.com/skhumera0202)