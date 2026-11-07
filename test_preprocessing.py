from src.preprocessing import clean_text

# Sample Kindle review
review = """
This BOOK is AMAZING!!! I really enjoyed reading it.
Visit https://example.com for more details.
The characters were running through the beautiful story.
"""

# Apply preprocessing
cleaned_review = clean_text(review)

print("Original Review:")
print(review)

print("\nCleaned Review:")
print(cleaned_review)