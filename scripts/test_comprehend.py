"""Test Amazon Comprehend integration."""
import sys
sys.path.insert(0, ".")

from src.aws.comprehend import get_comprehend_client

client = get_comprehend_client()
print("Comprehend client enabled:", client.enabled if client else False)

# Test with sample reviews
test_reviews = [
    "This product is terrible! It broke after 2 days. Very disappointed.",
    "Amazing quality, fast shipping. Will buy again!",
    "The item is okay, nothing special but does the job.",
]

print()
print("Testing sentiment analysis on sample reviews:")
print("=" * 60)

for review in test_reviews:
    result = client.analyze_sentiment(review)
    print(f"Review: {review[:50]}...")
    print(f"  Sentiment: {result['sentiment']}")
    print(f"  Source: {result['source']}")
    pos = result['scores']['positive']
    neg = result['scores']['negative']
    print(f"  Scores: Positive={pos:.1%}, Negative={neg:.1%}")
    print()

# Test key phrase extraction
print("=" * 60)
print("Testing key phrase extraction:")
print("=" * 60)
test_text = "The delivery was very slow and the product quality is poor. Customer service was unhelpful."
phrases = client.extract_key_phrases(test_text)
print(f"Text: {test_text}")
print(f"Key phrases found: {[p['text'] for p in phrases]}")

# Test full review analysis
print()
print("=" * 60)
print("Testing comprehensive review analysis:")
print("=" * 60)
analysis = client.analyze_review("I love this product! The quality is excellent and it arrived quickly. Best purchase ever!")
print(f"Overall sentiment: {analysis['sentiment']['sentiment']}")
print(f"Is complaint: {analysis['is_complaint']}")
print(f"Topics: {analysis['topics']}")
