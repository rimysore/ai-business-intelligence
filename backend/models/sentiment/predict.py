"""
Test BERT Sentiment Model on new reviews
"""

import torch
from model import SentimentBERT

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("="*60)
print("💬 SENTIMENT ANALYSIS - PREDICTIONS")
print("="*60)

# Load trained model
print("\n📦 Loading trained BERT model...")
checkpoint = torch.load('backend/models/saved_models/bert_best.pt', 
                       map_location=device, weights_only=False)

sentiment_model = SentimentBERT()
sentiment_model.model.load_state_dict(checkpoint['model_state_dict'])
sentiment_model.model.eval()

print(f"✅ Model loaded!")
print(f"   Training accuracy: {checkpoint['train_acc']:.4f}")
print(f"   Validation accuracy: {checkpoint['val_acc']:.4f}")

# Test reviews
test_reviews = [
    "This product is absolutely amazing! Best purchase I've ever made. Highly recommend!",
    "Terrible quality. Broke after one day. Complete waste of money.",
    "It's okay, nothing special. Works as expected but not impressive.",
    "Love it! Exceeded all my expectations. Will definitely buy again!",
    "Very disappointed. Poor customer service and defective product.",
    "Great value for money. Good quality and fast shipping.",
    "Do not buy this! Worst product ever. Save your money.",
    "Pretty good overall. Minor issues but mostly satisfied.",
]

print("\n" + "="*60)
print("🔮 ANALYZING CUSTOMER REVIEWS")
print("="*60)

# Analyze each review
predictions, probabilities = sentiment_model.predict(test_reviews)

for i, (review, pred, prob) in enumerate(zip(test_reviews, predictions, probabilities), 1):
    sentiment = "✅ POSITIVE" if pred == 1 else "❌ NEGATIVE"
    confidence = prob[pred] * 100
    
    print(f"\n{i}. Review: \"{review}\"")
    print(f"   Prediction: {sentiment}")
    print(f"   Confidence: {confidence:.1f}%")
    print(f"   Probabilities: Neg={prob[0]*100:.1f}%, Pos={prob[1]*100:.1f}%")

print("\n" + "="*60)
print("✅ SENTIMENT ANALYSIS COMPLETE!")
print("="*60)

# Summary
positive_count = sum(predictions == 1)
negative_count = sum(predictions == 0)
avg_confidence = probabilities.max(axis=1).mean() * 100

print(f"\nSummary of {len(test_reviews)} reviews:")
print(f"   Positive: {positive_count} ({positive_count/len(test_reviews)*100:.0f}%)")
print(f"   Negative: {negative_count} ({negative_count/len(test_reviews)*100:.0f}%)")
print(f"   Average confidence: {avg_confidence:.1f}%")
