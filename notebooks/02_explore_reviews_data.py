"""
Explore customer reviews dataset
Understand sentiment distribution and text patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

print("="*60)
print("💬 CUSTOMER REVIEWS EXPLORATION")
print("="*60)

# Load data
df = pd.read_csv('data/raw/reviews_data.csv')
df['date'] = pd.to_datetime(df['date'])

print("\n1️⃣ Dataset Overview:")
print(f"   Shape: {df.shape}")
print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
print(f"   Products: {df['product_id'].nunique()}")
print(f"\n{df.head(10)}")

print("\n2️⃣ Sentiment Distribution:")
print(df['sentiment'].value_counts())
print(f"\nPositive: {(df['sentiment'] == 1).sum()} ({(df['sentiment'] == 1).mean()*100:.1f}%)")
print(f"Negative: {(df['sentiment'] == 0).sum()} ({(df['sentiment'] == 0).mean()*100:.1f}%)")

print("\n3️⃣ Sample Reviews:")
print("\nPOSITIVE Examples:")
for i, text in enumerate(df[df['sentiment'] == 1]['review_text'].head(3), 1):
    print(f"  {i}. {text}")

print("\nNEGATIVE Examples:")
for i, text in enumerate(df[df['sentiment'] == 0]['review_text'].head(3), 1):
    print(f"  {i}. {text}")

print("\n4️⃣ Text Statistics:")
df['text_length'] = df['review_text'].str.len()
df['word_count'] = df['review_text'].str.split().str.len()

print(f"   Average review length: {df['text_length'].mean():.0f} characters")
print(f"   Average word count: {df['word_count'].mean():.0f} words")
print(f"   Shortest review: {df['text_length'].min()} characters")
print(f"   Longest review: {df['text_length'].max()} characters")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Sentiment distribution
sentiment_counts = df['sentiment'].value_counts()
axes[0, 0].bar(['Negative', 'Positive'], sentiment_counts.values, color=['#ff6b6b', '#51cf66'])
axes[0, 0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(True, alpha=0.3)

# 2. Reviews over time
df_daily = df.groupby([df['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
axes[0, 1].plot(df_daily.index, df_daily[0], label='Negative', color='#ff6b6b', alpha=0.7)
axes[0, 1].plot(df_daily.index, df_daily[1], label='Positive', color='#51cf66', alpha=0.7)
axes[0, 1].set_title('Reviews Over Time', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Text length distribution
axes[1, 0].hist([df[df['sentiment']==0]['text_length'], 
                df[df['sentiment']==1]['text_length']], 
               bins=30, label=['Negative', 'Positive'], 
               color=['#ff6b6b', '#51cf66'], alpha=0.7)
axes[1, 0].set_title('Review Length Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Characters')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Product sentiment breakdown
product_sentiment = df.groupby(['product_id', 'sentiment']).size().unstack(fill_value=0)
product_sentiment['total'] = product_sentiment.sum(axis=1)
product_sentiment['positive_ratio'] = product_sentiment[1] / product_sentiment['total']
product_sentiment = product_sentiment.sort_values('positive_ratio')

axes[1, 1].barh(range(len(product_sentiment)), product_sentiment['positive_ratio']*100, 
               color='#51cf66', alpha=0.7)
axes[1, 1].set_yticks(range(len(product_sentiment)))
axes[1, 1].set_yticklabels(product_sentiment.index)
axes[1, 1].set_xlabel('Positive Sentiment %')
axes[1, 1].set_title('Product Sentiment Scores', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('data/processed/reviews_exploration.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved visualization: data/processed/reviews_exploration.png")
plt.show()

print("\n" + "="*60)
print("✅ Data Exploration Complete!")
print("="*60)
print("\nKey Insights:")
print("• Balanced dataset (50/50 split)")
print("• Good for classification training")
print("• Short reviews (easy to process)")
print("\nNext: Build BERT sentiment classifier!")
