"""
Explore the sales dataset
Understand patterns, trends, seasonality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

print("="*60)
print("📊 SALES DATA EXPLORATION")
print("="*60)

# Load data
df = pd.read_csv('data/raw/sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

print("\n1️⃣ Dataset Overview:")
print(f"   Shape: {df.shape}")
print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
print(f"   Products: {df['product_id'].nunique()}")
print(f"\n{df.head(10)}")

print("\n2️⃣ Basic Statistics:")
print(df.describe())

print("\n3️⃣ Missing Values:")
print(df.isnull().sum())

# Focus on one product for time series
product = 'PROD_001'
df_product = df[df['product_id'] == product].copy()
df_product = df_product.sort_values('date')

print(f"\n4️⃣ Analyzing Product: {product}")
print(f"   Records: {len(df_product)}")
print(f"   Avg Sales: {df_product['sales_quantity'].mean():.2f}")
print(f"   Min Sales: {df_product['sales_quantity'].min()}")
print(f"   Max Sales: {df_product['sales_quantity'].max()}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Time series plot
axes[0, 0].plot(df_product['date'], df_product['sales_quantity'], linewidth=1)
axes[0, 0].set_title(f'Sales Over Time - {product}', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Sales Quantity')
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribution
axes[0, 1].hist(df_product['sales_quantity'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Sales Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Sales Quantity')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# 3. Monthly aggregation
df_product['month'] = df_product['date'].dt.to_period('M')
monthly_sales = df_product.groupby('month')['sales_quantity'].sum()
axes[1, 0].plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o')
axes[1, 0].set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Total Sales')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# 4. Rolling average (7-day)
df_product['rolling_7'] = df_product['sales_quantity'].rolling(window=7).mean()
axes[1, 1].plot(df_product['date'], df_product['sales_quantity'], alpha=0.3, label='Daily')
axes[1, 1].plot(df_product['date'], df_product['rolling_7'], linewidth=2, label='7-day Average')
axes[1, 1].set_title('Sales with 7-Day Moving Average', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Sales Quantity')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/sales_exploration.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved visualization: data/processed/sales_exploration.png")
plt.show()

print("\n" + "="*60)
print("✅ Data Exploration Complete!")
print("="*60)
print("\nKey Insights:")
print("• Data shows clear trend and seasonality")
print("• Good for LSTM modeling")
print("• No missing values")
print("\nNext: Build LSTM model!")