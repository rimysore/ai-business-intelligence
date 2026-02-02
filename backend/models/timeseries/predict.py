"""
Make predictions with trained LSTM model
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import SalesLSTM
import pickle

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("="*60)
print("🔮 SALES FORECASTING - PREDICTIONS")
print("="*60)

# Load trained model
print("\n📦 Loading trained model...")
checkpoint = torch.load('backend/models/saved_models/lstm_best.pt', 
                       map_location=device, weights_only=False)

config = checkpoint['config']
scaler = checkpoint['scaler']

model = SalesLSTM(
    input_size=1,
    hidden_size=config['hidden_size'],
    num_layers=config['num_layers'],
    output_size=config['prediction_horizon'],
    dropout=config['dropout']
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded!")
print(f"   Training loss: {checkpoint['train_loss']:.4f}")
print(f"   Validation loss: {checkpoint['val_loss']:.4f}")

# Load data
print("\n📊 Loading sales data...")
df = pd.read_csv('data/raw/sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Get last 30 days of data for prediction
product_id = 'PROD_001'
df_product = df[df['product_id'] == product_id].sort_values('date')
sales_data = df_product['sales_quantity'].values

# Get last 30 days
last_30_days = sales_data[-30:]
dates = df_product['date'].values[-30:]

print(f"   Using last 30 days of {product_id}")
print(f"   Date range: {dates[0]} to {dates[-1]}")
print(f"   Sales range: {last_30_days.min():.0f} to {last_30_days.max():.0f}")

# Normalize
last_30_days_normalized = scaler.transform(last_30_days.reshape(-1, 1)).flatten()

# Prepare for model
X = torch.FloatTensor(last_30_days_normalized).unsqueeze(0).unsqueeze(-1).to(device)
# Shape: (1, 30, 1) = (batch_size, sequence_length, features)

# Make prediction
print("\n🔮 Making prediction...")
with torch.no_grad():
    prediction_normalized = model(X)
    prediction = scaler.inverse_transform(
        prediction_normalized.cpu().numpy()
    ).flatten()

# Generate future dates
last_date = pd.to_datetime(dates[-1])
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), 
    periods=7, 
    freq='D'
)

print("\n" + "="*60)
print("📈 7-DAY SALES FORECAST")
print("="*60)

for i, (date, pred) in enumerate(zip(future_dates, prediction), 1):
    print(f"Day {i} ({date.strftime('%Y-%m-%d')}): {pred:.0f} units")

print("="*60)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Historical + Prediction (FIXED)
historical_x = list(range(30))
forecast_x = list(range(29, 37))  # Start from day 29 to connect
forecast_y = [last_30_days[-1]] + list(prediction)

axes[0].plot(historical_x, last_30_days, 'b-', linewidth=2, label='Historical (30 days)')
axes[0].plot(forecast_x, forecast_y, 'r--', linewidth=2, marker='o', label='Forecast (7 days)')
axes[0].axvline(x=29.5, color='gray', linestyle=':', alpha=0.5)
axes[0].set_xlabel('Days')
axes[0].set_ylabel('Sales Quantity')
axes[0].set_title('Sales Forecast - Next 7 Days')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Full context (last 90 days + prediction)
last_90_days = sales_data[-90:]
dates_90 = df_product['date'].values[-90:]

forecast_90_x = list(range(89, 97))  # Start from day 89 to connect
forecast_90_y = [last_90_days[-1]] + list(prediction)

axes[1].plot(range(90), last_90_days, 'b-', linewidth=1, alpha=0.7, label='Last 90 days')
axes[1].plot(forecast_90_x, forecast_90_y, 'r-', linewidth=2, marker='o', label='7-day Forecast')
axes[1].axvline(x=89.5, color='gray', linestyle=':', alpha=0.5, label='Today')
axes[1].set_xlabel('Days')
axes[1].set_ylabel('Sales Quantity')
axes[1].set_title('Sales Trend - 90 Days Historical + 7 Days Forecast')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/sales_forecast.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: data/processed/sales_forecast.png")
plt.show()

# Summary statistics
print("\n📊 Forecast Summary:")
print(f"   Average predicted sales: {prediction.mean():.0f} units/day")
print(f"   Min predicted sales: {prediction.min():.0f} units")
print(f"   Max predicted sales: {prediction.max():.0f} units")
print(f"   Total 7-day forecast: {prediction.sum():.0f} units")

# Compare to historical average
recent_avg = last_30_days.mean()
forecast_avg = prediction.mean()
change = ((forecast_avg - recent_avg) / recent_avg) * 100

print(f"\n📈 Trend Analysis:")
print(f"   Last 30 days average: {recent_avg:.0f} units/day")
print(f"   Next 7 days average: {forecast_avg:.0f} units/day")
print(f"   Expected change: {change:+.1f}%")

if change > 5:
    print("   📈 UPWARD trend - Sales expected to increase!")
elif change < -5:
    print("   📉 DOWNWARD trend - Sales expected to decrease")
else:
    print("   ➡️  STABLE trend - Sales expected to remain steady")

print("\n" + "="*60)
print("✅ PREDICTION COMPLETE!")
print("="*60)
