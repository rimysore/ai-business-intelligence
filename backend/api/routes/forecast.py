"""
Sales Forecasting API Endpoint
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import pandas as pd
import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.timeseries.model import SalesLSTM

router = APIRouter()

# Load model on startup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
checkpoint = torch.load('backend/models/saved_models/lstm_best.pt', 
                       map_location=device, weights_only=False)

model = SalesLSTM(
    input_size=1,
    hidden_size=checkpoint['config']['hidden_size'],
    num_layers=checkpoint['config']['num_layers'],
    output_size=checkpoint['config']['prediction_horizon'],
    dropout=checkpoint['config']['dropout']
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
scaler = checkpoint['scaler']

print("✅ Sales forecasting model loaded")

# Request/Response models
class ForecastRequest(BaseModel):
    product_id: str = "PROD_001"
    days_to_predict: int = 7

class ForecastResponse(BaseModel):
    product_id: str
    predictions: List[float]
    dates: List[str]
    historical_avg: float
    forecast_avg: float
    trend: str

@router.post("/forecast", response_model=ForecastResponse)
async def forecast_sales(request: ForecastRequest):
    """
    Predict future sales for a product
    
    - **product_id**: Product to forecast (default: PROD_001)
    - **days_to_predict**: Forecast horizon (default: 7 days)
    """
    try:
        # Load data
        df = pd.read_csv('data/raw/sales_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter product
        df_product = df[df['product_id'] == request.product_id].sort_values('date')
        
        if len(df_product) < 30:
            raise HTTPException(status_code=404, detail="Product not found or insufficient data")
        
        # Get last 30 days
        sales_data = df_product['sales_quantity'].values[-30:]
        last_date = df_product['date'].values[-1]
        
        # Normalize
        sales_normalized = scaler.transform(sales_data.reshape(-1, 1)).flatten()
        
        # Prepare input
        X = torch.FloatTensor(sales_normalized).unsqueeze(0).unsqueeze(-1).to(device)
        
        # Predict
        with torch.no_grad():
            pred_normalized = model(X)
            predictions = scaler.inverse_transform(
                pred_normalized.cpu().numpy()
            ).flatten()
        
        # Generate dates
        future_dates = pd.date_range(
            start=pd.to_datetime(last_date) + pd.Timedelta(days=1),
            periods=request.days_to_predict,
            freq='D'
        )
        
        # Calculate trend
        historical_avg = float(sales_data.mean())
        forecast_avg = float(predictions.mean())
        change = ((forecast_avg - historical_avg) / historical_avg) * 100
        
        if change > 5:
            trend = "upward"
        elif change < -5:
            trend = "downward"
        else:
            trend = "stable"
        
        return ForecastResponse(
            product_id=request.product_id,
            predictions=predictions.tolist()[:request.days_to_predict],
            dates=[d.strftime('%Y-%m-%d') for d in future_dates],
            historical_avg=historical_avg,
            forecast_avg=forecast_avg,
            trend=trend
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
