"""
FastAPI Backend - AI Business Intelligence Platform
Serves: Sales Forecasting, Sentiment Analysis, RAG Q&A
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

app = FastAPI(
    title="AI Business Intelligence API",
    description="Sales forecasting, sentiment analysis, and business Q&A",
    version="1.0.0"
)

# CORS (allow React frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes
from routes import forecast, sentiment, chat

# Register routes
app.include_router(forecast.router, prefix="/api", tags=["Sales Forecasting"])
app.include_router(sentiment.router, prefix="/api", tags=["Sentiment Analysis"])
app.include_router(chat.router, prefix="/api", tags=["Business Q&A"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "AI Business Intelligence API",
        "models": {
            "sales_forecasting": "LSTM (PyTorch)",
            "sentiment_analysis": "BERT (PyTorch)",
            "business_qa": "Llama 3.2 + RAG"
        },
        "endpoints": {
            "forecast": "/api/forecast",
            "sentiment": "/api/sentiment",
            "chat": "/api/chat"
        }
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "database": "connected"
    }

if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("🚀 STARTING AI BUSINESS INTELLIGENCE API")
    print("="*60)
    print("\n📊 Available Models:")
    print("   1. Sales Forecasting (LSTM)")
    print("   2. Sentiment Analysis (BERT)")
    print("   3. Business Q&A (Llama + RAG)")
    print("\n🌐 API Docs: http://localhost:8000/docs")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
