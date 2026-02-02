"""
Sentiment Analysis API Endpoint
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.sentiment.model import SentimentBERT

router = APIRouter()

# Load model on startup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
checkpoint = torch.load('backend/models/saved_models/bert_best.pt',
                       map_location=device, weights_only=False)

sentiment_model = SentimentBERT()
sentiment_model.model.load_state_dict(checkpoint['model_state_dict'])
sentiment_model.model.eval()

print("✅ Sentiment analysis model loaded")

# Request/Response models
class SentimentRequest(BaseModel):
    texts: List[str]

class SentimentResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict

class SentimentResponse(BaseModel):
    results: List[SentimentResult]
    summary: dict

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment of customer reviews
    
    - **texts**: List of review texts to analyze
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        # Get predictions
        predictions, probabilities = sentiment_model.predict(request.texts)
        
        # Format results
        results = []
        for text, pred, prob in zip(request.texts, predictions, probabilities):
            sentiment = "positive" if pred == 1 else "negative"
            confidence = float(prob[pred])
            
            results.append(SentimentResult(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                probabilities={
                    "negative": float(prob[0]),
                    "positive": float(prob[1])
                }
            ))
        
        # Summary
        positive_count = sum(1 for r in results if r.sentiment == "positive")
        negative_count = len(results) - positive_count
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        return SentimentResponse(
            results=results,
            summary={
                "total": len(results),
                "positive": positive_count,
                "negative": negative_count,
                "positive_percentage": (positive_count / len(results)) * 100,
                "average_confidence": avg_confidence
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
