"""
Business Q&A API Endpoint (RAG + Llama)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag.llama_rag import LlamaRAG

router = APIRouter()

# Load RAG system on startup
rag_system = LlamaRAG()
print("✅ RAG + Llama system loaded")

# Request/Response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

@router.post("/chat", response_model=ChatResponse)
async def business_qa(request: ChatRequest):
    """
    Ask questions about business documents
    
    - **question**: Your business question
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Get answer from RAG system
        result = rag_system.ask(request.question, verbose=False)
        
        return ChatResponse(
            question=result['question'],
            answer=result['answer'],
            sources=result['sources']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
