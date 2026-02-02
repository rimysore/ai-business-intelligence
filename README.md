# 🤖 AI Business Intelligence Platform

Full-stack AI application featuring **sales forecasting**, **sentiment analysis**, and **intelligent Q&A** powered by PyTorch, FastAPI, and React.

## ✨ Features

### 📈 Sales Forecasting
- LSTM neural network predicts next 7 days of sales
- 33,671 parameters trained on 3 years of data
- Interactive charts and trend analysis

### 💬 Sentiment Analysis  
- BERT model classifies customer review sentiment
- 109M parameters with 90%+ accuracy
- Real-time analysis of multiple reviews

### 🦙 Business Q&A
- Llama 3.2 with RAG answers questions from documents
- Semantic search through company knowledge base
- Contextual responses with source citations

## 🛠️ Tech Stack

**AI/ML**: PyTorch • BERT • LSTM • Llama 3.2 • LangChain • ChromaDB  
**Backend**: FastAPI • Python • MLflow  
**Frontend**: React • Recharts • Axios  
**Infrastructure**: Ollama • Vector Database

## 🚀 Quick Start
```bash
# Backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python backend/api/main.py

# Ollama (separate terminal)
ollama serve
ollama pull llama3.2:3b

# Frontend (separate terminal)
cd frontend
npm install
npm start
```

Visit: http://localhost:3000

## 📊 Architecture
```
React Frontend (3000) ──► FastAPI Backend (8000) ──► AI Models
                                                    ├─ LSTM (PyTorch)
                                                    ├─ BERT (PyTorch)
                                                    └─ Llama + RAG
```

## 📁 Structure
```
├── backend/
│   ├── models/          # AI models (LSTM, BERT)
│   ├── rag/             # RAG system
│   └── api/             # FastAPI endpoints
├── frontend/            # React dashboard
├── data/                # Datasets
└── notebooks/           # Exploratory analysis
```

## 🎯 Key Achievements

- ✅ End-to-end ML pipeline from data to deployment
- ✅ Production-ready REST API
- ✅ Local LLM with zero API costs
- ✅ Interactive web dashboard
- ✅ MLOps best practices (MLflow tracking)

## 📫 Contact

**GitHub**: [@rimysore](https://github.com/rimysore)

---

⭐ Star this repo if you found it helpful!
