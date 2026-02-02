import sys

def test_imports():
    """Test all critical imports"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        
        import langchain
        print(f"✅ LangChain: {langchain.__version__}")
        
        import chromadb
        print(f"✅ ChromaDB: {chromadb.__version__}")
        
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
        
        import mlflow
        print(f"✅ MLflow: {mlflow.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
        
        print("\n🎉 All packages installed successfully!")
        print("🚀 You're ready to build!")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()