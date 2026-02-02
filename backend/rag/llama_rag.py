"""
Llama + RAG Integration
Answer business questions using local Llama and document retrieval
"""

import requests
import json
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rag.embeddings import DocumentProcessor

class LlamaRAG:
    """
    RAG system powered by local Llama
    
    Flow:
    1. User asks question
    2. Search relevant documents (RAG)
    3. Send context + question to Llama
    4. Return answer with sources
    """
    
    def __init__(self, ollama_url="http://localhost:11434", model="llama3.2:3b"):
        """
        Initialize Llama RAG system
        
        Args:
            ollama_url: Ollama server URL
            model: Which Llama model to use
        """
        self.ollama_url = ollama_url
        self.model = model
        
        # Initialize document processor
        print("📚 Loading RAG system...")
        self.doc_processor = DocumentProcessor()
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✅ Ollama server connected")
                print(f"✅ Using model: {model}")
            else:
                print("⚠️  Warning: Ollama server may not be running")
        except:
            print("❌ Error: Ollama server not reachable")
            print("   Please run: ollama serve")
    
    def retrieve_context(self, question: str, n_results: int = 3):
        """
        Retrieve relevant document chunks for the question
        
        Args:
            question: User's question
            n_results: Number of chunks to retrieve
        
        Returns:
            context: Combined text from relevant documents
            sources: List of source documents
        """
        # Search documents
        results = self.doc_processor.search(question, n_results=n_results)
        
        # Extract documents and metadata
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # Combine into context
        context_parts = []
        sources = []
        
        for doc, metadata in zip(documents, metadatas):
            context_parts.append(f"[From {metadata['filename']}]\n{doc}")
            if metadata['filename'] not in sources:
                sources.append(metadata['filename'])
        
        context = "\n\n".join(context_parts)
        
        return context, sources
    
    def generate_answer(self, question: str, context: str):
        """
        Generate answer using Llama
        
        Args:
            question: User's question
            context: Retrieved context from documents
        
        Returns:
            answer: Llama's response
        """
        # Create prompt for Llama
        prompt = f"""You are a helpful business analyst assistant. Answer the question based ONLY on the provided context from company documents. If the answer is not in the context, say "I don't have that information in the available documents."

Context from company documents:
{context}

Question: {question}

Answer:"""
        
        # Call Ollama API
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                return f"Error: Could not get response from Llama (Status: {response.status_code})"
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def ask(self, question: str, verbose: bool = True):
        """
        Main function: Ask a question and get an answer
        
        Args:
            question: User's question
            verbose: Print detailed info
        
        Returns:
            Dictionary with answer and sources
        """
        if verbose:
            print(f"\n🔍 Question: {question}")
            print("📚 Searching documents...")
        
        # Retrieve relevant context
        context, sources = self.retrieve_context(question, n_results=3)
        
        if verbose:
            print(f"   Found {len(sources)} relevant documents")
            print("🦙 Asking Llama...")
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'context': context
        }


def demo():
    """
    Demo: Ask several business questions
    """
    print("="*60)
    print("🦙 LLAMA + RAG BUSINESS Q&A SYSTEM")
    print("="*60)
    
    # Initialize
    rag = LlamaRAG()
    
    # Test questions
    questions = [
        "What is our company's mission?",
        "What was the Q4 2023 revenue?",
        "What is our return policy?",
        "What are our main marketing initiatives for 2024?",
        "How many employees do we have?",
    ]
    
    print("\n" + "="*60)
    print("📋 ASKING BUSINESS QUESTIONS")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/{len(questions)}")
        print(f"{'='*60}")
        
        result = rag.ask(question, verbose=True)
        
        print(f"\n💡 Answer:")
        print(f"{result['answer']}")
        print(f"\n📄 Sources: {', '.join(result['sources'])}")
        
        if i < len(questions):
            input("\nPress Enter for next question...")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE!")
    print("="*60)
    print("\n🎉 Your AI business assistant is ready!")
    print("You can now ask questions about your company documents!")


def interactive():
    """
    Interactive mode: Ask your own questions
    """
    print("="*60)
    print("🦙 INTERACTIVE BUSINESS Q&A")
    print("="*60)
    print("\nType your questions (or 'quit' to exit)")
    
    rag = LlamaRAG()
    
    while True:
        print("\n" + "-"*60)
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        result = rag.ask(question, verbose=False)
        
        print(f"\n💡 Answer:")
        print(f"{result['answer']}")
        print(f"\n📄 Sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive()
    else:
        demo()
