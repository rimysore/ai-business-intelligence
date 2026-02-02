"""
Document Processing and Embeddings
Convert business documents into searchable vectors
"""

import os
import json
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

class DocumentProcessor:
    """
    Process documents and create embeddings for RAG
    
    Flow:
    1. Load documents
    2. Split into chunks
    3. Create embeddings (vector representations)
    4. Store in ChromaDB for fast retrieval
    """
    
    def __init__(self, persist_directory="data/chroma_db"):
        """
        Initialize document processor
        
        Args:
            persist_directory: Where to save the vector database
        """
        self.persist_directory = persist_directory
        
        # Create embedding model
        # This converts text to 384-dimensional vectors
        print("📦 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✅ Embedding model loaded (dimension: 384)")
        
        # Initialize ChromaDB
        print("🗄️  Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="business_documents",
            metadata={"description": "Company documents for RAG"}
        )
        print(f"✅ ChromaDB ready at: {persist_directory}")
    
    def load_documents(self, documents_path="data/documents"):
        """
        Load all documents from directory
        
        Returns:
            List of document dictionaries
        """
        documents = []
        
        # Load text files
        for filename in os.listdir(documents_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(documents_path, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                documents.append({
                    'filename': filename,
                    'content': content,
                    'doc_type': 'text'
                })
        
        # Load JSON if exists
        json_path = os.path.join(documents_path, 'documents.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_docs = json.load(f)
                for doc in json_docs:
                    if doc['filename'] not in [d['filename'] for d in documents]:
                        documents.append(doc)
        
        return documents
    
    def chunk_document(self, content: str, chunk_size=500, overlap=50):
        """
        Split document into overlapping chunks
        
        Why chunks?
        - Full documents are too long for LLM context
        - Chunks allow precise retrieval
        - Overlap ensures we don't lose context at boundaries
        
        Args:
            content: Full document text
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def process_and_store_documents(self):
        """
        Main function: Load, chunk, embed, and store all documents
        """
        print("\n" + "="*60)
        print("📚 PROCESSING BUSINESS DOCUMENTS")
        print("="*60)
        
        # Load documents
        print("\n1️⃣ Loading documents...")
        documents = self.load_documents()
        print(f"   Found {len(documents)} documents")
        
        # Process each document
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        chunk_id = 0
        
        print("\n2️⃣ Chunking documents...")
        for doc in documents:
            print(f"   Processing: {doc['filename']}")
            chunks = self.chunk_document(doc['content'])
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    'filename': doc['filename'],
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                })
                all_ids.append(f"doc_{chunk_id}")
                chunk_id += 1
        
        print(f"   Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Create embeddings
        print("\n3️⃣ Creating embeddings...")
        embeddings = self.embedding_model.encode(
            all_chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"   Shape: {embeddings.shape}")
        
        # Store in ChromaDB
        print("\n4️⃣ Storing in vector database...")
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        print(f"   ✅ Stored {len(all_chunks)} chunks in ChromaDB")
        
        print("\n" + "="*60)
        print("✅ DOCUMENT PROCESSING COMPLETE!")
        print("="*60)
        print(f"\nDatabase location: {self.persist_directory}")
        print(f"Total chunks: {len(all_chunks)}")
        print(f"Ready for semantic search!")
        
        return len(all_chunks)
    
    def search(self, query: str, n_results: int = 3):
        """
        Search for relevant document chunks
        
        Args:
            query: User's question
            n_results: Number of results to return
        
        Returns:
            List of relevant chunks with metadata
        """
        # Create embedding for query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return results


# Test the system
if __name__ == "__main__":
    # Create processor
    processor = DocumentProcessor()
    
    # Process documents
    num_chunks = processor.process_and_store_documents()
    
    # Test search
    print("\n🔍 Testing semantic search...")
    test_queries = [
        "What is our company mission?",
        "What was the Q4 revenue?",
        "What is the return policy?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = processor.search(query, n_results=2)
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], 
                                                results['metadatas'][0]), 1):
            print(f"\n  Result {i} (from {metadata['filename']}):")
            print(f"  {doc[:200]}...")
    
    print("\n✅ RAG system ready!")
