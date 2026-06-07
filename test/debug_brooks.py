#!/usr/bin/env python3
"""Debug Brooks's Law retrieval issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag_system import RAGSystem

def debug_brooks():
    """Debug Brooks's Law retrieval."""
    print("🔍 Debugging Brooks's Law Retrieval")
    print("=" * 50)
    
    # Initialize system
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    rag = RAGSystem(doc_path)
    rag.load_subtitles()
    rag.split_chunks()
    rag.generate_embeddings()
    rag.build_index()
    
    # Test Brooks's Law questions
    questions = [
        "define Brooks's Law",
        "what is Brooks's Law?",
        "explain Brooks's Law"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        
        # Get retrieval results
        context = rag.retrieve(question)
        
        print(f"Retrieved {len(context)} chunks:")
        for i, chunk in enumerate(context[:5]):
            print(f"  {i+1}. {chunk[:100]}...")
        
        # Get the answer
        answer = rag.query(question)
        print(f"\nAnswer: {answer}")
        print("=" * 50)

if __name__ == "__main__":
    debug_brooks()