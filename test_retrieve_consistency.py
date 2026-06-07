#!/usr/bin/env python3
"""Test script to check retrieve consistency."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def test_retrieve_consistency():
    """Test if retrieve returns consistent results."""
    print("Testing retrieve consistency...")
    
    # Initialize system with the software engineering laws document
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    
    try:
        rag = RAGSystem(doc_path)
        rag.load_content()
        rag.split_chunks()
        rag.generate_embeddings()
        rag.build_index()
        
        print(f"Loaded {len(rag.chunks)} chunks")
        
        # Test the DRY question multiple times
        question = "explain DRY"
        print(f"\nQuestion: {question}")
        
        for i in range(3):
            print(f"\n=== Call {i+1} ===")
            context = rag.retrieve(question)
            print(f"Retrieved {len(context)} chunks:")
            
            for j, chunk in enumerate(context):
                has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
                print(f"  {j+1}. Has DRY: {has_dry}")
                if j == 0:  # Show first chunk
                    print(f"     {chunk[:100]}...")
        
        # Test with different k values
        print(f"\n=== Testing different k values ===")
        for k in [3, 6, 12, 24]:
            context = rag.retrieve(question, k=k)
            print(f"k={k}: {len(context)} chunks, first has DRY: {'DRY' in context[0] if context else 'N/A'}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieve_consistency()
