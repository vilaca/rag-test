#!/usr/bin/env python3
"""Test script to verify DRY question handling."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def test_dry_question():
    """Test the DRY question specifically."""
    print("Testing DRY question handling...")
    
    # Initialize system with the software engineering laws document
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    
    try:
        rag = RAGSystem(doc_path)
        rag.load_content()
        rag.split_chunks()
        rag.generate_embeddings()
        rag.build_index()
        
        print(f"Loaded {len(rag.chunks)} chunks")
        
        # Test the DRY question
        question = "explain DRY"
        print(f"\nQuestion: {question}")
        
        answer = rag.query(question)
        print(f"Answer: {answer}")
        
        # Check if the answer is satisfactory
        if "don't repeat yourself" in answer.lower() or "knowledge" in answer.lower():
            print("✅ SUCCESS: Found relevant information about DRY")
            return True
        else:
            print("❌ FAILURE: Did not find relevant information about DRY")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dry_question()
    sys.exit(0 if success else 1)
