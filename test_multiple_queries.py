#!/usr/bin/env python3
"""Test script to verify multiple DRY queries."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def test_multiple_queries():
    """Test multiple DRY queries."""
    print("Testing multiple DRY queries...")
    
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
        
        for i in range(3):
            print(f"\n=== Query {i+1} ===")
            print(f"Question: {question}")
            
            answer = rag.query(question)
            print(f"Answer: {answer[:200]}...")
            
            # Check if the answer is satisfactory
            if "don't repeat yourself" in answer.lower() or "knowledge" in answer.lower():
                print("✅ SUCCESS: Found relevant information about DRY")
            else:
                print("❌ FAILURE: Did not find relevant information about DRY")
        
        # Test retrieval directly to see if it's consistent
        print(f"\n=== Direct Retrieval ===")
        context = rag.retrieve(question)
        dry_chunks = [chunk for chunk in context if "DRY" in chunk or "don't repeat yourself" in chunk.lower()]
        print(f"Retrieved {len(context)} chunks, {len(dry_chunks)} with DRY")
        
        if dry_chunks:
            print("✅ Retrieval is working")
        else:
            print("❌ Retrieval is not working")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multiple_queries()
