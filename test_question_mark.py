#!/usr/bin/env python3
"""Test script to verify if question mark affects retrieval."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def test_question_mark():
    """Test if question mark affects retrieval."""
    print("Testing question mark effect...")
    
    # Initialize system with the software engineering laws document
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    
    try:
        rag = RAGSystem(doc_path)
        rag.load_content()
        rag.split_chunks()
        rag.generate_embeddings()
        rag.build_index()
        
        print(f"Loaded {len(rag.chunks)} chunks")
        
        # Test with and without question mark
        question_with_q = "explain DRY?"
        question_without_q = "explain DRY"
        
        print(f"\n=== With question mark ===")
        context_with_q = rag.retrieve(question_with_q)
        dry_chunks_with_q = [chunk for chunk in context_with_q if "DRY" in chunk or "don't repeat yourself" in chunk.lower()]
        print(f"Retrieved {len(context_with_q)} chunks, {len(dry_chunks_with_q)} with DRY")
        
        print(f"\n=== Without question mark ===")
        context_without_q = rag.retrieve(question_without_q)
        dry_chunks_without_q = [chunk for chunk in context_without_q if "DRY" in chunk or "don't repeat yourself" in chunk.lower()]
        print(f"Retrieved {len(context_without_q)} chunks, {len(dry_chunks_without_q)} with DRY")
        
        # Test the query method
        print(f"\n=== Query method ===")
        answer = rag.query(question_without_q)
        print(f"Answer: {answer[:200]}...")
        
        if "don't repeat yourself" in answer.lower() or "knowledge" in answer.lower():
            print("✅ SUCCESS: Query method works")
            return True
        else:
            print("❌ FAILURE: Query method failed")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_question_mark()
    sys.exit(0 if success else 1)
