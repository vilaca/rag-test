#!/usr/bin/env python3
"""Test script to verify DRY question handling without filtering."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def test_dry_no_filtering():
    """Test the DRY question without filtering."""
    print("Testing DRY question handling without filtering...")
    
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
        
        # Temporarily disable filtering
        original_filter = rag._filter_relevant_chunks
        rag._filter_relevant_chunks = lambda query, chunks: chunks
        
        context = rag.retrieve(question)
        print(f"Retrieved {len(context)} chunks (no filtering):")
        
        for i, chunk in enumerate(context[:10]):  # Show first 10
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:150]}...")
        
        # Restore original method
        rag._filter_relevant_chunks = original_filter
        
        # Test the actual query
        print(f"\n=== Actual Query ===")
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
    success = test_dry_no_filtering()
    sys.exit(0 if success else 1)
