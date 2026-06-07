#!/usr/bin/env python3
"""Test script to verify DRY question handling without section boosting."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def test_dry_no_section_boosting():
    """Test the DRY question without section boosting."""
    print("Testing DRY question handling without section boosting...")
    
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
        
        # Temporarily disable section boosting
        original_should_boost = rag._should_boost_sections
        rag._should_boost_sections = lambda query: False
        
        context = rag.retrieve(question)
        print(f"Retrieved {len(context)} chunks (no section boosting):")
        
        for i, chunk in enumerate(context):
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:150]}...")
        
        # Restore original method
        rag._should_boost_sections = original_should_boost
        
        # Test with section boosting enabled
        print(f"\n=== With Section Boosting ===")
        context_with_boosting = rag.retrieve(question)
        print(f"Retrieved {len(context_with_boosting)} chunks (with section boosting):")
        
        for i, chunk in enumerate(context_with_boosting):
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:150]}...")
            else:
                print(f"     ❌ {chunk[:150]}...")
        
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
    success = test_dry_no_section_boosting()
    sys.exit(0 if success else 1)
