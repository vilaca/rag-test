#!/usr/bin/env python3
"""Test script to verify DRY question handling without hallucination guard in generate_answer."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def test_no_guard_in_generate():
    """Test the DRY question without hallucination guard in generate_answer."""
    print("Testing DRY question handling without hallucination guard in generate_answer...")
    
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
        
        # Get context
        context = rag.retrieve(question)
        print(f"Retrieved {len(context)} chunks")
        
        # Test generate_answer directly with the context
        answer = rag._synthesize_answer(question, context, "explanatory")
        print(f"Synthesized answer: {answer[:200]}...")
        
        # Check if the answer is satisfactory
        if "don't repeat yourself" in answer.lower() or "knowledge" in answer.lower():
            print("✅ SUCCESS: Synthesized answer contains DRY information")
            
            # Test the full generate_answer method
            full_answer = rag.generate_answer(question, context)
            print(f"Full answer: {full_answer[:200]}...")
            
            if "don't repeat yourself" in full_answer.lower() or "knowledge" in full_answer.lower():
                print("✅ SUCCESS: Full generate_answer works")
                return True
            else:
                print("❌ FAILURE: Full generate_answer failed")
                return False
        else:
            print("❌ FAILURE: Synthesized answer doesn't contain DRY information")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_no_guard_in_generate()
    sys.exit(0 if success else 1)
