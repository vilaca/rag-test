#!/usr/bin/env python3
"""Test script to verify DRY retrieval directly."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def test_retrieval_direct():
    """Test the DRY retrieval directly."""
    print("Testing DRY retrieval directly...")
    
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
        
        # Test retrieval directly
        context = rag.retrieve(question)
        print(f"Retrieved {len(context)} chunks:")
        
        for i, chunk in enumerate(context):
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:150]}...")
            else:
                print(f"     ❌ {chunk[:150]}...")
        
        # Test if we have any DRY chunks
        dry_chunks = [chunk for chunk in context if "DRY" in chunk or "don't repeat yourself" in chunk.lower()]
        print(f"\nDRY chunks found: {len(dry_chunks)}/{len(context)}")
        
        if dry_chunks:
            print("✅ SUCCESS: Retrieval found DRY chunks")
            
            # Test answer generation with the retrieved context
            print(f"\n=== Testing Answer Generation ===")
            answer = rag.generate_answer(question, context)
            print(f"Answer: {answer}")
            
            if "don't repeat yourself" in answer.lower() or "knowledge" in answer.lower():
                print("✅ SUCCESS: Answer generation worked")
                return True
            else:
                print("❌ FAILURE: Answer generation failed")
                return False
        else:
            print("❌ FAILURE: Retrieval did not find DRY chunks")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_retrieval_direct()
    sys.exit(0 if success else 1)
