#!/usr/bin/env python3
"""Debug script to understand DRY retrieval issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_dry_retrieval():
    """Debug the DRY retrieval process."""
    print("Debugging DRY retrieval...")
    
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
        
        # Get retrieval results directly
        context = rag.retrieve(question)
        print(f"Retrieved {len(context)} chunks:")
        
        for i, chunk in enumerate(context[:5]):  # Show first 5 chunks
            print(f"\nChunk {i+1} (first 200 chars):")
            print(chunk[:200] + "...")
            
            # Check if this chunk contains DRY-related content
            if "DRY" in chunk or "Don't Repeat Yourself" in chunk or "don't repeat yourself" in chunk:
                print("🎯 This chunk contains DRY content!")
            elif "dry" in chunk.lower():
                print("⚠️  This chunk contains 'dry' (lowercase)")
        
        if not context:
            print("No context retrieved - this is a retrieval failure")
        else:
            print(f"\nRetrieval found {len(context)} chunks, but answer generation failed")
            
            # Try to generate answer with debug
            try:
                answer = rag.generate_answer(question, context)
                print(f"Generated answer: {answer}")
            except Exception as e:
                print(f"Answer generation error: {e}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dry_retrieval()
