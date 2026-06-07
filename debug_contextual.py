#!/usr/bin/env python3
"""Debug script to understand contextual chunk addition."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_contextual():
    """Debug the contextual chunk addition."""
    print("Debugging contextual chunk addition...")
    
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
        
        # Get retrieval results step by step
        context = rag.retrieve(question)
        print(f"Final retrieval returned {len(context)} chunks:")
        
        for i, chunk in enumerate(context):
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"\nChunk {i+1}:")
            print(f"  Has DRY: {has_dry}")
            print(f"  Length: {len(chunk)} characters")
            print(f"  Preview: {chunk[:200]}...")
            
            # Check if it starts with a heading
            import re
            if re.match(r"^#{2,6}\s", chunk):
                print(f"  📌 Starts with heading")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_contextual()
