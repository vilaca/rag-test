#!/usr/bin/env python3
"""Debug the 'define' question handling."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag_system import RAGSystem

def debug_define():
    """Debug define question handling."""
    print("🔍 Debugging 'define' Question Handling")
    print("=" * 50)
    
    # Initialize system
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    rag = RAGSystem(doc_path)
    rag.load_subtitles()
    rag.split_chunks()
    rag.generate_embeddings()
    rag.build_index()
    
    question = "define Brooks's Law"
    print(f"Question: {question}")
    print()
    
    # Get context
    context = rag.retrieve(question)
    print(f"Retrieved {len(context)} chunks:")
    for i, chunk in enumerate(context[:5]):
        print(f"  {i+1}. {chunk[:100]}...")
    
    # Check if the definition is in the context
    definition_found = False
    for i, chunk in enumerate(context):
        if "Adding people to a late software project" in chunk:
            print(f"\n✅ Definition found in chunk {i+1}")
            print(f"Chunk content: {chunk}")
            definition_found = True
            break
    
    if not definition_found:
        print(f"\n❌ Definition not found in retrieved chunks")
        
        # Search all chunks
        for i, chunk in enumerate(rag.chunks):
            if "Adding people to a late software project" in chunk:
                print(f"Definition found in chunk {i+1} (not retrieved)")
                print(f"Chunk content: {chunk}")
                break
    
    # Get the actual answer
    answer = rag.query(question)
    print(f"\nFinal Answer: {answer}")

if __name__ == "__main__":
    debug_define()