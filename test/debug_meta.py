#!/usr/bin/env python3
"""Debug meta questions."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag_system import RAGSystem

def debug_meta():
    """Debug meta question handling."""
    print("🔍 Debugging Meta Questions")
    print("=" * 40)
    
    # Initialize system
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    rag = RAGSystem(doc_path)
    rag.load_subtitles()
    rag.split_chunks()
    rag.generate_embeddings()
    rag.build_index()
    
    # Test meta questions
    questions = [
        "how many laws are in this document?",
        "what is this document about?",
        "who would benefit from reading this?",
        "what topics does this cover?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        
        # Get context
        context = rag.retrieve(question)
        print(f"Retrieved {len(context)} chunks")
        
        # Get answer
        answer = rag.query(question)
        print(f"Answer: {answer[:100]}...")

if __name__ == "__main__":
    debug_meta()