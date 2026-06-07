#!/usr/bin/env python3
"""Debug script to understand main topic extraction."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_main_topic():
    """Debug the main topic extraction."""
    print("Debugging main topic extraction...")
    
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
        
        # Extract main topic using the same logic as _handle_explanation_question
        query_lower = question.lower()
        main_topic = ""
        if "explain " in query_lower:
            main_topic = query_lower.split("explain ", 1)[1].split()[0] if len(question.split()) > 1 else ""
            
        print(f"Main topic: '{main_topic}'")
        
        # Test candidate extraction
        context = rag.retrieve(question)
        print(f"Retrieved {len(context)} chunks")
        
        candidates = rag._extract_candidate_sentences(context)
        print(f"Extracted {len(candidates)} candidates")
        
        for i, candidate in enumerate(candidates[:5]):
            has_dry = "dry" in candidate.lower()
            has_main_topic = main_topic in candidate.lower()
            print(f"  Candidate {i+1}: Has DRY: {has_dry}, Has '{main_topic}': {has_main_topic}")
            print(f"    {candidate[:100]}...")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_main_topic()
