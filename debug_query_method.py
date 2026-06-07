#!/usr/bin/env python3
"""Debug script to understand what the query method is doing."""

import sys
import os
import re
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_query_method():
    """Debug the query method."""
    print("Debugging query method...")
    
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
        print(f"\nOriginal question: '{question}'")
        
        # Simulate what the query method does
        q_lower = question.lower().strip()
        has_question_word = bool(re.search(
            r"\b(what|how|why|when|where|who|which|are|is|do|does|can|will|should|could)\b",
            q_lower,
        ))
        is_question = q_lower.endswith("?") or has_question_word
        
        print(f"Has question word: {has_question_word}")
        print(f"Ends with ?: {q_lower.endswith('?')}")
        print(f"Is question: {is_question}")
        
        if not is_question and not q_lower.endswith("?"):
            keywords_in_query = rag._query_keywords(q_lower)
            print(f"Keywords in query: {keywords_in_query}")
            print(f"Keyword count: {len(keywords_in_query)}")
            
            if len(keywords_in_query) <= 2:
                modified_question = question.strip() + "?"
                print(f"Modified question: '{modified_question}'")
            else:
                modified_question = question
                print(f"No modification needed")
        else:
            modified_question = question
            print(f"No modification needed")
        
        # Test retrieval with the modified question
        context = rag.retrieve(modified_question)
        print(f"\nRetrieved {len(context)} chunks with modified question")
        dry_chunks = [chunk for chunk in context if "DRY" in chunk or "don't repeat yourself" in chunk.lower()]
        print(f"DRY chunks: {len(dry_chunks)}/{len(context)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_query_method()
