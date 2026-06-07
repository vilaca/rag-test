#!/usr/bin/env python3
"""Debug script to understand keyword extraction in re-ranking."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_keyword_extraction():
    """Debug the keyword extraction in re-ranking."""
    print("Debugging keyword extraction...")
    
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
        
        # Extract keywords using the re-ranking method's logic
        import re
        
        # Extract key terms from query (excluding stop words)
        stop_words = {"explain", "describe", "what", "is", "are", "the", "a", "an", "of", "and", "or", "to", "in", "for"}
        query_lower = question.lower()
        
        # Extract multi-word phrases first (like "Conway's Law")
        key_phrases = []
        words = query_lower.split()
        
        print(f"Words: {words}")
        
        # Look for 2-3 word phrases that contain meaningful terms
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            print(f"Checking 2-word phrase: '{phrase}'")
            if any(word not in stop_words for word in phrase.split()):
                key_phrases.append(phrase)
                print(f"  Added phrase: '{phrase}'")
            
            if i < len(words) - 2:
                phrase3 = f"{words[i]} {words[i+1]} {words[i+2]}"
                print(f"Checking 3-word phrase: '{phrase3}'")
                if any(word not in stop_words for word in phrase3.split()):
                    key_phrases.append(phrase3)
                    print(f"  Added phrase: '{phrase3}'")
        
        # Also add single key words
        key_words = [word for word in words if word not in stop_words and len(word) >= 2]
        print(f"Single key words: {key_words}")
        
        # Combine and deduplicate
        key_terms = list(set(key_phrases + key_words))
        print(f"Final key terms: {key_terms}")
        
        # Test if these key terms match DRY chunks
        print(f"\n=== Testing key term matching ===")
        
        # Get a DRY chunk
        dry_chunk = "### 31. DRY (Don't Repeat Yourself)\n\nThe principle is often misread as 'don't repeat code.'"
        
        for term in key_terms:
            if term in dry_chunk.lower():
                print(f"✅ Term '{term}' found in DRY chunk")
            else:
                print(f"❌ Term '{term}' NOT found in DRY chunk")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_keyword_extraction()
