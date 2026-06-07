#!/usr/bin/env python3
"""Debug script to understand filtering scores."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_filtering_scores():
    """Debug the filtering scores for DRY chunks."""
    print("Debugging filtering scores...")
    
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
        
        # Get some DRY chunks manually
        dry_chunks = []
        for i, chunk in enumerate(rag.chunks):
            if "DRY" in chunk or "don't repeat yourself" in chunk.lower():
                dry_chunks.append(chunk)
                if len(dry_chunks) >= 5:  # Get 5 DRY chunks
                    break
        
        print(f"Testing {len(dry_chunks)} DRY chunks:")
        
        # Extract keywords
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "explain", "describe", "what"}
        query_keywords = [word.lower() for word in question.split() if word.lower() not in stop_words and len(word) >= 2]
        
        print(f"Query keywords: {query_keywords}")
        
        # Score each DRY chunk
        scored_chunks = []
        for i, chunk in enumerate(dry_chunks):
            chunk_lower = chunk.lower()
            
            # Count keyword matches
            keyword_matches = sum(1 for keyword in query_keywords if keyword.lower() in chunk_lower)
            
            # Calculate relevance score
            relevance_score = keyword_matches / len(query_keywords)
            
            # Check for related terms
            related_score = 1.0 if rag._contains_related_terms(chunk_lower, query_keywords) else 0.0
            
            # Combined score
            final_score = relevance_score + (related_score * 0.3)
            
            scored_chunks.append((final_score, chunk))
            
            print(f"\nChunk {i+1}:")
            print(f"  Keyword matches: {keyword_matches}/{len(query_keywords)}")
            print(f"  Relevance score: {relevance_score:.3f}")
            print(f"  Related score: {related_score:.3f}")
            print(f"  Final score: {final_score:.3f}")
            print(f"  Preview: {chunk[:100]}...")
        
        # Calculate threshold
        if scored_chunks:
            top_score = scored_chunks[0][0]
            threshold = top_score * 0.4  # 40% for "explain" queries
            threshold = max(0.2, threshold)
            
            print(f"\n=== Filtering Results ===")
            print(f"Top score: {top_score:.3f}")
            print(f"Threshold (40%): {threshold:.3f}")
            
            filtered_chunks = [chunk for score, chunk in scored_chunks if score >= threshold]
            print(f"Filtered chunks: {len(filtered_chunks)}/{len(scored_chunks)}")
            
            for score, chunk in scored_chunks:
                passes = score >= threshold
                print(f"  Score {score:.3f} >= {threshold:.3f}: {passes}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_filtering_scores()
