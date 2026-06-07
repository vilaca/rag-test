#!/usr/bin/env python3
"""Debug script to understand re-ranking scores for DRY chunks."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_reranking_scores():
    """Debug the re-ranking scores for DRY chunks."""
    print("Debugging re-ranking scores...")
    
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
        
        # Get some DRY chunks and their indices
        dry_indices = []
        for i, chunk in enumerate(rag.chunks):
            if "DRY" in chunk or "don't repeat yourself" in chunk.lower():
                dry_indices.append(i)
                if len(dry_indices) >= 5:  # Get 5 DRY chunks
                    break
        
        print(f"Found {len(dry_indices)} DRY chunks at indices: {dry_indices}")
        
        # Simulate the re-ranking process for these chunks
        import numpy as np
        
        # Create dummy scores and indices for the DRY chunks
        dummy_scores = np.array([[1.0, 0.9, 0.8, 0.7, 0.6]])  # Dummy semantic scores
        dummy_indices = np.array([dry_indices])
        
        # Apply re-ranking
        ranked_chunks, debug_info = rag._re_rank_with_keywords(question, dummy_scores, dummy_indices)
        
        print(f"\nRe-ranked {len(ranked_chunks)} chunks:")
        
        for i, debug in enumerate(debug_info):
            chunk_idx = debug['index']
            chunk = rag.chunks[chunk_idx]
            print(f"\nRank {i+1}: Chunk {chunk_idx}")
            print(f"  Original score: {debug['original_score']:.3f}")
            print(f"  Keyword boost: {debug['keyword_boost']:.3f}")
            print(f"  Final score: {debug['final_score']:.3f}")
            print(f"  Preview: {chunk[:100]}...")
            
            # Check if it starts with a heading
            import re
            if re.match(r"^#{1,6}\s", chunk):
                print(f"  📌 Starts with heading")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_reranking_scores()
