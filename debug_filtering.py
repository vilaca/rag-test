#!/usr/bin/env python3
"""Debug script to understand the filtering step."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_filtering():
    """Debug the filtering process step by step."""
    print("Debugging filtering process...")
    
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
        
        # Get the combined results before filtering
        import faiss
        import numpy as np
        
        # Simulate the retrieval process step by step
        retrieval_k = 12  # default k
        
        # Dense retrieval
        query_embedding = rag.embedding_model.encode([question], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)
        dense_scores, dense_indices = rag.index.search(query_embedding, retrieval_k)
        
        # Sparse retrieval
        tokenized_query = question.split()
        bm25_scores = rag.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:retrieval_k]
        
        # Combine results
        combined_scores, combined_indices = rag._combine_retrieval_results(
            dense_scores, dense_indices, bm25_scores, bm25_indices, retrieval_k
        )
        
        # Get combined chunks
        all_chunks = [rag.chunks[idx] for idx in combined_indices[0] if idx < len(rag.chunks)]
        
        print(f"Combined retrieval found {len(all_chunks)} chunks:")
        for i, chunk in enumerate(all_chunks):
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:100]}...")
        
        # Now apply keyword-based re-ranking (this is what happens when no cross-encoder is available)
        print(f"\n=== Applying keyword-based re-ranking ===")
        ranked_chunks, debug_info = rag._re_rank_with_keywords(question, combined_scores, combined_indices)
        
        print(f"After keyword re-ranking: {len(ranked_chunks)} chunks")
        for i, chunk in enumerate(ranked_chunks[:10]):
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:100]}...")
        
        # Now apply filtering
        print(f"\n=== Applying filtering ===")
        filtered_chunks = rag._filter_relevant_chunks(question, ranked_chunks[:20])  # Top 20 from re-ranking
        
        print(f"After filtering: {len(filtered_chunks)} chunks")
        for i, chunk in enumerate(filtered_chunks):
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:100]}...")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_filtering()
