#!/usr/bin/env python3
"""Debug script to trace the full retrieve process."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_full_retrieve():
    """Debug the full retrieve process step by step."""
    print("Debugging full retrieve process...")
    
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
        
        # Manually trace through the retrieve method
        k = 12  # default k
        
        # Step 1: Hybrid retrieval
        print(f"\n=== Step 1: Hybrid Retrieval ===")
        retrieval_k = min(k * 2, len(rag.chunks))  # adaptive retrieval
        
        import faiss
        import numpy as np
        
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
        
        all_chunks = [rag.chunks[idx] for idx in combined_indices[0] if idx < len(rag.chunks)]
        print(f"Combined retrieval: {len(all_chunks)} chunks")
        
        # Step 2: Re-ranking
        print(f"\n=== Step 2: Re-ranking ===")
        ranked_chunks, debug_info = rag._re_rank_with_keywords(question, combined_scores, combined_indices)
        reranked_chunks = ranked_chunks[:k * 2]
        print(f"After re-ranking: {len(reranked_chunks)} chunks")
        
        # Step 3: Diversity reranking (MMR)
        print(f"\n=== Step 3: Diversity Reranking ===")
        if rag._should_use_diversity(question):
            reranked_chunks = rag._apply_mmr_diversity(question, reranked_chunks, k * 2)
            print(f"After MMR diversity: {len(reranked_chunks)} chunks")
        else:
            print(f"MMR diversity not applied")
        
        # Step 4: Filtering
        print(f"\n=== Step 4: Filtering ===")
        filtered_chunks = rag._filter_relevant_chunks(question, reranked_chunks)
        print(f"After filtering: {len(filtered_chunks)} chunks")
        
        # Step 5: Contextual chunks
        print(f"\n=== Step 5: Contextual Chunks ===")
        filtered_chunks = rag._add_contextual_chunks(question, filtered_chunks, combined_indices, combined_scores)
        print(f"After contextual chunks: {len(filtered_chunks)} chunks")
        
        # Step 6: Section boosting
        print(f"\n=== Step 6: Section Boosting ===")
        if rag._should_boost_sections(question):
            print(f"Section boosting triggered for 'explain' query")
            filtered_chunks = rag._apply_section_boosting(question, filtered_chunks, k * 2)
            print(f"After section boosting: {len(filtered_chunks)} chunks")
        else:
            print(f"Section boosting not applied")
        
        # Step 7: Context reconstruction
        print(f"\n=== Step 7: Context Reconstruction ===")
        if hasattr(rag, 'chunk_metadata') and rag.chunk_metadata:
            print(f"Context reconstruction applied")
            # Don't actually reconstruct to see the raw chunks
            print(f"Would reconstruct {len(filtered_chunks)} chunks")
        else:
            print(f"Context reconstruction not applied")
        
        # Final result
        print(f"\n=== Final Result ===")
        final_chunks = filtered_chunks[:k]
        print(f"Final {len(final_chunks)} chunks:")
        
        for i, chunk in enumerate(final_chunks):
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:150]}...")
            else:
                print(f"     ❌ {chunk[:150]}...")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_full_retrieve()
