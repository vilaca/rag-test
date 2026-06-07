#!/usr/bin/env python3
"""Test script to verify DRY question handling without context reconstruction."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def test_dry_no_reconstruction():
    """Test the DRY question without context reconstruction."""
    print("Testing DRY question handling without context reconstruction...")
    
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
        
        # Get retrieval results but skip context reconstruction
        import faiss
        import numpy as np
        
        k = 12
        retrieval_k = min(k * 2, len(rag.chunks))
        
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
        
        # Apply all processing steps except context reconstruction
        ranked_chunks, debug_info = rag._re_rank_with_keywords(question, combined_scores, combined_indices)
        reranked_chunks = ranked_chunks[:k * 2]
        
        if rag._should_use_diversity(question):
            reranked_chunks = rag._apply_mmr_diversity(question, reranked_chunks, k * 2)
        
        filtered_chunks = rag._filter_relevant_chunks(question, reranked_chunks)
        filtered_chunks = rag._add_contextual_chunks(question, filtered_chunks, combined_indices, combined_scores)
        
        if rag._should_boost_sections(question) and hasattr(rag, 'chunk_metadata') and rag.chunk_metadata:
            filtered_chunks = rag._apply_section_boosting(question, filtered_chunks, k * 2)
        
        # Skip context reconstruction
        final_chunks = filtered_chunks[:k]
        
        print(f"Retrieved {len(final_chunks)} chunks (no context reconstruction):")
        
        for i, chunk in enumerate(final_chunks):
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:150]}...")
        
        # Now test with context reconstruction
        print(f"\n=== With Context Reconstruction ===")
        if hasattr(rag, 'chunk_metadata') and rag.chunk_metadata:
            reconstructed_chunks = [rag._reconstruct_context(chunk) for chunk in final_chunks]
            
            print(f"After context reconstruction {len(reconstructed_chunks)} chunks:")
            
            for i, chunk in enumerate(reconstructed_chunks):
                has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
                print(f"  {i+1}. Has DRY: {has_dry}")
                if has_dry:
                    print(f"     🎯 {chunk[:150]}...")
                else:
                    print(f"     ❌ {chunk[:150]}...")
        
        # Test the actual query method
        print(f"\n=== Actual Query Method ===")
        answer = rag.query(question)
        print(f"Answer: {answer}")
        
        # Check if the answer is satisfactory
        if "don't repeat yourself" in answer.lower() or "knowledge" in answer.lower():
            print("✅ SUCCESS: Found relevant information about DRY")
            return True
        else:
            print("❌ FAILURE: Did not find relevant information about DRY")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dry_no_reconstruction()
    sys.exit(0 if success else 1)
