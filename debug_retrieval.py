#!/usr/bin/env python3
"""Debug script to understand retrieval failure for DRY."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem
import numpy as np

def debug_retrieval():
    """Debug the retrieval process step by step."""
    print("Debugging retrieval process...")
    
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
        
        # Step 1: Check BM25 retrieval
        print("\n=== BM25 Retrieval ===")
        tokenized_query = question.split()
        bm25_scores = rag.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:10]  # Top 10 from BM25
        
        print(f"BM25 top results:")
        for i, idx in enumerate(bm25_indices):
            chunk = rag.chunks[idx]
            score = bm25_scores[idx]
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Score: {score:.3f} - Chunk {idx} - Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:150]}...")
        
        # Step 2: Check dense retrieval
        print("\n=== Dense Retrieval (FAISS) ===")
        import faiss
        query_embedding = rag.embedding_model.encode([question], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)
        dense_scores, dense_indices = rag.index.search(query_embedding, 10)
        
        print(f"Dense retrieval top results:")
        for i, idx in enumerate(dense_indices[0]):
            chunk = rag.chunks[idx]
            score = dense_scores[0][i]
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Score: {score:.3f} - Chunk {idx} - Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:150]}...")
        
        # Step 3: Check what the actual retrieve method returns
        print("\n=== Full retrieve() method ===")
        context = rag.retrieve(question)
        print(f"Final retrieval returned {len(context)} chunks:")
        for i, chunk in enumerate(context):
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:150]}...")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_retrieval()
