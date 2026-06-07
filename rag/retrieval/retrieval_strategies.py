"""Retrieval strategies and hybrid retrieval."""

from typing import List, Tuple
import numpy as np


class RetrievalStrategiesMixin:
    def retrieve(self, query: str, k: int = 12) -> List[str]:
        """Retrieve relevant chunks for a query using hybrid retrieval."""
        if not hasattr(self, 'index') or not hasattr(self, 'bm25'):
            if not hasattr(self, 'index'):
                self.build_index()
            if not hasattr(self, 'bm25'):
                self._initialize_bm25()
        
        # Get adaptive retrieval parameter
        adaptive_k = self._get_adaptive_retrieval_k(query, k)
        
        # Dense retrieval using FAISS
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Get top-k results from dense retrieval
        dense_scores, dense_indices = self.index.search(query_embedding, k * 2)
        
        # BM25 retrieval
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k * 2]
        
        # Combine results using reciprocal rank fusion
        combined_scores, combined_indices = self._combine_retrieval_results(
            dense_scores[0], dense_indices[0], 
            bm25_scores, bm25_indices, 
            adaptive_k
        )
        
        # Get the actual chunks
        retrieved_chunks = [self.chunks[idx] for idx in combined_indices[:adaptive_k]]
        
        # Apply re-ranking and filtering
        retrieved_chunks = self._filter_relevant_chunks(query, retrieved_chunks)
        
        # Apply diversity and section boosting if needed
        if self._should_use_diversity(query):
            retrieved_chunks = self._apply_mmr_diversity(query, retrieved_chunks, adaptive_k)
        
        if self._should_boost_sections(query):
            retrieved_chunks = self._apply_section_boosting(query, retrieved_chunks, adaptive_k)
        
        # Add contextual chunks
        retrieved_chunks = self._add_contextual_chunks(query, retrieved_chunks, combined_indices, combined_scores)
        
        return retrieved_chunks[:adaptive_k]
    
    def _combine_retrieval_results(self, dense_scores, dense_indices, bm25_scores, bm25_indices, k):
        """Combine dense and sparse retrieval results using RRF."""
        # Create combined score dictionary
        combined_scores = {}
        
        # Add dense retrieval results
        for rank, (score, idx) in enumerate(zip(dense_scores, dense_indices), 1):
            if idx not in combined_scores:
                combined_scores[idx] = 0
            combined_scores[idx] += 1 / (60 + rank)
        
        # Add BM25 results
        for rank, idx in enumerate(bm25_indices, 1):
            if idx not in combined_scores:
                combined_scores[idx] = 0
            combined_scores[idx] += 1 / (60 + rank)
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top k indices and scores
        top_indices = [idx for idx, score in sorted_results[:k * 2]]
        top_scores = [score for idx, score in sorted_results[:k * 2]]
        
        return top_scores, top_indices
