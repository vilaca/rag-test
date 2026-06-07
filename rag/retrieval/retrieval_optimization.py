"""Retrieval optimization and advanced re-ranking."""

from typing import List
import numpy as np


class RetrievalOptimizationMixin:
    def _get_adaptive_retrieval_k(self, query: str, base_k: int) -> int:
        """Get adaptive retrieval parameter based on query complexity."""
        query_length = len(query.split())
        
        # Longer queries may need more context
        if query_length > 10:
            return min(base_k + 4, 20)
        elif query_length > 6:
            return min(base_k + 2, 15)
        else:
            return base_k
    
    def _should_use_diversity(self, query: str) -> bool:
        """Determine if diversity-based re-ranking should be used."""
        query_lower = query.lower()
        
        # Use diversity for broad questions
        diversity_trigger_phrases = [
            "what are the different", "what are various", "list the different",
            "compare", "contrast", "what are the main types", "what are the categories"
        ]
        
        return any(phrase in query_lower for phrase in diversity_trigger_phrases)
    
    def _apply_mmr_diversity(self, query: str, chunks: List[str], top_k: int) -> List[str]:
        """Apply Maximal Marginal Relevance for diverse retrieval."""
        if len(chunks) <= top_k:
            return chunks
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Get chunk embeddings
        chunk_embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        
        # MMR algorithm
        selected_indices = []
        selected_embeddings = []
        
        # Select first item: most relevant to query
        similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
        most_relevant_idx = np.argmax(similarities)
        selected_indices.append(most_relevant_idx)
        selected_embeddings.append(chunk_embeddings[most_relevant_idx])
        
        # Select remaining items for diversity
        lambda_param = 0.5  # Balance between relevance and diversity
        
        while len(selected_indices) < top_k and len(selected_indices) < len(chunks):
            remaining_indices = [i for i in range(len(chunks)) if i not in selected_indices]
            remaining_embeddings = chunk_embeddings[remaining_indices]
            
            # Calculate MMR scores
            mmr_scores = []
            for i, emb in zip(remaining_indices, remaining_embeddings):
                # Relevance to query
                relevance = np.dot(emb, query_embedding.T)
                
                # Diversity: negative maximum similarity to already selected
                if selected_embeddings:
                    similarities = np.dot(emb, np.array(selected_embeddings).T)
                    diversity = -np.max(similarities)
                else:
                    diversity = 0
                
                # MMR score
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                mmr_scores.append((i, mmr_score))
            
            # Select item with highest MMR score
            if mmr_scores:
                best_idx, _ = max(mmr_scores, key=lambda x: x[1])
                selected_indices.append(best_idx)
                selected_embeddings.append(chunk_embeddings[best_idx])
        
        # Return selected chunks in order
        return [chunks[i] for i in selected_indices]
    
    def _should_boost_sections(self, query: str) -> bool:
        """Determine if section boosting should be applied."""
        query_lower = query.lower()
        
        # Boost sections for specific queries
        boost_phrases = [
            "in the context of", "regarding", "concerning", "about",
            "related to", "pertaining to", "with respect to"
        ]
        
        return any(phrase in query_lower for phrase in boost_phrases)
    
    def _apply_section_boosting(self, query: str, chunks: List[str], top_k: int) -> List[str]:
        """Boost chunks from sections that are more relevant to the query."""
        if not hasattr(self, 'chunk_metadata') or not self.chunk_metadata:
            return chunks
        
        # Extract query keywords
        keywords = self._query_keywords(query)
        if not keywords:
            return chunks
        
        # Score sections based on keyword presence
        section_scores = {}
        
        for metadata in self.chunk_metadata:
            section_idx = metadata['section_index']
            section_title = metadata['section_title']
            chunk_text = metadata['chunk_text']
            
            if section_idx not in section_scores:
                section_scores[section_idx] = {
                    'score': 0,
                    'title': section_title,
                    'chunks': []
                }
            
            # Score based on keyword matches in chunk
            chunk_lower = chunk_text.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in chunk_lower)
            section_scores[section_idx]['score'] += keyword_matches
            section_scores[section_idx]['chunks'].append(len(chunks) - len(section_scores[section_idx]['chunks']) - 1)
        
        # Sort sections by score
        sorted_sections = sorted(section_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Reorder chunks based on section scores
        reordered_chunks = []
        used_indices = set()
        
        for section_idx, section_data in sorted_sections:
            for chunk_idx in section_data['chunks']:
                if chunk_idx not in used_indices and chunk_idx < len(chunks):
                    reordered_chunks.append(chunks[chunk_idx])
                    used_indices.add(chunk_idx)
                    if len(reordered_chunks) >= top_k:
                        break
            if len(reordered_chunks) >= top_k:
                break
        
        # Fill remaining slots with any missing chunks
        for i, chunk in enumerate(chunks):
            if i not in used_indices and len(reordered_chunks) < top_k:
                reordered_chunks.append(chunk)
        
        return reordered_chunks
    
    def _rerank_with_cross_encoder(self, query: str, chunks: List[str], top_k: int = 20) -> List[str]:
        """Re-rank chunks using cross-encoder (if available)."""
        if not hasattr(self, 'cross_encoder'):
            return chunks[:top_k]
        
        # Create query-chunk pairs
        pairs = [[query, chunk] for chunk in chunks]
        
        # Score pairs using cross-encoder
        try:
            scores = self.cross_encoder(pairs)
            
            # Sort by score
            scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            
            # Return top k
            return [chunk for chunk, score in scored_chunks[:top_k]]
        
        except Exception as e:
            print(f"Cross-encoder re-ranking failed: {e}")
            return chunks[:top_k]
    
    def _re_rank_with_keywords(self, query: str, scores, indices) -> tuple:
        """Re-rank results based on keyword matching."""
        if not hasattr(self, 'chunks') or not self.chunks:
            return scores, indices
        
        # Extract keywords from query
        keywords = self._query_keywords(query)
        if not keywords:
            return scores, indices
        
        # Boost scores for chunks containing keywords
        keyword_boosted_scores = []
        
        for score, idx in zip(scores, indices):
            chunk = self.chunks[idx]
            chunk_lower = chunk.lower()
            
            # Count keyword matches
            keyword_matches = sum(1 for keyword in keywords if keyword in chunk_lower)
            
            # Apply boost
            boosted_score = score * (1 + 0.2 * keyword_matches)
            keyword_boosted_scores.append(boosted_score)
        
        # Re-sort based on boosted scores
        combined = list(zip(keyword_boosted_scores, indices))
        combined.sort(key=lambda x: x[0], reverse=True)
        
        new_scores, new_indices = zip(*combined) if combined else ([], [])
        
        return new_scores, new_indices
    
    def _get_diverse_indices(self, indices: List[int], max_count: int = 4) -> List[int]:
        """Get diverse indices from retrieval results."""
        if len(indices) <= max_count:
            return indices
        
        # Simple diversity: select indices from different sections
        if not hasattr(self, 'chunk_metadata') or not self.chunk_metadata:
            return indices[:max_count]
        
        selected_indices = []
        selected_sections = set()
        
        for idx in indices:
            if len(selected_indices) >= max_count:
                break
            
            # Find section for this chunk
            for metadata in self.chunk_metadata:
                if metadata['chunk_text'] == self.chunks[idx]:
                    section_idx = metadata['section_index']
                    if section_idx not in selected_sections:
                        selected_indices.append(idx)
                        selected_sections.add(section_idx)
                    break
        
        return selected_indices
